from __future__ import annotations

import functools
import json
import logging
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.arg_groups.model_override_base import (
    ep_scale_joiner_of,
    resolving_view,
)
from sglang.srt.distributed import (
    GroupCoordinator,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    derive_attention_ranks,
    derive_attn_tp_size,
    get_device,
    get_exec,
    get_flags,
    get_forward,
    get_parallel,
    get_resources,
    get_stream,
)
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def deployment_attn_dp_size() -> int:
    """Return the deployment's attention-DP replica count.

    Draft scopes retain this count because their metadata gathers include
    the target's replicas.
    """
    parallel = get_parallel()
    return parallel.num_dp_ranks if parallel.attn_dp_enabled else 1


def dp_gather_width() -> int:
    """Return the DP gather width.

    After elastic scale-up, the gather spans the expanded WORLD; otherwise
    it spans the attention-DP replicas. Inside a draft scope that owns its
    attention, it is still the target's gather (see patch_tensor_parallel_group).
    """
    parallel = get_parallel()
    if world_dp_gather_enabled():
        return parallel.num_dp_ranks
    if get_flags().dp.scoped_gather_slot is not None:
        return deployment_attn_dp_size()
    return parallel.attn_dp_size


def dp_slot_in(per_rank) -> int:
    """Return this process's slot in a per-DP-replica sequence.

    The sequence carries one entry per replica of the gather this process
    takes part in, so its length is the gather width. Length one is the
    all-gather-skipped batch, which carries this process's entry alone.
    """
    if len(per_rank) == 1:
        return 0
    width = dp_gather_width()
    if len(per_rank) != width:
        raise ValueError(
            f"a per-replica sequence of {len(per_rank)} entries does not "
            f"belong to a DP gather of width {width}"
        )
    return dp_gather_slot()


def dp_gather_slot() -> int:
    """Return this process's index in the DP gather.

    After elastic scale-up, use the TP rank plus the join offset; otherwise
    use the attention-DP rank. Inside a draft scope that owns its attention,
    it is still the slot in the target's gather (see patch_tensor_parallel_group).
    """
    scoped = get_flags().dp.scoped_gather_slot
    if scoped is not None:
        return scoped
    parallel = get_parallel()
    if world_dp_gather_enabled():
        return parallel.tp_rank + parallel.ep_join_rank_offset
    return parallel.attn_dp_rank


def world_dp_gather_enabled() -> bool:
    """Whether DP gathers should use expanded WORLD after joiner admission."""
    dp = get_flags().dp
    return dp.use_world_group_for_gather and not dp.joiner_skip_all_gather


def enable_joiner_all_gather():
    get_flags().dp.joiner_skip_all_gather = False


def update_dp_attention_post_scale(new_dp_size: int, new_dp_rank: int):
    """Switch DP gathers to the expanded WORLD.

    The caller updates the configured widths; these arguments identify the
    scale-up in the log.
    """
    get_flags().dp.use_world_group_for_gather = True
    logger.debug(
        "[Elastic EP] dp_attention switched to WORLD: num_dp_ranks=%d dp_rank=%d",
        new_dp_size,
        new_dp_rank,
    )


_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")
_is_cpu = is_cpu()


class DpPaddingMode(IntEnum):
    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        gather_width = dp_gather_width()

        # (trangdough) pplx-kernels a2a is a symmetric collective: every EP rank
        # must dispatch the same number of tokens or the device-side handshake
        # deadlocks (idle DP ranks with 0 tokens never signal their peers).
        # Force MAX_LEN so all ranks are padded to equal token counts.
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        moe_a2a_backend = get_moe_a2a_backend()
        if moe_a2a_backend.is_pplx():
            return DpPaddingMode.MAX_LEN

        if moe_a2a_backend.is_deepep_v2() and envs.SGLANG_DEEPEP_V2_FORCE_MAX_LEN.get():
            return DpPaddingMode.MAX_LEN

        # When is_extend_in_batch and the gather spans several DP ranks, use
        # SUM_LEN to avoid padding overhead from uneven token distribution.
        # Over one rank, max_len equals sum_len, so prefer MAX_LEN mode
        # to enable symmetric memory optimization (needed for DSA CP, etc.).
        if is_extend_in_batch and gather_width > 1:
            # Hybrid-SSM models materialize idle ranks via the MAX_LEN
            # fabricated-row conversion; other models keep mainline SUM_LEN.
            if get_flags().dp.max_len_with_idle and min(global_num_tokens) == 0:
                return DpPaddingMode.MAX_LEN
            return DpPaddingMode.SUM_LEN

        # we choose the mode that minimizes the communication cost
        # prefer MAX_LEN when communication cost is equal to enable symmetric memory
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 >= max_len * gather_width:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:
    """Facade for the DP gathered-buffer state: allocation metadata lives on
    ``flags.dp`` (set once at initialize_dp_attention). The per-forward
    sizing quartet stays as class attributes: the values are read inside
    torch.compile-traced model code, and attribute-source ints get dynamo's
    automatic-dynamic treatment, while contextvars are untraceable and dict
    slots value-guard into the recompile limit (one recompile per distinct
    size)."""

    # Real defaults (not bare annotations): the sizing quartet is overwritten
    # per-forward by set_dp_buffer_len, but callers that run before the first
    # forward — notably the load-time mhc_pre prewarm, which has no ForwardBatch
    # yet — read _dp_max_padding via is_allocation_symmetric(). A bare
    # annotation creates no class attribute, so those reads raised
    # AttributeError. Defaulting _dp_max_padding to False (non-symmetric) is
    # safe for prewarm: it only JIT-compiles kernels and never enters a real
    # all-reduce, so the symmetric pool is not needed there.
    _global_dp_buffer_len: int = 0
    _local_dp_buffer_len: int = 0
    _dp_max_padding: bool = False
    _global_num_tokens: Optional[List[int]] = None
    _global_num_tokens_gpu: Optional[torch.Tensor] = None

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):

        dp = get_flags().dp
        dp.buffer_hidden_size = hidden_size
        dp.buffer_dtype = dtype
        dp.buffer_device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        dp_max_padding: bool,
        global_num_tokens: Optional[List[int]] = None,
        global_num_tokens_gpu: Optional[torch.Tensor] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._dp_max_padding = dp_max_padding
        cls._global_num_tokens = global_num_tokens
        cls._global_num_tokens_gpu = global_num_tokens_gpu

    @classmethod
    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._global_dp_buffer_len, dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_local_dp_buffer(
        cls, group: GroupCoordinator, hidden_size: Optional[int] = None
    ) -> torch.Tensor:

        dp = get_flags().dp
        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
            buffer = torch.empty(
                (cls._local_dp_buffer_len, hidden_size or dp.buffer_hidden_size),
                dtype=dp.buffer_dtype,
                device=dp.buffer_device,
            )
        return buffer

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def set_local_dp_buffer_len(cls, local_dp_buffer_len: int) -> None:
        cls._local_dp_buffer_len = local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_global_num_tokens_gpu(cls) -> Optional[torch.Tensor]:
        return cls._global_num_tokens_gpu

    @classmethod
    def get_dp_hidden_size(cls) -> int:

        return get_flags().dp.buffer_hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:

        return get_flags().dp.buffer_dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:

        return get_flags().dp.buffer_device

    @classmethod
    def is_dp_max_padding(cls) -> bool:
        return cls._dp_max_padding


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    dp_max_padding: bool,
    global_num_tokens: Optional[List[int]] = None,
    global_num_tokens_gpu: Optional[torch.Tensor] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len,
        local_dp_buffer_len,
        dp_max_padding,
        global_num_tokens,
        global_num_tokens_gpu,
    )


def get_global_dp_buffer(group: GroupCoordinator) -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer(group=group)


def get_local_dp_buffer(
    group: GroupCoordinator, hidden_size: Optional[int] = None
) -> torch.Tensor:
    """A buffer for this rank's local DP rows, ``hidden_size`` wide (the model's
    hidden size by default)."""
    return _DpGatheredBufferWrapper.get_local_dp_buffer(
        group=group, hidden_size=hidden_size
    )


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def set_local_dp_buffer_len(local_dp_buffer_len: int) -> None:
    _DpGatheredBufferWrapper.set_local_dp_buffer_len(local_dp_buffer_len)


def set_dp_buffer_len_from_batch(forward_batch: ForwardBatch) -> None:
    """Publish the DP gather sizes ``forward_batch`` carries: the buffer
    length, the per-rank token counts as padded for the gather, this rank's
    entry, and the padding mode. Capture batches carry no separately padded
    list, so the raw counts stand in for it."""
    global_num_tokens = forward_batch.global_num_tokens_padded_cpu
    if global_num_tokens is None:
        global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = get_parallel().attn_dp_rank if len(global_num_tokens) > 1 else 0
    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        global_num_tokens[dp_rank],
        forward_batch.dp_padding_mode.is_max_len(),
        global_num_tokens,
        forward_batch.global_num_tokens_gpu,
    )


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    # Sticky within the thread: every ForwardBatch construction writes it,
    # graph runners force False around capture; readers are the EP
    # dispatchers on the same (single) forward thread.

    get_forward().set("is_extend_in_batch", is_extend_in_batch)


def get_is_extend_in_batch() -> bool:

    return get_forward().is_extend_in_batch


def is_dp_max_padding() -> bool:
    return _DpGatheredBufferWrapper.is_dp_max_padding()


def compute_dp_attention_world_info(
    tp_rank, tp_size, attn_dp_size, attn_cp_size: int = 1
):
    """This rank's place in the attention topology, plus the widths it sits in.

    The attention-TP width comes from `derive_attn_tp_size`; what this adds is
    the two ranks, which are per-process and so are not among the widths
    `override_permanently` records.
    """
    attn_tp_size = derive_attn_tp_size(
        tp_size=tp_size, attn_cp_size=attn_cp_size, attn_dp_size=attn_dp_size
    )
    attn_tp_rank, attn_dp_rank = derive_attention_ranks(
        tp_rank=tp_rank, attn_tp_size=attn_tp_size, attn_cp_size=attn_cp_size
    )
    return attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size


def initialize_dp_attention_flags(server_args: ServerArgs):
    """Initialize DP runtime flags without changing the worker's placement."""
    dp = get_flags().dp
    dp.enabled = get_parallel().attn_dp_enabled

    if get_exec().moe.elastic_ep_backend is not None and get_parallel().max_ep_size:
        if ep_scale_joiner_of(resolving_view(server_args)):
            dp.joiner_skip_all_gather = True


def initialize_dp_attention(server_args: ServerArgs):
    """Initialize DP flags and state placement from the published topology."""
    initialize_dp_attention_flags(server_args)
    parallel = get_parallel()
    _, _, attn_dp_rank, attn_dp_size = compute_dp_attention_world_info(
        parallel.tp_rank,
        parallel.tp_size,
        parallel.attn_dp_size,
        parallel.attn_cp_size,
    )
    parallel.override_permanently(attn_dp_size=attn_dp_size, attn_dp_rank=attn_dp_rank)


def init_dp_gathered_buffer(model_config: ModelConfig):
    """Size the gathered buffer from the model this worker is about to run."""
    get_flags().dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
    )
    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(get_device().device),
    )


def is_dp_attention_enabled() -> bool:
    return get_flags().dp.enabled


def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
    if forward_batch.dp_local_start_pos is None:
        dp_rank = dp_slot_in(forward_batch.global_num_tokens_gpu)
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = dp_slot_in(global_num_tokens)
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens


from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton
from sglang.srt.distributed.utils import all_gather_single


# TODO: write c++ kernel for cpu
def memcpy_cpu(dst, src, dim, offset, sz, offset_src):
    assert dim == 0, "Only dim=0 supported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same trailing shape"

    total_rows_dst, total_rows_src = dst.shape[0], src.shape[0]
    dst_start, src_start = 0, 0

    if offset_src:
        # src[offset:] → dst[0:]
        src_start = offset
        dst_start = 0
    else:
        # src[0:] → dst[offset:]
        src_start = 0
        dst_start = offset

    dst_end = min(dst_start + sz, total_rows_dst)
    src_end = min(src_start + sz, total_rows_src)
    actual_sz = min(dst_end - dst_start, src_end - src_start)

    if actual_sz <= 0:
        return

    dst[dst_start : dst_start + actual_sz].copy_(src[src_start : src_start + actual_sz])


memcpy_func = memcpy_cpu if _is_cpu else memcpy_triton


def memcpy(dst, src, dim, offset, sz, offset_src):
    memcpy_func(dst, src, dim, offset, sz, offset_src)


def _cp_shard_rows(
    forward_batch: ForwardBatch, cp_shard_counts: Sequence[int]
) -> Tuple[int, int]:
    """(start, length) of this rank's CP shard in the gathered buffer: the CP ranks'
    shards lie back to back, in CP rank order, at the start of their DP slot."""
    cp_rank = get_parallel().attn_cp_rank
    dp_start = sum(forward_batch.global_num_tokens_cpu[: dp_gather_slot()])
    return dp_start + sum(cp_shard_counts[:cp_rank]), cp_shard_counts[cp_rank]


def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    # CP ranks hold the same rows of their DP group, and CP rank 0 writes them,
    # unless they pass the counts of their shards; then each writes its own.
    writes = (is_partial or get_parallel().attn_tp_rank == 0) and (
        cp_shard_counts is not None or get_parallel().attn_cp_rank == 0
    )

    if local_tokens.shape[0] > 0 and writes:
        assert local_tokens.untyped_storage() is not global_tokens.untyped_storage(), (
            "aliasing between global_tokens and local_tokens not allowed"
        )

        if cp_shard_counts is None:
            memcpy(
                global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
            )
        else:
            start, length = _cp_shard_rows(forward_batch, cp_shard_counts)
            global_tokens[start : start + length].copy_(local_tokens[:length])

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    if world_dp_gather_enabled():
        torch.distributed.all_reduce(
            global_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    else:
        NUM_GPUS_PER_NODE = 8
        if (
            not local_tokens.dtype.is_floating_point
            and get_parallel().tp_size <= NUM_GPUS_PER_NODE
        ):
            from sglang.srt.distributed.parallel_state import inplace_all_reduce

            inplace_all_reduce(
                global_tokens, group_name=get_parallel().tp_group.unique_name
            )

        else:
            global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    use_world = world_dp_gather_enabled()

    if get_parallel().attn_tp_size == 1:
        if use_world:
            all_gather_single(
                global_tokens,
                local_tokens,
                group=torch.distributed.group.WORLD,
            )
        else:
            get_parallel().tp_group.all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_parallel().attn_tp_rank != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(get_parallel().attn_tp_size)[
        get_parallel().attn_tp_rank
    ]
    get_parallel().attn_tp_group.reduce_scatter_tensor(
        scattered_local_tokens, local_tokens
    )
    if use_world:
        all_gather_single(
            global_tokens,
            scattered_local_tokens,
            group=torch.distributed.group.WORLD,
        )
    else:
        get_parallel().tp_group.all_gather_into_tensor(
            global_tokens, scattered_local_tokens
        )


# Variable-length DP-MoE gather (reference https://github.com/ROCm/ATOM/pull/930): instead of padding every
# rank to max_len (all_gather) or all-reducing a sum_len zero-buffer (all_reduce),
# gather exactly sum(per-rank tokens) via all_gatherv. Env-gated; only the simple
# tp_size==attn_dp_size (attn_tp_size==1) case is supported for now (e.g. tp8, attn_dp8).
_USE_DP_GATHERV = get_bool_env_var("SGLANG_DP_USE_GATHERV")

_DP_GATHER_FP8_GROUP = 128
# Grow-only gathered fp8 payload / scales buffers, keyed by device.
_dp_gather_fp8_bufs: dict = {}


@functools.lru_cache(maxsize=1)
def _use_dp_gather_fp8() -> bool:
    return envs.SGLANG_ENABLE_DP_GATHER_FP8.get()


def _get_dp_gather_fp8_bufs(rows: int, hidden: int, device: torch.device):
    key = str(device)
    bufs = _dp_gather_fp8_bufs.get(key)
    if bufs is None or bufs[0].shape[0] < rows:
        bufs = (
            torch.empty((rows, hidden), dtype=torch.uint8, device=device),
            torch.empty(
                (rows, hidden // _DP_GATHER_FP8_GROUP),
                dtype=torch.float32,
                device=device,
            ),
        )
        _dp_gather_fp8_bufs[key] = bufs
    return bufs[0][:rows], bufs[1][:rows]


@triton.jit
def _dequant_per_token_group_fp8_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # HIDDEN may not be a multiple of BLOCK (e.g. DeepSeek 7168 vs BLOCK
    # 2048): the tail iteration must be masked or it reads/writes up to
    # BLOCK-1 elements past the row (cross-row corruption + OOB on the last
    # row).  HIDDEN is constexpr, so the mask folds away when it divides.
    for start in tl.static_range(0, HIDDEN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HIDDEN
        qv = tl.load(q_ptr + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + row * NGROUPS + offs // GROUP, mask=mask, other=0.0)
        tl.store(out_ptr + row * HIDDEN + offs, (qv * sv).to(tl.bfloat16), mask=mask)


@triton.jit
def _mask_dp_pad_topk_ids_kernel(
    topk_ids_ptr,
    counts_ptr,
    max_len,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rank = row // max_len
    pos = row % max_len
    valid = pos < tl.load(counts_ptr + rank)
    if valid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(topk_ids_ptr + row * TOPK + offs, -1, mask=offs < TOPK)


def mask_dp_pad_moe_topk_ids(topk_ids: torch.Tensor) -> None:
    """Set MAX_LEN pad rows' (post-translation, local) topk_ids to -1 in place.

    Under dp-attention MAX_LEN padding the gathered MoE buffer is
    [num_dp_ranks * max_len, hidden] with rank r's real rows at
    [r*max_len, r*max_len + global_num_tokens[r]); the pad rows carry stale
    hidden values, run the router, and get dispatched into experts whose
    outputs are then discarded by the post-reorder scatter — pure wasted
    compute, and a masked-grouped-GEMM workspace blow-up when they collide
    on the same top-k.  -1 is the drop sentinel both the triton fused_moe
    (filter_expert) and the DeepGEMM EP preprocess honor; it must be applied
    AFTER the local_expert_mapping gather (a pre-translation -1 aliases to
    the mapping table's last entry).  Capture-safe: per-batch state is read
    only from the replay-updated global_num_tokens_gpu tensor.
    """
    counts = _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu()
    if counts is None:
        return
    max_len = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
    rows, topk = topk_ids.shape
    if max_len <= 0 or rows != counts.shape[0] * max_len:
        # Layout mismatch (e.g. non-DP or logits-path caller): do nothing.
        return
    _mask_dp_pad_topk_ids_kernel[(rows,)](
        topk_ids,
        counts,
        max_len,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
    )


def _dp_gather_via_all_gatherv_fp8(
    global_tokens: torch.Tensor,
    local_real: torch.Tensor,
    sizes: List[int],
):
    """fp8 wire format for the variable-length DP gather: quantize the local
    rows per-token-group (the SAME group-128 quantization the MoE expert GEMMs
    apply to their input downstream), gather payload (as uint8 — NCCL has no
    fp8 dtype; the gatherv leg is broadcast-only so a byte view is safe) and
    scales in two output-buffered gatherv calls, then dequantize into the
    bf16 global buffer.  Zero pad rows quantize to (q=0, s=eps) and so
    dequantize back to exact zeros — the MoE-tail invariant is preserved.
    The combine leg (reduce_scatterv) stays bf16: NCCL SUM cannot run on fp8."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    rows = global_tokens.shape[0]
    hidden = global_tokens.shape[-1]
    q, s = sglang_per_token_group_quant_fp8(
        local_real.contiguous(), _DP_GATHER_FP8_GROUP
    )
    gq, gs = _get_dp_gather_fp8_bufs(rows, hidden, global_tokens.device)
    tp_group = get_parallel().tp_group
    tp_group.all_gatherv(q.view(torch.uint8), sizes=sizes, output=gq)
    tp_group.all_gatherv(s, sizes=sizes, output=gs)
    _dequant_per_token_group_fp8_kernel[(rows,)](
        gq.view(torch.float8_e4m3fn),
        gs,
        global_tokens,
        HIDDEN=hidden,
        NGROUPS=hidden // _DP_GATHER_FP8_GROUP,
        GROUP=_DP_GATHER_FP8_GROUP,
        BLOCK=2048,
    )


def is_dp_gatherv_active() -> bool:
    """Variable-length DP-MoE gather/scatter (all_gatherv + reduce_scatterv) is
    enabled and applicable to the CURRENT forward. Requires:
      - env SGLANG_DP_USE_GATHERV (default off),
      - supported layout (attn_tp_size==1, tp_size==attn_dp_size),
      - SUM_LEN padding mode. The gatherv pair (all_gatherv + reduce_scatterv) is
        only valid under SUM_LEN; under MAX_LEN the buffer is equal-padded and the
        gather/combine use all_gather / (aiter) reduce_scatter instead. Reading the
        per-forward padding via _DpGatheredBufferWrapper.is_dp_max_padding() (set by
        set_dp_buffer_len) keeps callers that lack a ForwardBatch (e.g.
        dp_reduce_scatter_tensor) consistent."""
    return (
        _USE_DP_GATHERV
        and not world_dp_gather_enabled()
        and get_parallel().attn_tp_size == 1
        and get_parallel().tp_size == get_parallel().attn_dp_size
        and not _DpGatheredBufferWrapper.is_dp_max_padding()
    )


def _dp_gatherv_sizes(forward_batch) -> Optional[List[int]]:
    """Per-rank CPU token counts for the buffer being gathered. The MoE gather
    passes a ForwardBatch (global_num_tokens_cpu); the logits gather passes a
    LogitsMetadata (global_num_tokens_for_logprob_cpu). Return the sizes that
    match the LOCAL tensor for this context, or None to fall back."""
    sizes = getattr(forward_batch, "global_num_tokens_for_logprob_cpu", None)
    if sizes is None:
        sizes = getattr(forward_batch, "global_num_tokens_cpu", None)
    if sizes is None:
        return None
    try:
        return [int(x) for x in sizes]
    except (TypeError, ValueError):
        return None


def _dp_gather_via_all_gatherv(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    sizes: List[int],
):
    # attn_tp_size == 1: each DP rank contributes exactly `sizes[rank]` rows.
    # CRITICAL: the MoE downstream runs on the WHOLE `global_tokens` buffer
    # (M = global_tokens.shape[0]), so the gather MUST fill every row. We pad
    # each rank's local tensor up to sizes[rank] with zeros (matching the
    # buffer's reserved per-rank slot) so sum(sizes) == buffer rows and there
    # is no uninitialized tail for the MoE to read.
    rank = dp_slot_in(sizes)
    local_rows = sizes[rank]
    if local_tokens.shape[0] == local_rows:
        local_real = local_tokens
    elif local_tokens.shape[0] > local_rows:
        local_real = local_tokens[:local_rows]
    else:
        local_real = local_tokens.new_zeros((local_rows, *local_tokens.shape[1:]))
        local_real[: local_tokens.shape[0]].copy_(local_tokens)
    # sum(sizes) == global_tokens.shape[0] is guaranteed by the caller (else it
    # falls back to all_reduce). Pass global_tokens as the NCCL output buffer so
    # the gather writes directly into it -- avoids the previous extra full-buffer
    # torch.cat + copy_ (two ~sum(sizes)*hidden DtoD copies, ~700us/layer at c512).
    # NOTE: the fp8 branch condition must be identical on EVERY DP rank (all
    # ranks must issue the same NCCL op sequence) — env/dtype/hidden are
    # rank-uniform; never gate on per-rank state like forward_mode (ranks can
    # be extend/idle-mixed within one global forward).  Prefill-only is
    # already structural: the gatherv path runs only under SUM_LEN padding,
    # which decode-only steps and CUDA-graph capture never select.
    if (
        _use_dp_gather_fp8()
        and global_tokens.dtype == torch.bfloat16
        and global_tokens.shape[-1] % _DP_GATHER_FP8_GROUP == 0
    ):
        _dp_gather_via_all_gatherv_fp8(global_tokens, local_real, sizes)
        return
    get_parallel().tp_group.all_gatherv(local_real, sizes=sizes, output=global_tokens)


def _note_dp_gather_in_prefill_graph() -> None:
    dp = get_flags().dp
    if dp.capturing_prefill_graph:
        dp.prefill_graph_has_dp_gather = True


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    """Gather each DP group's rows into its slot of the global buffer.

    Under attention CP, without ``cp_shard_counts`` the CP ranks of a DP group
    must hold the same rows, and only CP rank 0's copy is gathered. With it,
    each CP rank holds a different shard of the group's tokens and places its
    own (see dp_gather_partial). A caller whose CP ranks hold different rows
    passes the counts, or restores the full rows on every CP rank first.
    """
    _note_dp_gather_in_prefill_graph()
    if get_parallel().attn_cp_size > 1:
        # Under CP the rows are placed before a sum: an all-gather takes a block
        # from every rank of the TP group, CP ranks included.
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial, cp_shard_counts
        )
        return
    if (
        is_dp_gatherv_active()
        and forward_batch.dp_padding_mode is not None
        and not forward_batch.dp_padding_mode.is_max_len()
    ):
        # The gatherv per-rank sizes MUST sum to the pre-allocated global buffer
        # (the MoE runs on the whole buffer, so any unfilled tail = garbage).
        # The buffer was sized from the ceil_align'd global_num_tokens stored via
        # set_dp_buffer_len (forward_batch_info), so the authoritative sizes are
        # get_dp_global_num_tokens() — the SAME source the reduce_scatterv combine
        # uses (symmetric). _dp_gatherv_sizes() reads the raw (un-aligned, and for
        # the MoE-gather context the logprob-token) counts, which do NOT match the
        # buffer for prefill steps -> would force an all_reduce fallback.
        # Prefer the buffer-aligned sizes; fall back to the per-batch sizes only
        # if they happen to match (e.g. the logits gather path).
        _gatherv_sizes = get_dp_global_num_tokens()
        if _gatherv_sizes is None or sum(_gatherv_sizes) != global_tokens.shape[0]:
            _gatherv_sizes = _dp_gatherv_sizes(forward_batch)
        if _gatherv_sizes is not None and sum(_gatherv_sizes) == global_tokens.shape[0]:
            _dp_gather_via_all_gatherv(
                global_tokens, local_tokens, forward_batch, is_partial, _gatherv_sizes
            )
            return
    if (
        forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
    ):
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    """``cp_shard_counts``: when the CP ranks of a DP group hold different shards
    of its tokens, the rows of each shard that hold tokens; None when every CP
    rank holds all of them. A shard padded past its tokens has the padding
    skipped here and zeroed by ``dp_scatter``."""
    _dp_gather(
        global_tokens,
        local_tokens,
        forward_batch,
        is_partial=True,
        cp_shard_counts=cp_shard_counts,
    )


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    _dp_gather(
        global_tokens,
        local_tokens,
        forward_batch,
        is_partial=False,
        cp_shard_counts=cp_shard_counts,
    )


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    """Copy this DP group's slot of the global buffer back to the rank. With
    ``cp_shard_counts`` (as in dp_gather_partial) the rank takes back only its
    own CP shard, and the rest of ``local_tokens`` is zero."""
    _note_dp_gather_in_prefill_graph()
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert local_tokens.untyped_storage() is not global_tokens.untyped_storage(), (
            "aliasing between local_tokens and global_tokens not allowed"
        )

        if cp_shard_counts is None:
            memcpy(
                local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
            )
        else:
            start, length = _cp_shard_rows(forward_batch, cp_shard_counts)
            local_tokens[:length].copy_(global_tokens[start : start + length])


def can_use_dp_reduce_scatter() -> bool:
    """Whether the fixed TP group tiles the current attention DP x TP layout."""
    if not world_dp_gather_enabled():
        return True

    parallel = get_parallel()
    return parallel.tp_size == parallel.num_dp_ranks * parallel.attn_tp_size


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    _note_dp_gather_in_prefill_graph()
    if is_dp_gatherv_active():
        # Variable-length combine matching all_gatherv dispatch: scatter the
        # global (sum_len) tensor back to per-rank token counts. Fall through to
        # the default reduce-scatter path if per-rank sizes are unavailable.
        sizes = get_dp_global_num_tokens()
        if sizes is not None:
            get_parallel().tp_group.reduce_scatterv(input, output=output, sizes=sizes)
            return
    if get_parallel().tp_size == get_parallel().attn_dp_size:
        get_parallel().tp_group.reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(get_parallel().tp_size)[
            get_parallel().tp_rank
        ]
        get_parallel().tp_group.reduce_scatter_tensor(scattered_local_tokens, input)
        get_parallel().attn_tp_group.all_gather_into_tensor(
            output, scattered_local_tokens
        )


# ---------------------------------------------------------------------------
# Two-batch-overlap (non-EP / DP TP-MoE) async gather + combine.
#
# The DP TP-MoE path (deepseek_v4) gathers local hidden -> a global buffer
# before the experts and reduce-scatters back after. For TBO we run those two
# collectives on a single shared comm stream (mirroring the mori dispatcher's
# _comm_stream) and return a CUDA event, so the op engine can yield and let the
# OTHER ubatch's attn+MoE compute run on the compute stream while this ubatch's
# gather/combine proceeds on the comm stream. Both ubatches share ONE comm
# stream -> their collectives serialize in-order (no concurrent-collective
# deadlock on the RCCL communicator), each overlapping the other's compute.
# ---------------------------------------------------------------------------
def get_dp_tbo_comm_stream() -> torch.cuda.Stream:

    return get_stream("dp_tbo_comm")


# Persistent reusable CUDA events for non-EP DP TBO, keyed by (kind, subbatch).
# CRITICAL: do NOT create a fresh event per gather/combine -- that is ~244 new
# torch.cuda.Event per forward (61 layers x 2 ubatches x 2), and the HSA signal
# pool is exhausted after a few hundred forwards -> HSA_STATUS_ERROR_OUT_OF_RESOURCES
# ("...create internal OS-specific events"). Reuse one event per (kind, subbatch)
# and just re-record it (mirrors the mori CommStreamPool event reuse).
def _tbo_event(key) -> torch.cuda.Event:

    pool = get_resources().tbo_event_pool
    ev = pool.get(key)
    if ev is None:
        ev = torch.cuda.Event()
        pool[key] = ev
    return ev


def dp_gather_partial_async(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    event_key=("gather", 0),
) -> torch.cuda.Event:
    """Launch `dp_gather_partial` (all_gatherv) on the shared DP TBO comm stream;
    re-record + return a PERSISTENT event (keyed by `event_key`) that fires when
    the gather completes. Caller yields, then `compute_stream.wait_event(ev)`
    before reading `global_tokens`."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    # Keep buffers alive across streams (caching allocator).
    local_tokens.record_stream(comm)
    global_tokens.record_stream(comm)
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)  # inputs were produced on the compute stream
        dp_gather_partial(global_tokens, local_tokens, forward_batch)
        ev.record(comm)
    return ev


# Persistent grow-only buffers for non-EP DP TBO, keyed by (kind, tbo_subbatch).
# Reused across ALL layers (and forwards) so the caching allocator does not churn
# a fresh per-layer `torch.empty` for the 8x DP-gather / combine buffers. That
# churn (different sizes per forward x 2 ubatches x 61 layers, kept alive by the
# comm-stream record_stream) ballooned `reserved` to ~270GB and tripped
# HSA_STATUS_ERROR_OUT_OF_RESOURCES at large prefill chunks, even though the live
# (allocated) working set was only ~10GB.
_TBO_PERSIST_BUF: dict = {}


def get_tbo_persistent_buffer(
    key, rows: int, hidden: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """Return a [rows, hidden] view of a grow-only persistent buffer for `key`.
    Reallocates only when the request exceeds the cached capacity / changes
    dtype|hidden. Caller must treat the returned view as scratch (overwritten)."""
    buf = _TBO_PERSIST_BUF.get(key)
    cap = 0 if buf is None else buf.shape[0]
    if buf is None or rows > cap or buf.shape[1] != hidden or buf.dtype != dtype:
        new_rows = max(rows, cap)
        buf = torch.empty((new_rows, hidden), dtype=dtype, device=device)
        _TBO_PERSIST_BUF[key] = buf
    return buf[:rows]


def dp_reduce_scatterv_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    sizes: List[int],
    event_key=("combine", 0),
) -> torch.cuda.Event:
    """Launch the variable-length reduce_scatterv (combine) on the shared DP TBO
    comm stream; re-record + return a PERSISTENT event (keyed by `event_key`).
    Matches the gatherv (SUM_LEN) path."""
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_parallel().tp_group.reduce_scatterv(
            global_tokens, output=output_local, sizes=sizes
        )
        ev.record(comm)
    return ev


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().attn_tp_group.reduce_scatter_tensor(output, input)


def attn_cp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().attn_cp_group.reduce_scatter_tensor(output, input)


def attn_tp_all_reduce(input: torch.Tensor):
    return get_parallel().attn_tp_group.all_reduce(input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().attn_tp_group.all_gather_into_tensor(output, input)


def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().attn_cp_group.all_gather_into_tensor(output, input)


def get_moe_cp_group() -> GroupCoordinator:
    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
    return get_parallel().moe_dp_group


def get_moe_cp_rank() -> int:
    return get_parallel().moe_dp_group.rank_in_group


def get_moe_cp_size() -> int:
    return get_parallel().moe_dp_group.world_size


def is_enable_moe_cp_allgather() -> bool:
    """True when moe_dp_size < attn_cp_size, requiring allgather across CP ranks before MoE.

    In that configuration ``initialize_model_parallel`` aliases ``_MOE_DP`` to
    ``_ATTN_CP``, so the two groups report equal widths.
    """
    return get_parallel().attn_cp_size > get_parallel().moe_dp_size


def moe_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().moe_dp_group.all_gather_into_tensor(output, input)


class _StripedMoECPOutputReducer:
    """Calibrated BF16 output stripes, with native AR fallback.

    The 2 MiB stripe is a measured layout hint, not an NCCL guarantee.
    Random calibration checks each layout; strict mode separately checks actual
    expert partials. Only eager, nonoverlapped execution is supported.
    """

    def __init__(self, group):
        self.group = group
        self.rank = group.rank_in_group
        self.n = group.world_size
        self.block_bytes = 2 * 1024 * 1024
        self.layouts = {}
        self.active = {}

    def _layout(self, partial):
        block = self.block_bytes // partial.element_size()
        blocks = partial.numel() // block
        if (
            partial.dtype != torch.bfloat16
            or blocks == 0
            or partial.numel() % block
            or blocks % self.n
        ):
            return None
        per_owner = blocks // self.n
        send = [
            sum((g * self.n + self.rank) // per_owner == dest for g in range(per_owner))
            for dest in range(self.n)
        ]
        received = [
            g * self.n + source
            for source in range(self.n)
            for g in range(per_owner)
            if (g * self.n + source) // per_owner == self.rank
        ]
        recv = [
            sum(
                (g * self.n + source) // per_owner == self.rank
                for g in range(per_owner)
            )
            for source in range(self.n)
        ]
        indices = torch.tensor(
            [b % per_owner for b in received], dtype=torch.long, device=partial.device
        )
        return block, [c * block for c in send], [c * block for c in recv], indices

    def _striped(self, partial, layout):
        block, send, recv, indices = layout
        packed = (
            partial.reshape(-1, self.n, block).permute(1, 0, 2).contiguous().flatten()
        )
        reduced = partial.new_empty(partial.numel() // self.n)
        self.group.reduce_scatter_tensor(reduced, packed)
        received = torch.empty_like(reduced)
        dist.all_to_all_single(
            received,
            reduced,
            output_split_sizes=recv,
            input_split_sizes=send,
            group=self.group.device_group,
        )
        output = torch.empty_like(reduced).view(-1, block)
        output.index_copy_(0, indices, received.view(-1, block))
        return output.view(partial.shape[0] // self.n, *partial.shape[1:])

    def __call__(self, partial):
        assert partial.shape[0] % self.n == 0
        key = (tuple(partial.shape), partial.dtype, partial.device)
        if key not in self.layouts:
            layout = self._layout(partial)
            active = False
            if layout is not None:
                gen = torch.Generator(device=partial.device).manual_seed(
                    712 + self.rank
                )
                sample = torch.randn(
                    partial.shape,
                    dtype=torch.float32,
                    device=partial.device,
                    generator=gen,
                ).to(partial.dtype)
                native = self.group.all_reduce(sample.clone()).chunk(self.n)[self.rank]
                proposed = self._striped(sample, layout)
                failed = torch.tensor(
                    int(not torch.equal(native, proposed)),
                    dtype=torch.int32,
                    device=partial.device,
                )
                dist.all_reduce(
                    failed, op=dist.ReduceOp.MAX, group=self.group.device_group
                )
                active = failed.item() == 0
            self.layouts[key] = layout
            self.active[key] = active
            payload = partial.numel() * partial.element_size()
            peer_send = (
                (sum(layout[1]) - layout[1][self.rank]) * partial.element_size()
                if layout is not None
                else None
            )
            print(
                "CP_OWNER_REDUCTION "
                + json.dumps(
                    dict(
                        rank=self.rank,
                        shape=list(partial.shape),
                        striped_active=active,
                        block_bytes=self.block_bytes,
                        native_ring_send_bytes=2 * (self.n - 1) * payload // self.n,
                        rs_ring_send_bytes=(
                            (self.n - 1) * payload // self.n if active else None
                        ),
                        owner_exchange_peer_send_bytes=peer_send if active else None,
                        fallback=None if active else "native_ar_layout_not_verified",
                    )
                ),
                flush=True,
            )
        if self.active[key]:
            return self._striped(partial, self.layouts[key])
        # Preserve expert partials for the surrounding strict/numerical checks.
        return (
            self.group.all_reduce(partial.clone()).chunk(self.n)[self.rank].contiguous()
        )


@functools.lru_cache(maxsize=8)
def _striped_reducer(group):
    # Reuse only immutable shape layouts/calibration; no in-flight batch state.
    return _StripedMoECPOutputReducer(group)


def reduce_moe_cp_output(partial, rows, group, *, mode, validation="none"):
    """Complete an EP partial sum and remove this owner's padding.

    The caller selects this operation before compute skips its all-reduce.
    Routing, expert computation and the input all-gather are unchanged.
    """
    if mode not in ("reduce_scatter", "striped_reduce_scatter"):
        raise ValueError(f"Unknown CP output reduction: {mode}")
    if validation not in ("none", "strict", "numerical"):
        raise ValueError(f"Unknown CP output validation: {validation}")
    if validation == "numerical" and mode != "reduce_scatter":
        raise ValueError("Numerical validation requires reduce_scatter")
    if (
        len(rows) != group.world_size
        or any(row < 0 for row in rows)
        or partial.shape[0] != max(rows) * group.world_size
    ):
        raise ValueError("CP output must contain one equally padded chunk per owner")
    rank = group.rank_in_group
    if mode == "striped_reduce_scatter":
        local = _striped_reducer(group)(partial)
    elif partial.numel() == 0:
        local = partial[:0]
    else:
        local = partial.new_empty((max(rows), *partial.shape[1:]))
        group.reduce_scatter_tensor(local, partial)
    output = local[: rows[rank]].contiguous()
    if validation != "none":
        native = group.all_reduce(partial.clone()).chunk(group.world_size)[rank]
        native = native[: rows[rank]]
        difference = output.float() - native.float()
        relative_l2 = (
            difference.norm() / native.float().norm().clamp_min(1e-12)
        ).item()
        exact = torch.equal(output, native)
        valid = bool(torch.isfinite(output).all() and torch.isfinite(native).all())
        record = dict(
            rank=rank,
            mode=mode,
            validation=validation,
            rows=list(rows),
            exact=exact,
            relative_l2=relative_l2,
        )
        if validation == "strict":
            valid = valid and exact
        else:
            oracle = group.all_reduce(partial.float()).chunk(group.world_size)[rank]
            oracle = oracle[: rows[rank]]
            scale = oracle.norm().clamp_min(1e-12)
            baseline_error = ((native.float() - oracle).norm() / scale).item()
            candidate_error = ((output.float() - oracle).norm() / scale).item()
            record.update(
                baseline_relative_l2=baseline_error,
                candidate_relative_l2=candidate_error,
            )
            valid = (
                valid
                and bool(torch.isfinite(oracle).all())
                and relative_l2 <= 0.01
                and baseline_error <= 0.01
                and candidate_error <= 0.01
            )
        # Every rank raises together, rather than stranding peers in a collective.
        failed = torch.tensor(int(not valid), dtype=torch.int32, device=partial.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group.device_group)
        print("CP_MOE_OUTPUT_CHECK " + json.dumps(record), flush=True)
        if failed.item():
            raise RuntimeError(f"CP output {validation} validation failed: {record}")
    return output


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_parallel().attn_tp_group.all_gather(
        input, output_tensor_list=output_list
    )
