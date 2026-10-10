"""When a MoE block all-reduces its own output, and why it does not."""

import ast
import contextlib
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import sglang
from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import (
    post_experts_output_is_complete,
    reduce_moe_output,
    should_add_replicated_moe_output,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODELS_DIR = Path(sglang.__file__).resolve().parent / "srt" / "models"


def a2a(name=None):
    names = ("flashinfer", "pplx", "flashinfer_megamoe")
    return types.SimpleNamespace(
        **{f"is_{n}": (lambda n=n: n == name) for n in names},
        is_none=lambda: name is None,
    )


@contextlib.contextmanager
def moe_config(
    *,
    tp_size=2,
    tp_rank=0,
    dwdp_size=1,
    backend=None,
    fp4_allgather=False,
    reduce_scatterv=False,
):
    with (
        patch.object(
            moe_utils,
            "get_parallel",
            return_value=types.SimpleNamespace(
                tp_size=tp_size, tp_rank=tp_rank, dwdp_size=dwdp_size
            ),
        ),
        patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a(backend)),
        patch.object(
            moe_utils,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=fp4_allgather,
        ),
        patch.object(
            moe_utils, "should_use_dp_reduce_scatterv", return_value=reduce_scatterv
        ),
    ):
        yield


class TestReduceMoeOutput(CustomTestCase):
    def all_reduces(self, **flags):
        calls = []
        with (
            patch(
                "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
                side_effect=lambda x: calls.append(x) or x * 2,
            ),
            get_forward().scoped(**flags),
        ):
            output = reduce_moe_output(torch.ones(2, 3))
        return len(calls), output

    def test_partial_output_is_reduced_once(self):
        with moe_config():
            count, output = self.all_reduces()
        self.assertEqual(count, 1)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))

    def test_single_rank_has_nothing_to_sum(self):
        with moe_config(tp_size=1):
            self.assertEqual(self.all_reduces()[0], 0)

    def test_a_later_step_owns_the_sum(self):
        for flags, config in (
            ({"fuse_mlp_allreduce": True}, {}),
            ({"mlp_reduce_scatter": True}, {}),
            ({"mlp_reduce_scatter": True}, {"reduce_scatterv": True}),
        ):
            with self.subTest(flags=flags, config=config), moe_config(**config):
                self.assertEqual(self.all_reduces(**flags)[0], 0)
                # Who runs the sum does not change whether one is owed.
                self.assertFalse(post_experts_output_is_complete(is_tp_path=True))

    def test_output_is_already_complete(self):
        for config in (
            {"backend": "flashinfer"},
            {"backend": "pplx"},
            {"backend": "flashinfer_megamoe"},
            {"dwdp_size": 2},
            {"fp4_allgather": True},
        ):
            with self.subTest(**config), moe_config(**config):
                self.assertEqual(self.all_reduces()[0], 0)
                self.assertTrue(post_experts_output_is_complete(is_tp_path=True))

    def test_fp4_allgather_only_completes_the_tp_sum(self):
        with moe_config(fp4_allgather=True):
            self.assertFalse(post_experts_output_is_complete(is_tp_path=False))


class TestReplicatedMoeOutput(CustomTestCase):
    """A shared expert replicated with tp_size=1 holds its full output on every
    rank; whatever sums the MoE output over TP must count it once."""

    def summed_over_two_ranks(self, *, flags=None, **config):
        routed = [torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)]
        shared = torch.full((2, 3), 10.0)
        outputs = []
        for rank in (0, 1):
            with (
                moe_config(tp_rank=rank, **config),
                get_forward().scoped(**(flags or {})),
            ):
                output = routed[rank]
                if should_add_replicated_moe_output():
                    output = output + shared
            outputs.append(output)
        return outputs

    def test_a_later_sum_counts_it_once(self):
        for flags, config in (
            ({"fuse_mlp_allreduce": True}, {}),
            ({"mlp_reduce_scatter": True}, {}),
            ({"mlp_reduce_scatter": True}, {"reduce_scatterv": True}),
        ):
            with self.subTest(flags=flags, config=config):
                outputs = self.summed_over_two_ranks(flags=flags, **config)
                torch.testing.assert_close(sum(outputs), torch.full((2, 3), 13.0))

    def test_every_rank_adds_it_to_a_reduced_or_complete_output(self):
        for config in (
            {},
            {"backend": "flashinfer"},
            {"fp4_allgather": True},
            {"reduce_scatterv": True},
        ):
            with self.subTest(**config):
                outputs = self.summed_over_two_ranks(**config)
                torch.testing.assert_close(outputs[1], torch.full((2, 3), 12.0))

    def test_a_single_rank_adds_it(self):
        with (
            moe_config(tp_size=1, tp_rank=0),
            get_forward().scoped(fuse_mlp_allreduce=True),
        ):
            self.assertTrue(should_add_replicated_moe_output())


class TestModelsWithExplicitDpCompletion(CustomTestCase):
    def test_direct_dp_exits_publish_their_selected_sum(self):
        # Execute the real orchestration methods without constructing weights.
        # These models own their DP exit instead of using StageBoundary.
        import __future__

        for filename, method in (
            ("deepseek_v4.py", "_run_moe_ffn_dp_sync"),
            ("qwen4_exp.py", "_run_qwen4_exp_mlp"),
        ):
            path = MODELS_DIR / filename
            node = next(
                n
                for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == method
            )
            for use_rsv in (False, True):
                with self.subTest(model=filename, reduce_scatterv=use_rsv):
                    trace = []

                    def rsv(value, *, output, sizes):
                        trace.append("RSv")
                        output.copy_(value[:2] * 2)

                    def scatter(output, value, batch):
                        trace.append("slice")
                        output.copy_(value[:2])

                    group = types.SimpleNamespace(reduce_scatterv=rsv)
                    parallel = types.SimpleNamespace(
                        attn_dp_size=2, attn_tp_size=1, tp_size=2, tp_group=group
                    )
                    namespace = dict(
                        torch=torch,
                        get_forward=get_forward,
                        get_parallel=lambda: parallel,
                        get_moe_a2a_backend=lambda: a2a(),
                        should_use_dp_reduce_scatterv=lambda: use_rsv,
                        is_dp_gatherv_active=lambda: False,
                        envs=types.SimpleNamespace(
                            SGLANG_DP_USE_REDUCE_SCATTER=types.SimpleNamespace(
                                get=lambda: False
                            )
                        ),
                        _SHARED_EXPERT_LOCAL=False,
                        nullcontext=contextlib.nullcontext,
                        get_global_dp_buffer=lambda g: torch.empty(4, 3),
                        get_local_dp_buffer=lambda g: torch.empty(2, 3),
                        get_dp_global_num_tokens=lambda: [2, 2],
                        dp_gather_replicate=lambda output, value, batch: output.fill_(
                            1
                        ),
                        dp_scatter=scatter,
                    )
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]),
                            str(path),
                            "exec",
                            flags=__future__.annotations.compiler_flag,
                        ),
                        namespace,
                    )
                    model = types.SimpleNamespace(
                        dsa_enable_prefill_cp=False,
                        config=types.SimpleNamespace(num_experts=4),
                        _qwen4_exp_use_dp_moe_gather=lambda: True,
                        _qwen4_exp_use_attn_tp_a2a_scatter=lambda: False,
                        mlp=lambda value, batch, **kwargs: reduce_moe_output(value),
                    )
                    batch = types.SimpleNamespace(
                        dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: True)
                    )
                    kwargs = (
                        dict(input_ids=None, input_ids_global=None)
                        if filename == "deepseek_v4.py"
                        else {}
                    )
                    with (
                        moe_config(),
                        get_forward().scoped(
                            fuse_mlp_allreduce=False, mlp_reduce_scatter=False
                        ),
                        patch(
                            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
                            side_effect=lambda value: trace.append("AR") or value * 2,
                        ),
                    ):
                        output = namespace[method](
                            model, torch.ones(2, 3), batch, **kwargs
                        )
                        self.assertFalse(get_forward().mlp_reduce_scatter)
                    self.assertEqual(trace, ["RSv"] if use_rsv else ["AR", "slice"])
                    torch.testing.assert_close(output, torch.full((2, 3), 2.0))


class TestMoeCPOutput(unittest.TestCase):
    def group(self, size, rank):
        return Mock(world_size=size, rank_in_group=rank, device_group=object())

    def test_stripe_exchange_restores_contiguous_owner_rows(self):
        for size in (2, 4, 8):
            for rank in range(size):
                with self.subTest(size=size, rank=rank):
                    group = self.group(size, rank)
                    reducer = dp_attention._StripedMoECPOutputReducer(group)
                    reducer.block_bytes = 8  # Tiny layout-only CPU fixture.
                    value = torch.arange(size * 8).to(torch.bfloat16).view(size * 2, 4)
                    layout = reducer._layout(value)
                    block, send, recv, _ = layout
                    reduced = value.reshape(-1, size, block)[:, rank].flatten()
                    group.reduce_scatter_tensor.side_effect = (
                        lambda output, packed: output.copy_(reduced)
                    )

                    def exchange(
                        output, input, *, output_split_sizes, input_split_sizes, group
                    ):
                        self.assertTrue(torch.equal(input, reduced))
                        self.assertEqual(input_split_sizes, send)
                        self.assertEqual(output_split_sizes, recv)
                        per_owner = value.numel() // block // size
                        received_blocks = [
                            g * size + source
                            for source in range(size)
                            for g in range(per_owner)
                            if (g * size + source) // per_owner == rank
                        ]
                        output.copy_(
                            value.flatten().view(-1, block)[received_blocks].flatten()
                        )

                    with patch.object(
                        dp_attention.dist, "all_to_all_single", side_effect=exchange
                    ):
                        output = reducer._striped(value, layout)
                    torch.testing.assert_close(
                        output, value.chunk(size)[rank], rtol=0, atol=0
                    )
                    packed = group.reduce_scatter_tensor.call_args.args[1]
                    torch.testing.assert_close(
                        packed,
                        value.reshape(-1, size, block)
                        .permute(1, 0, 2)
                        .contiguous()
                        .flatten(),
                        rtol=0,
                        atol=0,
                    )

    def test_unsupported_layout_falls_back_without_modifying_partials(self):
        for size in (2, 4, 8):
            group = self.group(size, 0)
            group.all_reduce.side_effect = lambda x: x * size
            reducer = dp_attention._StripedMoECPOutputReducer(group)
            for dtype in (torch.float32, torch.bfloat16):
                value = torch.ones(size * 3, 4, dtype=dtype)
                before = value.clone()
                output = reducer(value)
                torch.testing.assert_close(output, before.chunk(size)[0] * size)
                torch.testing.assert_close(value, before)
            group.reduce_scatter_tensor.assert_not_called()

    def test_peer_calibration_failure_rejects_local_matching_layout(self):
        group = self.group(4, 1)
        group.all_reduce.side_effect = lambda x: x * 4
        reducer = dp_attention._StripedMoECPOutputReducer(group)
        reducer.block_bytes = 8
        value = torch.ones(16, 4, dtype=torch.bfloat16)
        with (
            patch.object(
                reducer, "_striped", side_effect=lambda x, layout: x.chunk(4)[1] * 4
            ),
            patch.object(
                dp_attention.dist,
                "all_reduce",
                side_effect=lambda failed, **kw: failed.fill_(1),
            ),
        ):
            output = reducer(value)
        self.assertFalse(next(iter(reducer.active.values())))
        torch.testing.assert_close(output, value.chunk(4)[1] * 4)

    def test_reduce_scatter_trims_uneven_and_empty_owner_rows(self):
        for size in (2, 4, 8):
            for rows in ([0] * size, list(range(size)), [3] * size):
                for rank in range(size):
                    group = self.group(size, rank)
                    value = torch.ones(max(rows) * size, 4, dtype=torch.bfloat16)
                    group.reduce_scatter_tensor.side_effect = (
                        lambda output, partial: output.fill_(size)
                    )
                    output = dp_attention.reduce_moe_cp_output(
                        value, rows, group, mode="reduce_scatter"
                    )
                    self.assertEqual(output.shape, (rows[rank], 4))
                    self.assertTrue(output.is_contiguous())
                    torch.testing.assert_close(output, torch.full_like(output, size))

    def test_numerical_track_checks_actual_partials_against_fp32_oracle(self):
        group = self.group(2, 0)
        group.all_reduce.side_effect = lambda value: value * 2
        value = torch.ones(6, 4, dtype=torch.bfloat16)
        with patch.object(dp_attention.dist, "all_reduce"):
            group.reduce_scatter_tensor.side_effect = (
                lambda output, partial: output.fill_(2.015625)
            )
            output = dp_attention.reduce_moe_cp_output(
                value, [2, 3], group, mode="reduce_scatter", validation="numerical"
            )
            self.assertTrue(torch.all(output == 2.015625))
            group.reduce_scatter_tensor.side_effect = (
                lambda output, partial: output.fill_(2.0625)
            )
            with self.assertRaisesRegex(RuntimeError, "numerical validation failed"):
                dp_attention.reduce_moe_cp_output(
                    value, [2, 3], group, mode="reduce_scatter", validation="numerical"
                )

    def test_strict_checks_actual_partials_and_propagates_peer_failure(self):
        group = self.group(2, 0)
        group.all_reduce.side_effect = lambda x: x * 2
        group.reduce_scatter_tensor.side_effect = lambda output, partial: output.fill_(
            2
        )
        value = torch.ones(6, 4, dtype=torch.bfloat16)
        with patch.object(dp_attention.dist, "all_reduce"):
            output = dp_attention.reduce_moe_cp_output(
                value, [2, 3], group, mode="reduce_scatter", validation="strict"
            )
        torch.testing.assert_close(output, torch.full((2, 4), 2, dtype=value.dtype))
        for corrupt_local in (False, True):
            if corrupt_local:
                group.reduce_scatter_tensor.side_effect = (
                    lambda output, partial: output.fill_(3)
                )
            with (
                patch.object(
                    dp_attention.dist,
                    "all_reduce",
                    side_effect=lambda failed, **kw: failed.fill_(1),
                ),
                self.assertRaisesRegex(RuntimeError, "strict validation failed"),
            ):
                dp_attention.reduce_moe_cp_output(
                    value, [2, 3], group, mode="reduce_scatter", validation="strict"
                )


if __name__ == "__main__":
    unittest.main()
