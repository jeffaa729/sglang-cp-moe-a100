"""Pinned Qwen3 eager-prefill extension: direct dispatch or AG + RS.

RS experiments accept aligned TP=CP=EP in (2, 4, 8); direct remains two-rank.
Accepting a configuration does not establish model correctness on that size.
Restricted to attention DP=1, MoE TP=1, zigzag, CUTLASS,
standard top-k, no shared experts, no overlap or CUDA graphs. Decode falls back.
All configuration is contained in run.py rather than persistent environment.
"""
import json
import os
import torch
from dispatcher import direct_dispatch


def install():
    from sglang.srt.layers.layer_boundary import ops, prepare
    from sglang.srt.models import qwen3_moe as model
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.runtime_context import get_parallel
    mode = os.environ["CP_MOE_VARIANT"]
    is_direct = mode in ("direct", "direct_fast")
    validation_mode = os.environ.get("CP_MOE_VALIDATE", "0")
    if validation_mode not in ("0", "1", "numerical"):
        raise ValueError(f"Unknown CP validation mode: {validation_mode}")
    validate = validation_mode != "0"
    numerical = validation_mode == "numerical"
    if numerical and mode != "rs":
        raise ValueError("Numerical validation is only implemented for reduce-scatter")
    original_gather = ops.moe_cp_gather
    original_take = ops.moe_cp_take_back
    original_forward = model.Qwen3MoeSparseMoeBlock.forward_normal
    state = {}
    seen = set()

    def gather(x, rows, size):
        p = get_parallel()
        assert size in (2, 4, 8), size
        assert (p.tp_size, p.attn_cp_size, p.moe_ep_size, p.moe_tp_size,
                p.attn_dp_size) == (size, size, size, 1, 1)
        assert p.attn_cp_rank == p.moe_ep_rank
        assert p.attn_cp_group.ranks == p.moe_ep_group.ranks
        assert len(rows) == size and x.shape[0] == rows[p.attn_cp_rank]
        assert not state, "Overlapping CP batches are not supported"
        if is_direct and size != 2:
            raise ValueError("Direct dispatch is not generalized beyond CP=EP=2")
        # Dynamic packing overhead is not worthwhile for tiny prefill batches.
        # Preserve the unmodified path for warmup, health checks and short chats.
        if is_direct and max(rows) < 128:
            return original_gather(x, rows, size)
        state.update(rows=list(rows), local=x, complete=False)
        return x if is_direct else original_gather(x, rows, size)

    def take(x, rows):
        if state.get("complete"):
            expected = state["rows"][get_parallel().attn_cp_rank]
            assert x.shape[0] == expected
            state.clear()
            return x
        return original_take(x, rows)

    def forward(block, x):
        if not state:
            return original_forward(block, x)
        p = get_parallel()
        rank = p.attn_cp_rank
        rows = state["rows"]
        key = (block.layer_id, tuple(rows))
        do_check = validate and key not in seen
        ref = None
        reference_ids = None
        reference_weights = None
        if do_check:
            full = original_gather(state["local"], rows, len(rows))
            reference_logits, _ = block.gate(full)
            reference_topk = block.topk(full, reference_logits)
            reference_ids = original_take(reference_topk.topk_ids, rows).clone()
            reference_weights = original_take(reference_topk.topk_weights, rows).clone()
            ref = original_take(original_forward(block, full), rows).clone()
        if is_direct:
            # Preserve baseline GEMM/top-k geometry, including owner row offset.
            # Only local REAL activations are present; other rows are zeros.
            # This avoids BF16 router tactic changes caused by a smaller M.
            gate_input = x.new_zeros((max(rows)*len(rows), x.shape[1]))
            gate_input[rank*max(rows):rank*max(rows) + x.shape[0]].copy_(x)
            logits, _ = block.gate(gate_input)
            padded_topk = block.topk(gate_input, logits)
            topk = StandardTopKOutput(original_take(padded_topk.topk_weights, rows),
                                      original_take(padded_topk.topk_ids, rows), None)
        else:
            logits, _ = block.gate(x)
            topk = block.topk(x, logits)
        assert isinstance(topk, StandardTopKOutput), type(topk)
        stats = None
        oracle_checks = None
        if is_direct:
            def experts(hidden, ids, weights):
                return block.experts(hidden, StandardTopKOutput(weights, ids, None))
            out, stats = direct_dispatch(x, topk.topk_ids, topk.topk_weights,
                                         experts, rank, p.moe_ep_group.device_group,
                                         collect_stats=do_check,
                                         keep_local_rows=mode == "direct_fast")
        elif mode == "rs":
            partial = block.experts(x, topk)
            # Same reduction as all-reduce + slice, but never replicates the sum.
            padded_rows = max(rows)
            local = partial.new_empty((padded_rows, partial.shape[1]))
            p.moe_ep_group.reduce_scatter_tensor(local, partial)
            out = local[:rows[rank]].contiguous()
            if do_check and numerical:
                # Untimed: isolate the collective using the SAME expert partials.
                native = original_take(p.moe_ep_group.all_reduce(partial.clone()), rows)
                oracle = original_take(p.moe_ep_group.all_reduce(partial.float()), rows)
                scale = oracle.norm().clamp_min(1e-12)
                oracle_checks = dict(
                    same_partial_ar_exact=torch.equal(native, ref),
                    finite=bool(torch.isfinite(oracle).all() and torch.isfinite(ref).all()),
                    baseline_relative_l2=((ref.float()-oracle).norm()/scale).item(),
                    candidate_relative_l2=((out.float()-oracle).norm()/scale).item())
        else:
            raise ValueError(mode)
        state["complete"] = True
        if do_check:
            seen.add(key)
            difference = (out.float() - ref.float()).abs()
            scale = ref.float().norm().clamp_min(1e-12)
            record = dict(variant=mode, layer=block.layer_id, rank=rank, rows=rows,
                          validation_mode="numerical" if numerical else "strict",
                          exact=torch.equal(out, ref), max_abs=difference.max().item(),
                          relative_l2=(difference.norm()/scale).item(),
                          finite=bool(torch.isfinite(out).all()), traffic=stats)
            if oracle_checks is not None:
                record["fp32_oracle"] = oracle_checks
            checked_ids = topk.topk_ids if is_direct else original_take(topk.topk_ids, rows)
            checked_weights = topk.topk_weights if is_direct else original_take(topk.topk_weights, rows)
            record["routing_weights_exact"] = torch.equal(checked_weights, reference_weights)
            record["routing_slot_mismatches"] = int((checked_ids != reference_ids).sum().item())
            record["routing_expert_set_mismatches"] = int((checked_ids.sort(-1).values
                                                          != reference_ids.sort(-1).values).sum().item())
            print("CP_DISPATCH_CHECK " + json.dumps(record), flush=True)
            # Strict model gate: numerical bound plus exact native output/routing.
            assert record["finite"] and record["relative_l2"] <= 0.01, record
            assert record["routing_expert_set_mismatches"] == 0, record
            if numerical:
                assert record["routing_slot_mismatches"] == 0, record
                assert record["routing_weights_exact"], record
                assert oracle_checks["same_partial_ar_exact"] and oracle_checks["finite"], record
                assert oracle_checks["baseline_relative_l2"] <= 0.01, record
                assert oracle_checks["candidate_relative_l2"] <= 0.01, record
            else:
                assert record["exact"], record
        return out

    ops.moe_cp_gather = gather
    prepare.moe_cp_gather = gather
    ops.moe_cp_take_back = take
    model.Qwen3MoeSparseMoeBlock.forward_normal = forward
