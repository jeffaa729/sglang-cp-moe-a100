"""CPU oracle/NCCL checks: direct CP2 or padded owner reduce-scatter CP2/4/8."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from dispatcher import direct_dispatch


def expert_partial(x, ids, weights, rank):
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        expert = ids[:, slot]
        belongs = (expert // 64 == rank).float()
        # Nonlinear per-expert oracle, FP32 accumulated then BF16 partials.
        value = torch.nn.functional.silu(x.float() * (1 + expert[:, None].float()/128))
        result += value * (weights[:, slot] * belongs)[:, None]
    return result.to(x.dtype)


def check_reduce_scatter(rank, world_size, device, model_shape=False):
    """Check rank-major ownership, padding, empty owners and BF16 sum error.

    Integer cases require exact ownership/results. Random BF16 cases use a
    predeclared 1% relative-L2 gate against an FP32 sum of the same partials.
    Also report differences versus native BF16 all-reduce without hiding them.
    """
    row_cases = [(9,) * world_size,
                 tuple(7 + r for r in range(world_size)),
                 (0,) + (11,) * (world_size - 1),
                 (13,) + (0,) * (world_size - 1),
                 (0,) * world_size]
    cases = [(rows, 32) for rows in row_cases]
    if model_shape:
        cases.append(((16384 // world_size,) * world_size, 2048))
    results = []
    for rows, width in cases:
        shape = (max(rows) * world_size, width)
        for integer in (True, False):
            gen = torch.Generator().manual_seed(712)
            partials = []
            for source in range(world_size):
                if integer:
                    x = (torch.arange(shape[0])[:, None] % 7
                         + torch.arange(shape[1])[None, :] % 4
                         + source).to(torch.bfloat16)
                else:
                    x = torch.randn(shape, generator=gen).to(torch.bfloat16)
                partials.append(x)
            partial = partials[rank].to(device)
            summed = partial.clone()
            dist.all_reduce(summed)
            local = partial.new_empty((max(rows), shape[1]))
            dist.reduce_scatter_tensor(local, partial)
            actual = local[:rows[rank]].float().cpu()
            offset = rank * max(rows)
            expected = torch.stack(partials).float().sum(0)[offset:offset + rows[rank]]
            baseline = summed[offset:offset + rows[rank]].float().cpu()
            difference = actual - expected
            relative_l2 = (difference.norm() / expected.norm().clamp_min(1e-12)).item()
            assert bool(torch.isfinite(actual).all()) and relative_l2 <= 0.01, (
                rows, rank, integer, relative_l2)
            if integer:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            results.append(dict(rows=rows, width=width, rank=rank, integer=integer,
                relative_l2=relative_l2, exact_allreduce=torch.equal(actual, baseline),
                max_abs_vs_fp32=difference.abs().max().item() if actual.numel() else 0,
                max_abs_vs_allreduce=(actual - baseline).abs().max().item()
                if actual.numel() else 0))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--variant', choices=['direct', 'rs'], default='direct')
    parser.add_argument('--model-shape', action='store_true',
                        help='Also test the 64 MiB BF16 16K x 2048 collective')
    args = parser.parse_args()
    if args.model_shape and args.variant != 'rs':
        parser.error('--model-shape requires --variant rs')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device(f'cuda:{rank}'))
    device = torch.device(f'cuda:{rank}')
    world_size = dist.get_world_size()
    if args.variant == 'rs':
        results = check_reduce_scatter(rank, world_size, device, args.model_shape)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir/f'rs_rank{rank}.json').write_text(json.dumps(results, indent=2))
        print(f'rank={rank}: {len(results)} owner-RS/FP32-oracle cases passed', flush=True)
        dist.destroy_process_group()
        return
    if world_size != 2:
        raise ValueError('Direct dispatcher tests still require two ranks')
    gen = torch.Generator().manual_seed(712)
    results = []
    for rows in ((9, 7), (0, 11), (13, 0), (0, 0)):
        for routing in ('mixed', 'rank0_only', 'rank1_only'):
            xs = [torch.randn(n, 32, generator=gen).to(torch.bfloat16) for n in rows]
            ids = []
            weights = []
            for n in rows:
                logits = torch.randn(n, 128, generator=gen)
                if routing != 'mixed':
                    logits[:, 64:] -= 100 if routing == 'rank0_only' else 0
                    logits[:, :64] -= 100 if routing == 'rank1_only' else 0
                chosen, selected = logits.topk(8, dim=1)
                ids.append(selected.to(torch.int32))
                weights.append(chosen.softmax(-1))
            # CPU oracle reproduces reference per-EP partial BF16 rounding.
            full_x, full_ids, full_w = torch.cat(xs), torch.cat(ids), torch.cat(weights)
            reference = (expert_partial(full_x, full_ids, full_w, 0)
                         + expert_partial(full_x, full_ids, full_w, 1))
            start = sum(rows[:rank])
            expected = reference[start:start + rows[rank]]
            for fast in (False, True):
                out, stats = direct_dispatch(xs[rank].to(device), ids[rank].to(device),
                                            weights[rank].to(device),
                                            lambda x, i, w: expert_partial(x, i, w, rank),
                                            rank, dist.group.WORLD, collect_stats=True,
                                            keep_local_rows=fast)
                actual = out.cpu()
                # CPU/GPU sigmoid implementations can differ at rounding boundaries.
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0.008, atol=0.008)
                results.append(dict(rows=rows, routing=routing, rank=rank, fast=fast,
                                    exact=torch.equal(actual, expected),
                                    max_abs=(actual.float()-expected.float()).abs().max().item() if actual.numel() else 0,
                                    **stats))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/f'unit_rank{rank}.json').write_text(json.dumps(results, indent=2))
    print(f'rank={rank}: {len(results)} CPU-oracle/NCCL cases passed', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
