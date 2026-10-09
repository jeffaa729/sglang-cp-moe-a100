"""CPU oracle plus real two-GPU NCCL checks for dispatcher contracts."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device(f'cuda:{rank}'))
    device = torch.device(f'cuda:{rank}')
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
