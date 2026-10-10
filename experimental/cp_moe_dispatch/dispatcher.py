"""Two-rank CP-owned-token dispatcher. No changes to expert arithmetic.

One payload per remote destination, not per selected expert. IDs and FP32
weights are transported bitwise alongside BF16 activations in one NCCL packet.
This correctness-first prototype deliberately uses dynamic sizes/allocations.
"""
import torch
import torch.distributed as dist


def exchange(tensor, send_rows, recv_rows, rank, group):
    sends = [0, 0]
    receives = [0, 0]
    sends[1 - rank] = send_rows
    receives[1 - rank] = recv_rows
    output = tensor.new_empty((recv_rows, *tensor.shape[1:]))
    dist.all_to_all_single(output, tensor.contiguous(), receives, sends, group=group)
    return output


def direct_dispatch(x, ids, weights, expert_fn, rank, group, num_experts=128,
                    collect_stats=False, keep_local_rows=False):
    """expert_fn consumes GLOBAL IDs and returns this rank's weighted partial.

    Returned rows are in exactly the input CP owner's order. Supports empty
    destinations and uneven CP counts. Expert ownership must be contiguous.
    """
    if dist.get_world_size(group) != 2 or rank not in (0, 1):
        raise ValueError("This prototype requires aligned CP=EP=2")
    if x.dtype != torch.bfloat16 or num_experts % 2:
        raise ValueError("BF16 communication and even expert partition required")
    k, h = ids.shape[1], x.shape[1]
    owner = ids // (num_experts // 2)
    # Avoid a second dynamic nonzero/CPU size synchronization in the fast path.
    # The existing EP kernel already emits zero for rows with no local experts.
    local_idx = None if keep_local_rows else torch.where((owner == rank).any(dim=1))[0]
    local_rows = x.shape[0] if keep_local_rows else local_idx.numel()
    remote_idx = torch.where((owner == 1 - rank).any(dim=1))[0]
    send_rows = remote_idx.numel()
    counts = torch.zeros(2, dtype=torch.int64, device=x.device)
    counts[1 - rank] = send_rows
    incoming_counts = torch.empty_like(counts)
    dist.all_to_all_single(incoming_counts, counts, group=group)
    recv_rows = int(incoming_counts[1 - rank].item())
    packet = torch.cat((x[remote_idx],
                        ids[remote_idx].to(torch.int32).contiguous().view(torch.bfloat16),
                        weights[remote_idx].float().contiguous().view(torch.bfloat16)), dim=1)
    incoming = exchange(packet, send_rows, recv_rows, rank, group)
    remote_x = incoming[:, :h].contiguous()
    remote_ids = incoming[:, h:h + 2*k].contiguous().view(torch.int32)
    remote_weights = incoming[:, h + 2*k:].contiguous().view(torch.float32)
    # Keep origin rank order, matching the reference's rank-major gather.
    local_x = x if keep_local_rows else x[local_idx]
    local_ids = ids if keep_local_rows else ids[local_idx]
    local_weights = weights if keep_local_rows else weights[local_idx]
    pieces_x = [local_x, remote_x] if rank == 0 else [remote_x, local_x]
    pieces_ids = [local_ids, remote_ids] if rank == 0 else [remote_ids, local_ids]
    pieces_weights = ([local_weights, remote_weights] if rank == 0
                     else [remote_weights, local_weights])
    expert_x = torch.cat(pieces_x)
    if expert_x.shape[0]:
        partial = expert_fn(expert_x, torch.cat(pieces_ids).to(torch.int32),
                            torch.cat(pieces_weights).float())
    else:
        partial = x.new_empty((0, h))
    if rank == 0:
        local_partial, remote_partial = partial[:local_rows], partial[local_rows:]
    else:
        remote_partial, local_partial = partial[:recv_rows], partial[recv_rows:]
    returned = exchange(remote_partial, recv_rows, send_rows, rank, group)
    if keep_local_rows:
        # expert_fn returns a fresh partial; reusing its local view is safe.
        result = local_partial
    else:
        result = torch.zeros_like(x)
        result.index_copy_(0, local_idx, local_partial)
    result.index_add_(0, remote_idx, returned)
    stats = None
    if collect_stats:
        local_expert_tokens = (int((owner == rank).any(dim=1).sum().item())
                               if keep_local_rows else local_rows)
        stats = dict(local_tokens=x.shape[0], local_expert_tokens=local_expert_tokens,
                     remote_tokens=send_rows, received_tokens=recv_rows,
                     send_payload_bytes=send_rows * (h + 4*k) * 2 + recv_rows * h * 2 + 8,
                     send_activation_bytes=(send_rows + recv_rows) * h * 2)
    return result, stats
