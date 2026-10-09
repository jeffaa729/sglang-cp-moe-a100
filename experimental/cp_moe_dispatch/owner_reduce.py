"""Experimental native-order striped RS, followed by a CP-owner exchange.

The 2 MiB stripe is a measured layout hint, NOT a portable NCCL guarantee.
Each new shape is calibrated bitwise against native AR on random BF16 data.
Unsupported or nonmatching layouts fall back to native AR plus owner slicing.
Only eager, nonoverlapped use is supported, as in the surrounding extension.
"""
import json
import torch
import torch.distributed as dist


class StripedOwnerReduce:
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
        if (partial.dtype != torch.bfloat16 or blocks == 0
                or partial.numel() % block or blocks % self.n):
            return None
        per_owner = blocks // self.n
        send = [sum((g*self.n+self.rank)//per_owner == dest for g in range(per_owner))
                for dest in range(self.n)]
        received = [g*self.n+source for source in range(self.n) for g in range(per_owner)
                    if (g*self.n+source)//per_owner == self.rank]
        recv = [sum((g*self.n+source)//per_owner == self.rank for g in range(per_owner))
                for source in range(self.n)]
        indices = torch.tensor([b % per_owner for b in received],
                               dtype=torch.long, device=partial.device)
        return block, [c*block for c in send], [c*block for c in recv], indices

    def _striped(self, partial, layout):
        block, send, recv, indices = layout
        packed = partial.reshape(-1, self.n, block).permute(1, 0, 2).contiguous().flatten()
        reduced = partial.new_empty(partial.numel() // self.n)
        self.group.reduce_scatter_tensor(reduced, packed)
        received = torch.empty_like(reduced)
        dist.all_to_all_single(received, reduced, output_split_sizes=recv,
            input_split_sizes=send, group=self.group.device_group)
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
                gen = torch.Generator(device=partial.device).manual_seed(712 + self.rank)
                sample = torch.randn(partial.shape, dtype=torch.float32,
                    device=partial.device, generator=gen).to(partial.dtype)
                native = self.group.all_reduce(sample.clone()).chunk(self.n)[self.rank]
                proposed = self._striped(sample, layout)
                failed = torch.tensor(int(not torch.equal(native, proposed)),
                                      dtype=torch.int32, device=partial.device)
                dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.group.device_group)
                active = failed.item() == 0
            self.layouts[key] = layout
            self.active[key] = active
            payload = partial.numel() * partial.element_size()
            peer_send = ((sum(layout[1])-layout[1][self.rank]) * partial.element_size()
                         if layout is not None else None)
            print("CP_OWNER_REDUCTION " + json.dumps(dict(rank=self.rank,
                shape=list(partial.shape), striped_active=active,
                block_bytes=self.block_bytes,
                native_ring_send_bytes=2*(self.n-1)*payload//self.n,
                rs_ring_send_bytes=(self.n-1)*payload//self.n if active else None,
                owner_exchange_peer_send_bytes=peer_send if active else None,
                fallback=None if active else "native_ar_layout_not_verified")), flush=True)
        if self.active[key]:
            return self._striped(partial, self.layouts[key])
        # Preserve expert partials for the surrounding strict/numerical checks.
        return self.group.all_reduce(partial.clone()).chunk(self.n)[self.rank].contiguous()
