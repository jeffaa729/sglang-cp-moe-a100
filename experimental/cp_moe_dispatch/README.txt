Experimental CP2+EP2 MoE dispatcher for Qwen3-30B-A3B-NVFP4.

GPU implementation is the version tested on two RTX 5060 Ti cards.
Modes: baseline, direct, direct_fast, rs (all-gather + reduce-scatter).
No production SGLang modules are edited; the extension is opt-in per process.

Run on a Linux GPU host with the existing dependencies installed:
  python experimental/cp_moe_dispatch/compare.py

Run only the long-context before/after:
  python experimental/cp_moe_dispatch/run.py baseline
  python experimental/cp_moe_dispatch/run.py rs

Results/logs default to /workspace/cp_moe_dispatch_results, outside this repo.
Both runners accept --model and --output-dir. The default model path is
/workspace/models/Qwen3-30B-A3B-NVFP4. Runtime settings are contained in run.py.
The launcher accepts extension commits but verifies the baseline Python tree.

CPU-only checks:
  cd experimental/cp_moe_dispatch
  python -m unittest test_benchmark_client test_runner

Two-GPU oracle checks (from the same directory):
  python -m torch.distributed.run --standalone --nproc-per-node=2 test_dispatcher.py

Reports and measured results are intentionally not committed here.
This is a restricted eager-mode prototype, not a general production backend.
