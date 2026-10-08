Experimental Qwen3 CP2+EP2 MoE dispatcher; two RTX 5060 Ti GPUs.
baseline: unchanged SGLang. rs: retain input all-gather, replace output all-reduce
and slice with reduce-scatter. direct_fast: CP-local direct-to-expert dispatch.
Restricted eager-mode runtime patch; not a general production backend.

From the repository root, with dependencies installed and its Python package active:
  python experimental/cp_moe_dispatch/run.py baseline
  python experimental/cp_moe_dispatch/run.py rs
  python experimental/cp_moe_dispatch/run.py direct_fast

The thin launcher reuses sglang.test.test_utils.popen_launch_server and
python -m sglang.benchmark.serving. Fixed server settings are in run.py.
Native benchmark: deterministic random token IDs, exact 2K/8K/16K input lengths,
32 output tokens, concurrency 1, two warmups, ten measured requests per length.
Results include TTFT/E2E/TPOT/ITL, throughput, per-request outputs and cache stats.
Defaults: model /workspace/models/Qwen3-30B-A3B-NVFP4, results outside the repo
at /workspace/cp_moe_native_results. Use --label for repeats; --seed, --model,
--lengths, --repetitions and --output-dir override the corresponding defaults.

Untimed model-layer correctness checks (exclude these results from timing):
  python experimental/cp_moe_dispatch/run.py rs --validate
  python experimental/cp_moe_dispatch/run.py direct_fast --validate

Two-GPU dispatcher oracle (unequal/empty shards and destination routing):
  cd experimental/cp_moe_dispatch
  python -m torch.distributed.run --standalone --nproc-per-node=2 test_dispatcher.py

Reports and measured results are not committed. NVFP4 timings are not A100 claims.
