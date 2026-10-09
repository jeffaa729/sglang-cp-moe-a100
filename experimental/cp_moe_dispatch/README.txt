Experimental Qwen3 CP-aware MoE dispatcher on RTX 5060 Ti GPUs.
baseline: unchanged SGLang. rs: retain input all-gather, replace output all-reduce
and slice with reduce-scatter. direct_fast: CP-local direct-to-expert dispatch.
RS accepts aligned TP=CP=EP=2/4/8. Direct dispatch is still CP2+EP2 only.
Restricted eager-mode runtime patch; not a general production backend.

Validation status (2026-10-09): owner-collective tests pass at 2/4/8 ranks;
the generalized two-GPU model passes strict checks. Four-GPU 16K validation
fails bitwise equality at layer 0 despite exact routing (relative L2 0.233-0.264%).
Eight-GPU full-model validation and 4/8-GPU performance are not established.
The strict gate remains unchanged; do not treat configuration acceptance as
full-model support or run performance acceptance ahead of correctness.

From the repository root, with dependencies installed and its Python package active:
  python experimental/cp_moe_dispatch/run.py baseline
  python experimental/cp_moe_dispatch/run.py rs
  python experimental/cp_moe_dispatch/run.py direct_fast

The thin launcher reuses sglang.test.test_utils.popen_launch_server and
python -m sglang.benchmark.serving. Conservative server defaults are in run.py.
Native benchmark: deterministic random token IDs, exact 2K/8K/16K input lengths,
32 output tokens, concurrency 1, two warmups, ten measured requests per length.
Results include TTFT/E2E/TPOT/ITL, throughput, per-request outputs and cache stats.
Defaults: model /workspace/models/Qwen3-30B-A3B-NVFP4, results outside the repo
at /workspace/cp_moe_native_results. Use --label for repeats; --seed, --model,
--lengths, --repetitions and --output-dir override the corresponding defaults.

16K real-document comparison template (only after that GPU count passes validation):
  python experimental/cp_moe_dispatch/run.py baseline --parallel-size 8 --max-running-requests 2 --max-total-tokens 36864 --dataset-path /workspace/cp_moe_8gpu/longbench_16k.jsonl --concurrencies 1 2 8 --repetitions 16 --label n8_baseline_r1 --output-dir /workspace/cp_moe_8gpu
  python experimental/cp_moe_dispatch/run.py rs --parallel-size 8 --max-running-requests 2 --max-total-tokens 36864 --dataset-path /workspace/cp_moe_8gpu/longbench_16k.jsonl --concurrencies 8 2 1 --repetitions 16 --label n8_rs_r1 --output-dir /workspace/cp_moe_8gpu
Use --parallel-size 2 or 4 for smaller aligned groups. Select physical GPUs with
CUDA_VISIBLE_DEVICES. Client concurrency is not the GPU resident batch size;
the launcher's explicit prefill request cap remains one. Each label produces
native JSONL records plus a server log; new labels are required for repeats.
Add --profile for a separate native Torch trace (use two requests and a unique
profile label); never include profiled timings in performance aggregates.

Untimed model-layer correctness checks (exclude these results from timing):
  python experimental/cp_moe_dispatch/run.py rs --validate
  python experimental/cp_moe_dispatch/run.py direct_fast --validate

Two-GPU dispatcher oracle (unequal/empty shards and destination routing):
  cd experimental/cp_moe_dispatch
  python -m torch.distributed.run --standalone --nproc-per-node=2 test_dispatcher.py

Owner reduce-scatter padding/FP32 oracle (set nproc to 2, 4, or 8):
  python -m torch.distributed.run --standalone --nproc-per-node=8 test_dispatcher.py --variant rs --output-dir /workspace/cp_moe_8gpu/unit_rs_n8
Integer cases require exact results; random BF16 cases predeclare relative L2
<= 1% versus an FP32 sum. Native BF16 all-reduce differences are also recorded.
Real model --validate still requires exact output/routing parity; collective
tests alone do not establish full-model or serving correctness.

Reports and measured results are not committed. NVFP4 timings are not A100 claims.
