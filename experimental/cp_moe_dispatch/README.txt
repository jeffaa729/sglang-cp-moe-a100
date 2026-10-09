Experimental Qwen3 CP-aware MoE dispatcher on RTX 5060 Ti GPUs.
baseline: unchanged SGLang. rs: retain input all-gather, replace output all-reduce
and slice with reduce-scatter. direct_fast: CP-local direct-to-expert dispatch.
RS accepts aligned TP=CP=EP=2/4/8. Direct dispatch is still CP2+EP2 only.
Restricted eager-mode runtime patch; not a general production backend.

rs_striped is a separate candidate: pack native reduction stripes, reduce-scatter,
then exchange reduced pieces to their CP owners. A 2 MiB stripe matches the
measured host; each new shape is bitwise-calibrated on random BF16 values.
Unsupported/nonmatching layouts fall back to native AR and log that decision.
Four-GPU 16K strict model checks pass (192/192), with 8/8 fixed-admission C1/C2
responses and all 256 generated-token log probabilities exactly matching native.
Eight-GPU strict checks also pass (384/384), with 8/8 fixed-admission C1/C2
responses and all 256 generated-token log probabilities exactly matching native.
Matched native serving results are retained outside the repository. Logged ring
send bytes are analytical payload estimates, not measured PCIe traffic or elapsed time.

Ordinary rs status (2026-10-09): owner-collective tests pass at 2/4/8 ranks;
the generalized two-GPU model passes strict checks. Four-GPU 16K validation
fails bitwise equality at layer 0 despite exact routing (relative L2 0.233-0.264%).
Eight-GPU full-model validation and 4/8-GPU performance are not established.
The separate four-GPU numerical layer check passes, but the initial real-document
output check matches only 1/4 C1 and 0/4 C2 responses. Native C1 repeats 4/4
exactly; native C2 repeats 3/4, so scheduling also needs control at concurrency.
Ring-only strict validation still fails. No ordinary-rs 4/8-GPU performance is accepted.
Strict remains the default. A separately authorized numerical-validation track
is available; its results do not establish end-to-end correctness by themselves.
Do not treat configuration acceptance as full-model support or run performance
acceptance ahead of correctness.

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
  python experimental/cp_moe_dispatch/run.py rs_striped --parallel-size 8 --max-running-requests 2 --max-total-tokens 36864 --dataset-path /workspace/cp_moe_8gpu/longbench_16k.jsonl --concurrencies 8 2 1 --repetitions 16 --label n8_rs_striped_r1 --output-dir /workspace/cp_moe_8gpu
Use --parallel-size 2 or 4 for smaller aligned groups. Select physical GPUs with
CUDA_VISIBLE_DEVICES. Client concurrency is not the GPU resident batch size;
the launcher's explicit prefill request cap remains one. Each label produces
native JSONL records plus a server log; new labels are required for repeats.
Add --profile for a separate native Torch trace (use two requests and a unique
profile label); never include profiled timings in performance aggregates.

Untimed model-layer correctness checks (exclude these results from timing):
  python experimental/cp_moe_dispatch/run.py rs --validate
  python experimental/cp_moe_dispatch/run.py direct_fast --validate
  python experimental/cp_moe_dispatch/run.py rs_striped --parallel-size 4 --validate --lengths 16384
  python experimental/cp_moe_dispatch/run.py rs --parallel-size 4 --validate --validation-mode numerical --lengths 16384
Numerical mode preserves the predeclared 1% relative-L2 bound, requires exact
routing slots and same-partial native all-reduce/reference agreement, and checks
both native and candidate output against an FP32 sum. Bitwise differences remain
reported. It is validation-only; output/log-probability comparisons are a separate
end-to-end gate before performance acceptance.

Two-GPU dispatcher oracle (unequal/empty shards and destination routing):
  cd experimental/cp_moe_dispatch
  python -m torch.distributed.run --standalone --nproc-per-node=2 test_dispatcher.py

Owner reduce-scatter padding/FP32 oracle (set nproc to 2, 4, or 8):
  python -m torch.distributed.run --standalone --nproc-per-node=8 test_dispatcher.py --variant rs --output-dir /workspace/cp_moe_8gpu/unit_rs_n8
Add --model-shape to include the actual 16K x 2048 message size: small collectives
alone do not expose the model-sized BF16 all-reduce/reduce-scatter discrepancy.
Use --variant rs_striped for the compatible candidate: it additionally requires
bitwise native-AR equality, active optimization at the model shape, and correct
all-rank fallback when a candidate parity fault is deliberately injected on rank 0.
Integer cases require exact results; random BF16 cases predeclare relative L2
<= 1% versus an FP32 sum. Native BF16 all-reduce differences are also recorded.
Default model --validate requires exact output/routing parity; collective
tests alone do not establish full-model or serving correctness.

Reports and measured results are not committed. NVFP4 timings are not A100 claims.
