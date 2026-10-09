"""Aligned CP/EP experiment; reuse SGLang's launcher and serving benchmark."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import is_port_available
from sglang.test.test_utils import popen_launch_server

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODEL = "/workspace/models/Qwen3-30B-A3B-NVFP4"
SERVER_ARGS = [
    "--tp-size", "2", "--ep-size", "2", "--moe-dp-size", "1",
    "--attn-cp-size", "2", "--enable-prefill-cp", "--cp-strategy", "zigzag",
    "--dtype", "bfloat16", "--quantization", "modelopt_fp4",
    "--moe-a2a-backend", "none", "--moe-runner-backend", "flashinfer_cutlass",
    "--disable-flashinfer-cutlass-moe-fp4-allgather", "--attention-backend", "fa4",
    "--boundary-reduction", "ar", "--kv-cache-dtype", "bfloat16",
    "--context-length", "18432", "--max-total-tokens", "18432",
    "--max-running-requests", "1", "--chunked-prefill-size", "-1",
    "--max-prefill-tokens", "18432", "--mem-fraction-static", "0.85",
    "--disable-radix-cache", "--disable-overlap-schedule", "--disable-custom-all-reduce",
    "--cuda-graph-backend-prefill", "disabled", "--cuda-graph-backend-decode", "disabled",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["baseline", "rs", "direct", "direct_fast"])
    parser.add_argument("--validate", action="store_true", help="Untimed layerwise reference checks")
    parser.add_argument("--profile", action="store_true", help="Separate native Torch profile; not a timed result")
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 16384])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-size", type=int, choices=[2, 4, 8], default=2,
                        help="Aligned TP=CP=EP size; not independent replicas")
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=18432)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1],
                        help="Client limits; actual GPU residency must be measured")
    parser.add_argument("--dataset-path", type=Path,
                        help="Optional real-text dataset for the native custom loader")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--label", help="Use a unique label for each run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/cp_moe_native_results"))
    args = parser.parse_args()
    if args.validate and args.profile:
        parser.error("Run reference validation separately from profiling")
    if args.variant in ("direct", "direct_fast") and args.parallel_size != 2:
        parser.error("Direct dispatch still requires --parallel-size 2")
    if min(args.concurrencies + [args.max_running_requests, args.max_total_tokens,
                                args.output_tokens, args.repetitions]) <= 0:
        parser.error("Request, token, repetition and concurrency limits must be positive")
    if not args.validate and args.repetitions < max(args.concurrencies):
        parser.error("Use at least as many requests as the largest client concurrency")
    if args.dataset_path is not None and not args.dataset_path.is_file():
        parser.error("--dataset-path must name an existing dataset")
    if args.dataset_path is not None and not args.validate and args.output_tokens < 4:
        parser.error("SGLang's custom dataset requires at least four output tokens")
    results = args.output_dir.resolve()
    if results == REPO or REPO in results.parents:
        parser.error("Keep benchmark results outside the repository")
    results.mkdir(parents=True, exist_ok=True)
    label = args.label or args.variant + ("_validation" if args.validate else "_profile" if args.profile else "")
    output = results / f"{label}.jsonl"
    if output.exists():
        parser.error(f"Results already exist: {output}; choose a new --label")
    if not is_port_available(30000):
        parser.error("Port 30000 is occupied; leave the existing service untouched")
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT / "extension", ROOT, REPO / "python"])),
               HF_HUB_OFFLINE="1", CP_MOE_VARIANT=args.variant,
               CP_MOE_VALIDATE="1" if args.validate else "0")
    server_args = list(SERVER_ARGS)
    for flag, value in [("--tp-size", args.parallel_size), ("--ep-size", args.parallel_size),
                        ("--attn-cp-size", args.parallel_size),
                        ("--max-running-requests", args.max_running_requests),
                        ("--max-total-tokens", args.max_total_tokens)]:
        server_args[server_args.index(flag) + 1] = str(value)
    server_args += ["--prefill-max-requests", "1", "--decode-log-interval", "1"]
    profile_args = (["--profile", "--profile-activities", "CPU", "GPU", "--profile-by-stage",
                     "--profile-num-steps", "2", "--profile-output-dir", str(results / "traces"),
                     "--profile-prefix", label] if args.profile else [])
    with (results / f"{label}_server.log").open("w") as log:
        server = popen_launch_server(args.model, "http://127.0.0.1:30000", timeout=1200,
                                    other_args=server_args, env=env,
                                    return_stdout_stderr=(log, log))
        try:
            output_tokens = ("4" if args.dataset_path is not None else "1") if args.validate else str(args.output_tokens)
            for length in ([None] if args.dataset_path is not None else args.lengths):
                dataset_args = (["--dataset-name", "custom", "--dataset-path", str(args.dataset_path.resolve()),
                                 "--sharegpt-output-len", output_tokens]
                                if args.dataset_path is not None else
                                ["--dataset-name", "random-ids", "--tokenize-prompt",
                                 "--random-input-len", str(length), "--random-output-len", output_tokens,
                                 "--random-range-ratio", "1"])
                for concurrency in args.concurrencies:
                    subprocess.run([
                        sys.executable, "-m", "sglang.benchmark.serving", "--backend", "sglang",
                        "--host", "127.0.0.1", "--port", "30000", "--model", args.model,
                        *dataset_args, *profile_args,
                        "--num-prompts", "1" if args.validate else str(args.repetitions),
                        "--max-concurrency", str(concurrency),
                        "--warmup-requests", "0" if args.validate else "2",
                        "--seed", str(args.seed), "--temperature", "0", "--cache-report",
                        "--output-details", "--output-file", str(output), "--disable-tqdm",
                    ], check=True, env=env)
        finally:
            kill_process_tree(server.pid)


if __name__ == "__main__":
    main()
