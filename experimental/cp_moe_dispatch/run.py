"""Fixed CP2+EP2 configuration; reuse SGLang's launcher and serving benchmark."""

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
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 16384])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", help="Use a unique label for each run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/cp_moe_native_results"))
    args = parser.parse_args()
    results = args.output_dir.resolve()
    if results == REPO or REPO in results.parents:
        parser.error("Keep benchmark results outside the repository")
    results.mkdir(parents=True, exist_ok=True)
    label = args.label or args.variant + ("_validation" if args.validate else "")
    output = results / f"{label}.jsonl"
    if output.exists():
        parser.error(f"Results already exist: {output}; choose a new --label")
    if not is_port_available(30000):
        parser.error("Port 30000 is occupied; leave the existing service untouched")
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT / "extension", ROOT, REPO / "python"])),
               HF_HUB_OFFLINE="1", CP_MOE_VARIANT=args.variant,
               CP_MOE_VALIDATE="1" if args.validate else "0")
    with (results / f"{label}_server.log").open("w") as log:
        server = popen_launch_server(args.model, "http://127.0.0.1:30000", timeout=1200,
                                    other_args=SERVER_ARGS, env=env,
                                    return_stdout_stderr=(log, log))
        try:
            for length in args.lengths:
                subprocess.run([
                    sys.executable, "-m", "sglang.benchmark.serving", "--backend", "sglang",
                    "--host", "127.0.0.1", "--port", "30000", "--model", args.model,
                    "--dataset-name", "random-ids", "--tokenize-prompt",
                    "--random-input-len", str(length), "--random-output-len", "1" if args.validate else "32",
                    "--random-range-ratio", "1", "--num-prompts", "1" if args.validate else str(args.repetitions),
                    "--max-concurrency", "1", "--warmup-requests", "0" if args.validate else "2",
                    "--seed", str(args.seed), "--temperature", "0", "--cache-report",
                    "--output-details", "--output-file", str(output), "--disable-tqdm",
                ], check=True, env=env)
        finally:
            kill_process_tree(server.pid)


if __name__ == "__main__":
    main()
