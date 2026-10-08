"""Single-client SGLang serving baseline; no synthetic expert replacement.

Uses exact input IDs derived from pinned repository source text. Streaming TTFT
includes HTTP, scheduling, prefill and first-token delivery, not just GPU kernels.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo) if hi != lo else ordered[lo]


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def request(base_url, ids, output_tokens):
    payload = dict(input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=output_tokens,
                   ignore_eos=True), stream=True)
    req = urllib.request.Request(base_url + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    final = None
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            count = event.get("meta_info", {}).get("completion_tokens", 0)
            if count > 0 and first is None:
                first = time.perf_counter()
            final = event
    end = time.perf_counter()
    if first is None or final is None:
        raise RuntimeError("No generated-token SSE event received")
    meta = final["meta_info"]
    if meta.get("prompt_tokens") != len(ids) or meta.get("completion_tokens") != output_tokens:
        raise RuntimeError(f"Token count mismatch: {meta}")
    if meta.get("cached_tokens", 0) != 0:
        raise RuntimeError(f"Cached prefill invalidates uncached baseline: {meta}")
    return dict(ttft_ms=(first - start) * 1000, e2e_ms=(end - start) * 1000,
                prompt_tokens=meta["prompt_tokens"], completion_tokens=meta["completion_tokens"],
                cached_tokens=meta.get("cached_tokens", 0), finish_reason=meta.get("finish_reason"),
                output_text=final.get("text", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    repo = Path(args.repo)
    corpus_paths = ["python/sglang/srt/models/qwen3_moe.py", "python/sglang/srt/models/qwen2_moe.py",
                    "python/sglang/srt/layers/layer_boundary/boundary.py"]
    corpus = "Explain the following distributed inference source code.\n" + "\n".join(
        (repo / path).read_text(encoding="utf-8") for path in corpus_paths)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    if not tokens:
        raise RuntimeError("Empty prompt corpus")
    maximum = max(args.lengths)
    tokens = (tokens * ((maximum + len(tokens) - 1) // len(tokens)))[:maximum]
    raw_server_info = get_json(args.base_url + "/server_info")
    safe_keys = {"model_path", "revision", "dtype", "quantization", "tp_size", "ep_size",
                 "attn_cp_size", "moe_dp_size", "enable_prefill_cp", "cp_strategy",
                 "moe_runner_backend", "moe_a2a_backend", "attention_backend", "boundary_reduction",
                 "disable_radix_cache", "chunked_prefill_size", "context_length", "max_total_tokens",
                 "max_running_requests", "disable_overlap_schedule", "cuda_graph_backend_decode",
                 "cuda_graph_backend_prefill"}
    server_info = {"version": raw_server_info.get("version"),
                   "server_args": {key: value for key, value in raw_server_info.get("server_args", raw_server_info).items()
                                   if key in safe_keys}}
    rows = []
    for length in args.lengths:
        ids = tokens[:length]
        for _ in range(args.warmup):
            request(args.base_url, ids, args.output_tokens)
        samples = []
        for iteration in range(args.repetitions):
            sample = request(args.base_url, ids, args.output_tokens)
            sample["iteration"] = iteration
            samples.append(sample)
            print(f"{args.label} N={length} iteration={iteration} "
                  f"TTFT={sample['ttft_ms']:.3f}ms E2E={sample['e2e_ms']:.3f}ms", flush=True)
        ttfts = [sample["ttft_ms"] for sample in samples]
        durations = [sample["e2e_ms"] for sample in samples]
        rows.append(dict(input_tokens=length, output_tokens=args.output_tokens, warmup=args.warmup,
                         repetitions=args.repetitions, prompt_ids_sha256=hashlib.sha256(
                             json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                         median_ttft_ms=statistics.median(ttfts), mean_ttft_ms=statistics.mean(ttfts),
                         p95_ttft_ms=percentile(ttfts, 0.95), median_e2e_ms=statistics.median(durations),
                         serial_output_tokens_per_second=sum(sample["completion_tokens"] for sample in samples)
                             / (sum(durations) / 1000),
                         samples=samples))
    result = dict(label=args.label, scope="real SGLang serving, one sequential client, uncached code prompts",
                  model_path=args.model, git_commit=subprocess.check_output(
                      ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                  corpus_paths=corpus_paths, server_info=server_info, cases=rows)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"label": args.label, "cases": [{k: v for k, v in row.items() if k != "samples"}
                                                       for row in rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
