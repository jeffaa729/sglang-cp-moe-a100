"""Reproduce the fixed CP2+EP2 serving comparison; dependencies preinstalled."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODEL = '/workspace/models/Qwen3-30B-A3B-NVFP4'
BASE = ROOT
RESULTS = Path('/workspace/cp_moe_dispatch_results')
PINNED_PYTHON_TREE = '9c975ab4a2081b89c6e29806b6ded0d3c1ce122b'


def verify_source(repo):
    """Accept extension commits, but reject changes to the measured runtime."""
    tree = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD:python'], text=True).strip()
    if tree != PINNED_PYTHON_TREE:
        raise RuntimeError('Python runtime differs from the measured baseline')
    dirty = subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all', '--', 'python'],
        text=True).strip()
    if dirty:
        raise RuntimeError('Python runtime has uncommitted changes')


def prepare_results_dir(path):
    results = Path(path).resolve()
    repo = REPO.resolve()
    if results == repo or repo in results.parents:
        raise ValueError('Benchmark results must stay outside the repository')
    results.mkdir(parents=True, exist_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['baseline', 'rs', 'direct', 'direct_fast'])
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--lengths', nargs='+', type=int, default=[2048, 8192, 16384])
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--label')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--output-dir', type=Path, default=RESULTS)
    args = parser.parse_args()
    verify_source(REPO)
    results = prepare_results_dir(args.output_dir)
    model = args.model
    label = args.label or args.variant + ('_validation' if args.validate else '')
    try:
        connection = socket.create_connection(('127.0.0.1', 30000), timeout=2)
    except OSError:
        pass
    else:
        connection.close()
        raise RuntimeError('Port 30000 already has a server; refusing to replace it')
    env = dict(os.environ)
    env.update(PYTHONPATH=f'{ROOT / "extension"}:{ROOT}:{REPO / "python"}',
               HF_HUB_OFFLINE='1', CP_MOE_VARIANT=args.variant,
               CP_MOE_VALIDATE='1' if args.validate else '0')
    for name in ('CP_MOE_SHAPE_PROBE', 'CP_MOE_EP_DIAGNOSTIC', 'CUDA_LAUNCH_BLOCKING'):
        env.pop(name, None)
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', model,
               '--host', '127.0.0.1', '--port', '30000', '--tp-size', '2', '--ep-size', '2',
               '--moe-dp-size', '1', '--attn-cp-size', '2', '--enable-prefill-cp',
               '--cp-strategy', 'zigzag', '--dtype', 'bfloat16', '--quantization', 'modelopt_fp4',
               '--moe-a2a-backend', 'none', '--moe-runner-backend', 'flashinfer_cutlass',
               '--disable-flashinfer-cutlass-moe-fp4-allgather', '--attention-backend', 'fa4',
               '--boundary-reduction', 'ar', '--kv-cache-dtype', 'bfloat16',
               '--context-length', '18432', '--max-total-tokens', '18432',
               '--max-running-requests', '1', '--chunked-prefill-size', '-1',
               '--max-prefill-tokens', '18432', '--mem-fraction-static', '0.85',
               '--disable-radix-cache', '--disable-overlap-schedule', '--disable-custom-all-reduce',
               '--cuda-graph-backend-prefill', 'disabled', '--cuda-graph-backend-decode', 'disabled']
    with (results / f'{label}_server.log').open('w') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited: see {label}_server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=5) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError('Server startup timeout')
            client = [sys.executable, str(BASE / 'benchmark_client.py'), '--model', model,
                      '--repo', str(REPO), '--label', label, '--lengths', *map(str, args.lengths),
                      '--warmup', '0' if args.validate else '2',
                      '--repetitions', '1' if args.validate else str(args.repetitions),
                      '--output-tokens', '1' if args.validate else '32',
                      '--output', str(results / f'{label}.json')]
            subprocess.run(client, check=True, env=env)
            if not args.validate:
                subprocess.run([sys.executable, str(BASE / 'sanity_client.py'), '--model', model,
                                '--output', str(results / f'{label}_sanity.json')], check=True, env=env)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'Experiment process group {server.pid} did not stop; inspect it')
            time.sleep(5)


if __name__ == '__main__':
    main()
