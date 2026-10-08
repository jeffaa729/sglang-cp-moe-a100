"""One-command reproduction of the measured serving comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run import MODEL, RESULTS, prepare_results_dir

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--output-dir', type=Path, default=RESULTS)
    args = parser.parse_args()
    results = prepare_results_dir(args.output_dir)
    runs = [('baseline', 'baseline'), ('rs', 'rs'), ('direct', 'direct'),
            ('direct_fast', 'direct_fast'), ('rs', 'rs_repeat'),
            ('baseline', 'baseline_repeat')]
    for variant, label in runs:
        print(f'Benchmarking {label}...', flush=True)
        with (results/f'{label}_client.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'run.py'), variant, '--label', label,
                            '--model', args.model, '--output-dir', str(results)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT/'analyze.py'), '--output-dir', str(results)], check=True,
                   stdout=subprocess.DEVNULL)
    result = json.loads((results/'summary.json').read_text())
    print('Input tokens | Baseline TTFT ms | Direct-fast ms | Reduce-scatter ms')
    for row, direct in zip(result['combined_baseline_vs_rs'], result['serving']['direct_fast']):
        print(f"{row['tokens']:12} | {row['before_ttft_ms']:16.1f} | "
              f"{direct['after_ttft_ms']:14.1f} | {row['after_ttft_ms']:17.1f}")


if __name__ == '__main__':
    main()
