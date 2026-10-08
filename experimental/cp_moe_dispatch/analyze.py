"""Produce compact result summaries, retaining raw results separately."""
import argparse
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=Path('/workspace/cp_moe_dispatch_results'))
args = parser.parse_args()
root = args.output_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
decoder = json.JSONDecoder()
records = {}
traffic = {}
for log in root.glob('*validation*_server.log'):
    text = log.read_text(errors='replace')
    entries = []
    for match in re.finditer(r'CP_DISPATCH_CHECK\s+', text):
        record, _ = decoder.raw_decode(text[match.end():])
        entries.append(record)
    if entries:
        records[log.stem] = dict(checks=len(entries), exact=sum(r['exact'] for r in entries),
                                max_abs=max(r['max_abs'] for r in entries),
                                max_relative_l2=max(r['relative_l2'] for r in entries),
                                all_finite=all(r['finite'] for r in entries),
                                routing_mismatches=sum(r.get('routing_expert_set_mismatches',
                                                            r.get('routing_id_mismatches', 0)) for r in entries))
        (root / f'{log.stem}_checks.json').write_text(json.dumps(entries, indent=2))
        if 'v3' in log.stem or 'direct_fast' in log.stem:
            for size in sorted({sum(r['rows']) for r in entries}):
                batch = [r for r in entries if sum(r['rows']) == size and r['traffic']]
                if batch:
                    total_tokens = sum(r['traffic']['local_tokens'] for r in batch)
                    # Logical ring-algorithm accounting, NOT PCIe counter data.
                    reference_bytes = sum(3*max(r['rows'])*2048*2 for r in batch)
                    sent = sum(r['traffic']['send_payload_bytes'] for r in batch)
                    traffic[f'{log.stem}:{size}'] = dict(
                        remote_token_percent=100*sum(r['traffic']['remote_tokens'] for r in batch)/total_tokens,
                        mean_direct_send_mib=sent/len(batch)/2**20,
                        mean_reference_ring_send_mib=reference_bytes/len(batch)/2**20,
                        logical_ring_byte_reduction_percent=100*(1-sent/reference_bytes))

results = {}
if (root/'baseline.json').exists():
    baseline = json.loads((root/'baseline.json').read_text())
    for variant in ('rs', 'direct', 'direct_fast', 'rs_repeat', 'baseline_repeat'):
        file = root/f'{variant}.json'
        if not file.exists():
            continue
        result = json.loads(file.read_text())
        results[variant] = []
        for before, after in zip(baseline['cases'], result['cases']):
            assert before['prompt_ids_sha256'] == after['prompt_ids_sha256']
            assert before['input_tokens'] == after['input_tokens']
            text_equal = all(s['output_text'] == before['samples'][0]['output_text']
                             for s in before['samples'] + after['samples'])
            results[variant].append(dict(tokens=before['input_tokens'],
                before_ttft_ms=before['median_ttft_ms'], after_ttft_ms=after['median_ttft_ms'],
                ttft_reduction_percent=100*(1-after['median_ttft_ms']/before['median_ttft_ms']),
                before_e2e_ms=before['median_e2e_ms'], after_e2e_ms=after['median_e2e_ms'],
                e2e_reduction_percent=100*(1-after['median_e2e_ms']/before['median_e2e_ms']),
                generated_text_exact=text_equal))
combined = []
if all((root/f'{name}.json').exists() for name in ('baseline', 'baseline_repeat', 'rs', 'rs_repeat')):
    data = {name: json.loads((root/f'{name}.json').read_text())
            for name in ('baseline', 'baseline_repeat', 'rs', 'rs_repeat')}
    for index, case in enumerate(data['baseline']['cases']):
        row = dict(tokens=case['input_tokens'], samples_per_variant=10)
        for metric in ('ttft_ms', 'e2e_ms'):
            before = [s[metric] for name in ('baseline', 'baseline_repeat')
                      for s in data[name]['cases'][index]['samples']]
            after = [s[metric] for name in ('rs', 'rs_repeat')
                     for s in data[name]['cases'][index]['samples']]
            row[f'before_{metric}'] = statistics.median(before)
            row[f'after_{metric}'] = statistics.median(after)
            row[f'{metric}_reduction_percent'] = 100*(1-statistics.median(after)/statistics.median(before))
        combined.append(row)
summary = dict(validation=records, serving=results, combined_baseline_vs_rs=combined, traffic=traffic)
(root/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
