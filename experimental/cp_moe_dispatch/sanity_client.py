"""Untimed generation/log-probability sanity check; not an accuracy benchmark."""
import argparse
import json
import math
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--model', default='/workspace/models/Qwen3-30B-A3B-GPTQ-Int4')
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
ids = tokenizer.apply_chat_template([dict(role='user', content='What is 2 + 2? Reply with one digit.')],
                                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
if hasattr(ids, 'keys'):
    ids = ids['input_ids']
payload = dict(input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=16),
               return_logprob=True, top_logprobs_num=3, stream=False)
req = urllib.request.Request('http://127.0.0.1:30000/generate', data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=900) as response:
    result = json.load(response)
values = [entry[0] for entry in result.get('meta_info', {}).get('output_token_logprobs', [])]
record = dict(prompt_tokens=len(ids), result=result,
              logprobs_present=bool(values), finite_output_logprobs=bool(values) and all(
                  isinstance(value, (int, float)) and math.isfinite(value) for value in values),
              expected_simple_answer='4')
Path(args.output).write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
