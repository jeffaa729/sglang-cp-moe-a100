import io
import json
import unittest
from unittest.mock import patch
import benchmark_client as client


class ClientTests(unittest.TestCase):
    def response(self, *, cached=0, prompt=4, completion=1):
        event = dict(text='4', meta_info=dict(prompt_tokens=prompt,
                     completion_tokens=completion, cached_tokens=cached))
        return io.BytesIO(('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n').encode())

    def test_streaming_timing_and_token_contract(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=self.response()), \
             patch.object(client.time, 'perf_counter', side_effect=[0, 0.1, 0.3]):
            result = client.request('http://localhost', [1, 2, 3, 4], 1)
        self.assertAlmostEqual(result['ttft_ms'], 100)
        self.assertAlmostEqual(result['e2e_ms'], 300)
        self.assertEqual(result['output_text'], '4')

    def test_cached_prefill_is_rejected(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=self.response(cached=4)):
            with self.assertRaisesRegex(RuntimeError, 'Cached prefill'):
                client.request('http://localhost', [1, 2, 3, 4], 1)

    def test_wrong_token_count_is_rejected(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=self.response(prompt=3)):
            with self.assertRaisesRegex(RuntimeError, 'Token count mismatch'):
                client.request('http://localhost', [1, 2, 3, 4], 1)

    def test_empty_generation_is_rejected(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=io.BytesIO(b'data: [DONE]\n')):
            with self.assertRaisesRegex(RuntimeError, 'No generated-token'):
                client.request('http://localhost', [1], 1)

    def test_percentile_interpolation(self):
        self.assertEqual(client.percentile([5, 1, 3], 0.5), 3)
        self.assertAlmostEqual(client.percentile([1, 2, 3, 4, 5], 0.95), 4.8)


if __name__ == '__main__':
    unittest.main()
