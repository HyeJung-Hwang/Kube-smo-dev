#!/usr/bin/env python3
"""
Azure LLM Inference Trace Replay for vLLM Server

Replays real Azure inference trace with realistic request patterns.
Measures per-request latency and monitors queue metrics.

Usage:
    # Terminal 1: Port-forward
    kubectl port-forward test-vllm-server 8000:8000

    # Terminal 2: Run trace replay
    python locust-trace-replay.py \
      --host http://localhost:8000 \
      --trace-file /home/skt6g/AI-RAN/KubeSMO/data/AzureLMMInferenceTrace_multimodal.csv \
      --speed 1.0
"""

import csv
import time
import requests
from datetime import datetime
from collections import defaultdict
import argparse
import sys


class TraceReplayer:
    """Replays Azure trace and measures latency + queue metrics"""

    def __init__(self, host, trace_file, speed=1.0):
        self.host = host
        self.trace_file = trace_file
        self.speed = speed  # 1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower

        # Results storage
        self.results = []
        self.queue_metrics = []

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    def load_trace(self):
        """Load trace from CSV"""
        requests_data = []

        with open(self.trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse timestamp with variable fractional seconds
                timestamp_str = row['TIMESTAMP'].rstrip('Z')

                # Pad fractional seconds to 6 digits if present
                if '.' in timestamp_str:
                    parts = timestamp_str.split('.')
                    fractional = parts[1].ljust(6, '0')[:6]  # Pad to 6 digits
                    timestamp_str = f"{parts[0]}.{fractional}+00:00"
                else:
                    timestamp_str = f"{timestamp_str}+00:00"

                requests_data.append({
                    'timestamp': datetime.fromisoformat(timestamp_str),
                    'context_tokens': int(row['ContextTokens']),
                    'generated_tokens': int(row['GeneratedTokens']),
                    'num_images': int(row['NumImages'])
                })

        print(f"✅ Loaded {len(requests_data)} requests from trace")
        return requests_data

    def get_queue_metrics(self):
        """Fetch current queue metrics from vLLM"""
        try:
            response = requests.get(f"{self.host}/metrics", timeout=2)
            if response.status_code != 200:
                return None

            metrics = {}
            for line in response.text.split('\n'):
                if line.startswith('vllm:num_requests_waiting'):
                    metrics['waiting'] = float(line.split()[-1])
                elif line.startswith('vllm:num_requests_running'):
                    metrics['running'] = float(line.split()[-1])

            return metrics
        except Exception as e:
            print(f"⚠️  Failed to fetch metrics: {e}")
            return None

    def send_request(self, context_tokens, generated_tokens):
        """Send a single request and measure latency"""
        # Generate prompt of approximate token count
        # Rough approximation: 1 token ≈ 4 characters
        prompt_length = context_tokens * 4
        prompt = "A " * (prompt_length // 2)  # Simple repetitive prompt

        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": prompt,
            "max_tokens": generated_tokens,
            "temperature": 0.7,
        }

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.host}/v1/completions",
                json=payload,
                timeout=300  # 5 min timeout for long requests
            )

            latency = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                actual_tokens = data.get('usage', {}).get('completion_tokens', 0)
                return {
                    'success': True,
                    'latency': latency,
                    'context_tokens': context_tokens,
                    'generated_tokens': generated_tokens,
                    'actual_tokens': actual_tokens,
                    'status_code': 200
                }
            else:
                return {
                    'success': False,
                    'latency': latency,
                    'context_tokens': context_tokens,
                    'generated_tokens': generated_tokens,
                    'status_code': response.status_code,
                    'error': response.text[:100]
                }

        except Exception as e:
            latency = time.time() - start_time
            return {
                'success': False,
                'latency': latency,
                'context_tokens': context_tokens,
                'generated_tokens': generated_tokens,
                'error': str(e)
            }

    def replay(self):
        """Replay trace requests with timing"""
        requests_data = self.load_trace()

        if not requests_data:
            print("❌ No requests to replay")
            return

        print(f"\n🚀 Starting trace replay at {self.speed}x speed")
        print(f"   Target: {self.host}")
        print("-" * 80)

        start_time = time.time()
        base_timestamp = requests_data[0]['timestamp']

        for idx, req in enumerate(requests_data):
            # Calculate when this request should be sent
            trace_offset = (req['timestamp'] - base_timestamp).total_seconds()
            target_time = start_time + (trace_offset / self.speed)

            # Wait until target time
            wait_time = target_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            # Get queue metrics BEFORE sending request
            queue_before = self.get_queue_metrics()

            # Send request
            result = self.send_request(req['context_tokens'], req['generated_tokens'])

            # Get queue metrics AFTER sending request
            queue_after = self.get_queue_metrics()

            # Store results
            self.total_requests += 1
            if result['success']:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            result['request_id'] = idx
            result['queue_before'] = queue_before
            result['queue_after'] = queue_after
            self.results.append(result)

            # Print progress
            status = "✅" if result['success'] else "❌"
            queue_str = f"Queue: {queue_after['waiting']:.0f}W/{queue_after['running']:.0f}R" if queue_after else "Queue: N/A"

            print(f"[{idx+1}/{len(requests_data)}] {status} "
                  f"Ctx:{req['context_tokens']:4d} Gen:{req['generated_tokens']:4d} "
                  f"Latency:{result['latency']:6.2f}s {queue_str}")

        print("-" * 80)
        self.print_summary()

    def print_summary(self):
        """Print test summary statistics"""
        print(f"\n📊 Test Summary")
        print("=" * 80)
        print(f"Total Requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests} ({self.successful_requests/self.total_requests*100:.1f}%)")
        print(f"Failed:              {self.failed_requests} ({self.failed_requests/self.total_requests*100:.1f}%)")

        if self.successful_requests > 0:
            successful_results = [r for r in self.results if r['success']]
            latencies = [r['latency'] for r in successful_results]

            print(f"\n⏱️  Latency Statistics")
            print(f"Mean:                {sum(latencies)/len(latencies):.2f}s")
            print(f"Min:                 {min(latencies):.2f}s")
            print(f"Max:                 {max(latencies):.2f}s")
            print(f"p50:                 {sorted(latencies)[len(latencies)//2]:.2f}s")
            print(f"p95:                 {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
            print(f"p99:                 {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")

        # Queue statistics
        queue_data = [r['queue_after'] for r in self.results if r.get('queue_after')]
        if queue_data:
            waiting = [q['waiting'] for q in queue_data]
            running = [q['running'] for q in queue_data]

            print(f"\n📈 Queue Statistics")
            print(f"Avg Waiting:         {sum(waiting)/len(waiting):.1f}")
            print(f"Max Waiting:         {max(waiting):.0f}")
            print(f"Avg Running:         {sum(running)/len(running):.1f}")
            print(f"Max Running:         {max(running):.0f}")

    def save_results(self, output_file):
        """Save detailed results to CSV"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'request_id', 'success', 'latency', 'context_tokens', 'generated_tokens',
                'queue_waiting_before', 'queue_running_before',
                'queue_waiting_after', 'queue_running_after'
            ])
            writer.writeheader()

            for r in self.results:
                writer.writerow({
                    'request_id': r['request_id'],
                    'success': r['success'],
                    'latency': r['latency'],
                    'context_tokens': r['context_tokens'],
                    'generated_tokens': r['generated_tokens'],
                    'queue_waiting_before': r['queue_before']['waiting'] if r.get('queue_before') else None,
                    'queue_running_before': r['queue_before']['running'] if r.get('queue_before') else None,
                    'queue_waiting_after': r['queue_after']['waiting'] if r.get('queue_after') else None,
                    'queue_running_after': r['queue_after']['running'] if r.get('queue_after') else None,
                })

        print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Replay Azure LLM trace on vLLM server')
    parser.add_argument('--host', default='http://localhost:8000',
                        help='vLLM server URL (default: http://localhost:8000)')
    parser.add_argument('--trace-file',
                        default='/home/skt6g/AI-RAN/KubeSMO/data/AzureLMMInferenceTrace_multimodal.csv',
                        help='Path to Azure trace CSV file')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Replay speed multiplier (1.0=real-time, 2.0=2x faster)')
    parser.add_argument('--output', default='trace_replay_results.csv',
                        help='Output CSV file for results')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of requests to replay (for testing)')

    args = parser.parse_args()

    # Check server health
    print(f"🔍 Checking vLLM server at {args.host}")
    try:
        response = requests.get(f"{args.host}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is healthy")
        else:
            print(f"⚠️  Server returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        print(f"\n💡 Make sure to run: kubectl port-forward test-vllm-server 8000:8000")
        sys.exit(1)

    # Run replay
    replayer = TraceReplayer(args.host, args.trace_file, args.speed)
    replayer.replay()
    replayer.save_results(args.output)


if __name__ == "__main__":
    main()
