#!/usr/bin/env python3
"""
Azure LLM Inference Trace Replay - Async Version

Replays trace with TRUE concurrency - requests are sent at trace timestamps
without waiting for previous requests to complete.

By default, uses uniform short requests (200 context + 100 generated tokens)
to avoid failures from exceeding max_model_len.

Usage:
    # Terminal 1: Port-forward
    kubectl port-forward test-vllm-server 8000:8000

    # Terminal 2: Run async replay (uniform short requests)
    python trace-replay-async.py --limit 1000

    # Multiple concurrent users (builds queue naturally!)
    python trace-replay-async.py --users 10 --limit 1000

    # 10 users + faster speed (heavy load!)
    python trace-replay-async.py --users 10 --speed 10.0 --limit 1000

    # With custom uniform sizes
    python trace-replay-async.py --uniform-context 500 --uniform-generated 200 --limit 1000

    # Use actual trace sizes (may fail on large contexts)
    python trace-replay-async.py --use-trace-sizes --limit 1000
"""

import csv
import time
import asyncio
import aiohttp
from datetime import datetime
import argparse
import sys
from collections import defaultdict


class AsyncTraceReplayer:
    """Async trace replayer with true concurrency"""

    def __init__(self, host, trace_file, speed=1.0, limit=None, uniform_size=None, users=1):
        self.host = host
        self.trace_file = trace_file
        self.speed = speed
        self.limit = limit
        self.uniform_size = uniform_size  # (context_tokens, generated_tokens) or None
        self.users = users  # Number of concurrent users

        # Results storage (thread-safe with asyncio)
        self.results = []
        self.queue_metrics = []

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.in_flight_requests = 0

        # Lock for stats
        self.stats_lock = asyncio.Lock()

    def load_trace(self):
        """Load trace from CSV"""
        requests_data = []

        with open(self.trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if self.limit and idx >= self.limit:
                    break

                # Parse timestamp with variable fractional seconds
                timestamp_str = row['TIMESTAMP'].rstrip('Z')

                # Pad fractional seconds to 6 digits if present
                if '.' in timestamp_str:
                    parts = timestamp_str.split('.')
                    fractional = parts[1].ljust(6, '0')[:6]
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

    async def get_queue_metrics(self, session):
        """Fetch current queue metrics from vLLM"""
        try:
            async with session.get(f"{self.host}/metrics", timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status != 200:
                    return None

                text = await response.text()
                metrics = {}
                for line in text.split('\n'):
                    if line.startswith('vllm:num_requests_waiting'):
                        metrics['waiting'] = float(line.split()[-1])
                    elif line.startswith('vllm:num_requests_running'):
                        metrics['running'] = float(line.split()[-1])

                return metrics
        except Exception:
            return None

    async def send_request(self, session, request_id, context_tokens, generated_tokens):
        """Send a single request asynchronously"""
        # Generate prompt
        prompt_length = context_tokens * 4
        prompt = "A " * (prompt_length // 2)

        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": prompt,
            "max_tokens": generated_tokens,
            "temperature": 0.7,
        }

        start_time = time.time()

        # Increment in-flight counter
        async with self.stats_lock:
            self.in_flight_requests += 1

        try:
            async with session.post(
                f"{self.host}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                latency = time.time() - start_time
                text = await response.text()

                if response.status == 200:
                    try:
                        data = await response.json()
                        actual_tokens = data.get('usage', {}).get('completion_tokens', 0)
                    except:
                        actual_tokens = 0

                    # Calculate throughput (tokens/s)
                    tokens_per_sec = actual_tokens / latency if latency > 0 else 0

                    result = {
                        'request_id': request_id,
                        'success': True,
                        'latency': latency,
                        'context_tokens': context_tokens,
                        'generated_tokens': generated_tokens,
                        'actual_tokens': actual_tokens,
                        'tokens_per_sec': tokens_per_sec,
                        'status_code': 200
                    }

                    async with self.stats_lock:
                        self.successful_requests += 1
                else:
                    result = {
                        'request_id': request_id,
                        'success': False,
                        'latency': latency,
                        'context_tokens': context_tokens,
                        'generated_tokens': generated_tokens,
                        'tokens_per_sec': 0,
                        'status_code': response.status,
                        'error': text[:100]
                    }

                    async with self.stats_lock:
                        self.failed_requests += 1

        except Exception as e:
            latency = time.time() - start_time
            result = {
                'request_id': request_id,
                'success': False,
                'latency': latency,
                'context_tokens': context_tokens,
                'generated_tokens': generated_tokens,
                'tokens_per_sec': 0,
                'error': str(e)[:100]
            }

            async with self.stats_lock:
                self.failed_requests += 1

        # Decrement in-flight counter
        async with self.stats_lock:
            self.in_flight_requests -= 1

        return result

    async def monitor_queue(self, session, duration):
        """Continuously monitor queue metrics"""
        start_time = time.time()

        while time.time() - start_time < duration + 30:  # Extra 30s for completion
            metrics = await self.get_queue_metrics(session)
            if metrics:
                self.queue_metrics.append({
                    'timestamp': time.time(),
                    'waiting': metrics['waiting'],
                    'running': metrics['running']
                })

            await asyncio.sleep(1)  # Poll every 1 second

    async def replay(self):
        """Replay trace with true concurrency"""
        requests_data = self.load_trace()

        if not requests_data:
            print("❌ No requests to replay")
            return

        print(f"\n🚀 Starting ASYNC trace replay at {self.speed}x speed with {self.users} concurrent users")
        print(f"   Target: {self.host}")
        print(f"   Total requests: {len(requests_data)} (each user will replay all)")
        print(f"   Requests will be sent concurrently based on trace timestamps")
        print("-" * 80)

        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=2000)  # Allow up to 2000 concurrent connections
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            start_time = time.time()
            base_timestamp = requests_data[0]['timestamp']

            # Calculate total duration
            last_timestamp = requests_data[-1]['timestamp']
            total_duration = (last_timestamp - base_timestamp).total_seconds() / self.speed

            # Start queue monitoring in background
            monitor_task = asyncio.create_task(self.monitor_queue(session, total_duration))

            # Create user tasks - each user replays the full trace
            user_tasks = []
            for user_id in range(self.users):
                user_task = asyncio.create_task(
                    self.replay_user(session, user_id, requests_data, start_time, base_timestamp)
                )
                user_tasks.append(user_task)

            # Wait for all users to complete
            all_results = await asyncio.gather(*user_tasks, return_exceptions=True)

            # Cancel monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Flatten results from all users
            self.results = []
            for user_results in all_results:
                if not isinstance(user_results, Exception):
                    self.results.extend(user_results)

            self.total_requests = len(self.results)

        print("-" * 80)
        self.print_summary()

    async def replay_user(self, session, user_id, requests_data, start_time, base_timestamp):
        """Single user replaying the trace"""
        tasks = []
        for idx, req in enumerate(requests_data):
            # Calculate when this request should be sent
            trace_offset = (req['timestamp'] - base_timestamp).total_seconds()
            target_time = start_time + (trace_offset / self.speed)

            # Create task that waits then sends
            global_idx = user_id * len(requests_data) + idx
            task = asyncio.create_task(
                self.scheduled_request(
                    session, target_time, global_idx,
                    req['context_tokens'], req['generated_tokens'],
                    len(requests_data) * self.users,
                    user_id
                )
            )
            tasks.append(task)

        # Wait for all this user's requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def scheduled_request(self, session, target_time, request_id,
                                context_tokens, generated_tokens, total_requests, user_id=0):
        """Wait until target time, then send request"""
        # Wait until target time
        wait_time = target_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # Use uniform size if specified
        if self.uniform_size:
            context_tokens, generated_tokens = self.uniform_size

        # Send request (non-blocking - returns immediately)
        result = await self.send_request(session, request_id, context_tokens, generated_tokens)

        # Print progress (every 10 requests or if failed)
        if (request_id + 1) % 10 == 0 or not result['success']:
            status = "✅" if result['success'] else "❌"
            async with self.stats_lock:
                in_flight = self.in_flight_requests

            # Get latest queue metrics
            latest_queue = self.queue_metrics[-1] if self.queue_metrics else None
            if latest_queue:
                queue_str = f"Queue: {latest_queue['waiting']:.0f}W/{latest_queue['running']:.0f}R"
            else:
                queue_str = "Queue: N/A"

            user_prefix = f"U{user_id}" if self.users > 1 else ""
            tok_per_sec = result.get('tokens_per_sec', 0)
            print(f"[{request_id+1}/{total_requests}] {user_prefix} {status} "
                  f"Ctx:{context_tokens:4d} Gen:{generated_tokens:4d} "
                  f"Latency:{result['latency']:6.2f}s "
                  f"Tok/s:{tok_per_sec:6.1f} "
                  f"InFlight:{in_flight:3d} {queue_str}")

        return result

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
            throughputs = [r['tokens_per_sec'] for r in successful_results if r.get('tokens_per_sec', 0) > 0]

            print(f"\n⏱️  Latency Statistics")
            print(f"Mean:                {sum(latencies)/len(latencies):.2f}s")
            print(f"Min:                 {min(latencies):.2f}s")
            print(f"Max:                 {max(latencies):.2f}s")
            print(f"p50:                 {sorted(latencies)[len(latencies)//2]:.2f}s")
            print(f"p95:                 {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
            print(f"p99:                 {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")

            if throughputs:
                print(f"\n🚀 Throughput Statistics (tokens/s)")
                print(f"Mean:                {sum(throughputs)/len(throughputs):.1f} tok/s")
                print(f"Min:                 {min(throughputs):.1f} tok/s")
                print(f"Max:                 {max(throughputs):.1f} tok/s")
                print(f"p50:                 {sorted(throughputs)[len(throughputs)//2]:.1f} tok/s")
                print(f"p95:                 {sorted(throughputs)[int(len(throughputs)*0.95)]:.1f} tok/s")

        # Queue statistics
        if self.queue_metrics:
            waiting = [q['waiting'] for q in self.queue_metrics]
            running = [q['running'] for q in self.queue_metrics]

            print(f"\n📈 Queue Statistics")
            print(f"Avg Waiting:         {sum(waiting)/len(waiting):.1f}")
            print(f"Max Waiting:         {max(waiting):.0f}")
            print(f"Avg Running:         {sum(running)/len(running):.1f}")
            print(f"Max Running:         {max(running):.0f}")

    def save_results(self, output_file):
        """Save detailed results to CSV"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'request_id', 'success', 'latency', 'context_tokens', 'generated_tokens', 'tokens_per_sec'
            ])
            writer.writeheader()

            for r in self.results:
                writer.writerow({
                    'request_id': r['request_id'],
                    'success': r['success'],
                    'latency': r['latency'],
                    'context_tokens': r['context_tokens'],
                    'generated_tokens': r['generated_tokens'],
                    'tokens_per_sec': r.get('tokens_per_sec', 0),
                })

        print(f"\n💾 Results saved to: {output_file}")

        # Save queue metrics
        queue_file = output_file.replace('.csv', '_queue.csv')
        with open(queue_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'waiting', 'running'])
            writer.writeheader()
            for q in self.queue_metrics:
                writer.writerow(q)

        print(f"💾 Queue metrics saved to: {queue_file}")


async def main_async():
    parser = argparse.ArgumentParser(description='Async replay of Azure LLM trace')
    parser.add_argument('--host', default='http://localhost:8000',
                        help='vLLM server URL')
    parser.add_argument('--trace-file',
                        default='/home/skt6g/AI-RAN/KubeSMO/data/AzureLMMInferenceTrace_multimodal.csv',
                        help='Path to Azure trace CSV')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Replay speed (1.0=real-time, 10.0=10x faster)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of requests (for testing)')
    parser.add_argument('--output', default='async_trace_results.csv',
                        help='Output CSV file')
    parser.add_argument('--uniform-context', type=int, default=200,
                        help='Use uniform context token size (default: 200)')
    parser.add_argument('--uniform-generated', type=int, default=100,
                        help='Use uniform generated token size (default: 100)')
    parser.add_argument('--use-trace-sizes', action='store_true',
                        help='Use actual token sizes from trace (default: use uniform sizes)')
    parser.add_argument('--users', type=int, default=1,
                        help='Number of concurrent users (each replays full trace)')

    args = parser.parse_args()

    # Determine uniform size
    if args.use_trace_sizes:
        uniform_size = None
        print(f"📊 Using actual token sizes from trace")
    else:
        uniform_size = (args.uniform_context, args.uniform_generated)
        print(f"📊 Using uniform request size: {args.uniform_context} context + {args.uniform_generated} generated tokens")

    # Check server health
    print(f"🔍 Checking vLLM server at {args.host}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.host}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print(f"✅ Server is healthy")
                else:
                    print(f"⚠️  Server returned status {response.status}")
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        print(f"\n💡 Make sure to run: kubectl port-forward test-vllm-server 8000:8000")
        sys.exit(1)

    # Run async replay
    replayer = AsyncTraceReplayer(args.host, args.trace_file, args.speed, args.limit, uniform_size, args.users)
    await replayer.replay()
    replayer.save_results(args.output)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
