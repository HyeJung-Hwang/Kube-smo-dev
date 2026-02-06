#!/usr/bin/env python3
"""
ShareGPT Trace Replay for vLLM Server

Uses real conversation data from ShareGPT dataset to generate realistic inference traffic.
Sends actual human messages as prompts to vLLM server.

Usage:
    # Terminal 1: Port-forward
    kubectl port-forward test-vllm-server 8000:8000

    # Terminal 2: Run ShareGPT replay
    python sharegpt-replay.py --limit 1000 --users 10
"""

import json
import time
import asyncio
import aiohttp
from datetime import datetime
import argparse
import sys
import csv
import random


class ShareGPTReplayer:
    """Replays ShareGPT conversations with realistic traffic patterns"""

    def __init__(self, host, sharegpt_file, speed=1.0, limit=None, users=1, max_tokens=100):
        self.host = host
        self.sharegpt_file = sharegpt_file
        self.speed = speed
        self.limit = limit
        self.users = users
        self.max_tokens = max_tokens

        # Results storage
        self.results = []
        self.queue_metrics = []

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.in_flight_requests = 0

        # Lock for stats
        self.stats_lock = asyncio.Lock()

    def load_conversations(self):
        """Load conversations from ShareGPT JSON"""
        conversations = []

        print(f"📖 Loading ShareGPT conversations from {self.sharegpt_file}")
        with open(self.sharegpt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract human messages
        for item in data:
            if self.limit and len(conversations) >= self.limit:
                break

            conv = item.get('conversations', [])
            for msg in conv:
                if msg.get('from') == 'human':
                    prompt = msg.get('value', '').strip()
                    if prompt and len(prompt) > 10:  # Skip very short prompts
                        conversations.append(prompt)

                        if self.limit and len(conversations) >= self.limit:
                            break

        print(f"✅ Loaded {len(conversations)} human prompts")
        return conversations

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

    async def send_request(self, session, request_id, prompt):
        """Send a single request asynchronously"""
        start_time = time.time()

        # Increment in-flight counter
        async with self.stats_lock:
            self.in_flight_requests += 1

        try:
            payload = {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0.7,
            }

            async with session.post(
                f"{self.host}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                latency = time.time() - start_time

                if response.status == 200:
                    try:
                        data = await response.json()
                        actual_tokens = data.get('usage', {}).get('completion_tokens', 0)
                        prompt_tokens = data.get('usage', {}).get('prompt_tokens', 0)
                    except:
                        actual_tokens = 0
                        prompt_tokens = 0

                    # Calculate throughput
                    tokens_per_sec = actual_tokens / latency if latency > 0 else 0

                    result = {
                        'request_id': request_id,
                        'success': True,
                        'latency': latency,
                        'prompt_tokens': prompt_tokens,
                        'generated_tokens': actual_tokens,
                        'tokens_per_sec': tokens_per_sec,
                        'prompt_preview': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                        'status_code': 200
                    }

                    async with self.stats_lock:
                        self.successful_requests += 1
                else:
                    text = await response.text()
                    result = {
                        'request_id': request_id,
                        'success': False,
                        'latency': latency,
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

        while time.time() - start_time < duration + 30:
            metrics = await self.get_queue_metrics(session)
            if metrics:
                self.queue_metrics.append({
                    'timestamp': time.time(),
                    'waiting': metrics['waiting'],
                    'running': metrics['running']
                })

            await asyncio.sleep(1)

    async def replay_user(self, session, user_id, prompts, base_interval):
        """Single user sending requests"""
        results = []

        for idx, prompt in enumerate(prompts):
            # Random think time between requests
            if idx > 0:
                think_time = random.expovariate(1.0 / base_interval)
                await asyncio.sleep(think_time / self.speed)

            request_id = user_id * len(prompts) + idx
            result = await self.send_request(session, request_id, prompt)
            results.append(result)

            # Print progress
            if (idx + 1) % 10 == 0 or not result['success']:
                status = "✅" if result['success'] else "❌"
                async with self.stats_lock:
                    in_flight = self.in_flight_requests

                latest_queue = self.queue_metrics[-1] if self.queue_metrics else None
                if latest_queue:
                    queue_str = f"Queue: {latest_queue['waiting']:.0f}W/{latest_queue['running']:.0f}R"
                else:
                    queue_str = "Queue: N/A"

                user_prefix = f"U{user_id}" if self.users > 1 else ""
                print(f"[{request_id+1}] {user_prefix} {status} "
                      f"Prompt:{result.get('prompt_tokens', 0):4d}tok "
                      f"Gen:{result.get('generated_tokens', 0):4d}tok "
                      f"Latency:{result['latency']:6.2f}s "
                      f"Tok/s:{result.get('tokens_per_sec', 0):6.1f} "
                      f"InFlight:{in_flight:3d} {queue_str}")

        return results

    async def replay(self):
        """Replay ShareGPT conversations"""
        prompts = self.load_conversations()

        if not prompts:
            print("❌ No prompts loaded")
            return

        print(f"\n🚀 Starting ShareGPT replay with {self.users} concurrent users")
        print(f"   Target: {self.host}")
        print(f"   Total prompts: {len(prompts)}")
        print(f"   Speed: {self.speed}x")
        print("-" * 80)

        # Distribute prompts to users
        prompts_per_user = len(prompts) // self.users
        user_prompts = [prompts[i*prompts_per_user:(i+1)*prompts_per_user] for i in range(self.users)]

        # Give remaining prompts to last user
        if len(prompts) % self.users != 0:
            user_prompts[-1].extend(prompts[self.users * prompts_per_user:])

        # Base interval between requests (seconds)
        base_interval = 5.0  # Average 5 seconds between requests per user

        # Estimate duration
        estimated_duration = (prompts_per_user * base_interval) / self.speed

        # Create session
        connector = aiohttp.TCPConnector(limit=2000)
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Start queue monitoring
            monitor_task = asyncio.create_task(self.monitor_queue(session, estimated_duration))

            # Create user tasks
            user_tasks = []
            for user_id in range(self.users):
                task = asyncio.create_task(
                    self.replay_user(session, user_id, user_prompts[user_id], base_interval)
                )
                user_tasks.append(task)

            # Wait for all users
            all_results = await asyncio.gather(*user_tasks, return_exceptions=True)

            # Cancel monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Flatten results
            self.results = []
            for user_results in all_results:
                if not isinstance(user_results, Exception):
                    self.results.extend(user_results)

            self.total_requests = len(self.results)

        print("-" * 80)
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print(f"\n📊 ShareGPT Replay Summary")
        print("=" * 80)
        print(f"Total Requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests} ({self.successful_requests/self.total_requests*100:.1f}%)")
        print(f"Failed:              {self.failed_requests} ({self.failed_requests/self.total_requests*100:.1f}%)")

        if self.successful_requests > 0:
            successful = [r for r in self.results if r['success']]
            latencies = [r['latency'] for r in successful]
            throughputs = [r['tokens_per_sec'] for r in successful if r.get('tokens_per_sec', 0) > 0]

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
        """Save results to CSV"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'request_id', 'success', 'latency', 'prompt_tokens', 'generated_tokens', 'tokens_per_sec'
            ])
            writer.writeheader()

            for r in self.results:
                writer.writerow({
                    'request_id': r['request_id'],
                    'success': r['success'],
                    'latency': r['latency'],
                    'prompt_tokens': r.get('prompt_tokens', 0),
                    'generated_tokens': r.get('generated_tokens', 0),
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
    parser = argparse.ArgumentParser(description='ShareGPT conversation replay')
    parser.add_argument('--host', default='http://localhost:8000',
                        help='vLLM server URL')
    parser.add_argument('--sharegpt-file',
                        default='/home/skt6g/AI-RAN/KubeSMO/data/sg_90k_part1.json',
                        help='Path to ShareGPT JSON file')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Replay speed multiplier')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Limit number of prompts to use')
    parser.add_argument('--users', type=int, default=1,
                        help='Number of concurrent users')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Max tokens to generate per request')
    parser.add_argument('--output', default='sharegpt_results.csv',
                        help='Output CSV file')

    args = parser.parse_args()

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

    # Run replay
    replayer = ShareGPTReplayer(
        args.host, args.sharegpt_file, args.speed,
        args.limit, args.users, args.max_tokens
    )
    await replayer.replay()
    replayer.save_results(args.output)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
