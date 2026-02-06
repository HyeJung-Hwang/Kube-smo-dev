#!/usr/bin/env python3
"""
Hybrid Trace Replay: Azure Timing + ShareGPT Content

Uses actual Azure trace timestamps for realistic arrival patterns,
but uses real ShareGPT conversation content instead of synthetic prompts.

Usage:
    # Terminal 1: Port-forward
    kubectl port-forward test-vllm-server 8000:8000

    # Terminal 2: Run hybrid replay
    python trace-sharegpt-hybrid.py --limit 1000 --users 1
"""

import csv
import json
import time
import asyncio
import aiohttp
from datetime import datetime
import argparse
import sys
import random


class HybridTraceReplayer:
    """Uses Azure trace timing with ShareGPT content"""

    def __init__(self, host, azure_trace, sharegpt_file, speed=1.0, limit=None, users=1, max_tokens=100):
        self.host = host
        self.azure_trace = azure_trace
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

        # Lock
        self.stats_lock = asyncio.Lock()

    def load_azure_timestamps(self):
        """Load timestamps from Azure trace"""
        timestamps = []

        with open(self.azure_trace, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if self.limit and idx >= self.limit:
                    break

                # Parse timestamp - handle multiple formats
                timestamp_str = row['TIMESTAMP'].strip()

                # Remove Z suffix
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1]

                # Remove existing timezone offset if present
                if '+' in timestamp_str:
                    timestamp_str = timestamp_str.split('+')[0]
                elif timestamp_str.count('-') > 2:  # Has timezone like -00:00
                    # Keep only date part (YYYY-MM-DD HH:MM:SS)
                    parts = timestamp_str.rsplit('-', 1)
                    if ':' in parts[1]:  # This is timezone offset
                        timestamp_str = parts[0]

                # Handle fractional seconds
                if '.' in timestamp_str:
                    parts = timestamp_str.split('.')
                    fractional = parts[1][:6].ljust(6, '0')  # Keep max 6 digits
                    timestamp_str = f"{parts[0]}.{fractional}+00:00"
                else:
                    timestamp_str = f"{timestamp_str}+00:00"

                timestamps.append(datetime.fromisoformat(timestamp_str))

        print(f"Loaded {len(timestamps)} timestamps from Azure trace")
        return timestamps

    def load_sharegpt_prompts(self):
        """Load prompts from ShareGPT"""
        prompts = []

        print(f"Loading ShareGPT prompts from {self.sharegpt_file}")
        with open(self.sharegpt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            conv = item.get('conversations', [])
            for msg in conv:
                if msg.get('from') == 'human':
                    prompt = msg.get('value', '').strip()
                    if prompt and len(prompt) > 10:
                        prompts.append(prompt)

        print(f"Loaded {len(prompts)} ShareGPT prompts")
        return prompts

    async def get_queue_metrics(self, session):
        """Fetch queue metrics"""
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
        """Send request with streaming to measure TTFT"""
        # Get queue metrics BEFORE sending request
        queue_before = await self.get_queue_metrics(session)

        start_time = time.time()

        async with self.stats_lock:
            self.in_flight_requests += 1

        try:
            payload = {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0.7,
                "stream": True,  # Enable streaming
            }

            async with session.post(
                f"{self.host}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    ttft = None
                    token_count = 0
                    output_text = ""
                    prompt_tokens = 0

                    # Process streaming response
                    async for chunk_bytes in response.content.iter_chunks():
                        chunk_bytes = chunk_bytes[0].strip()
                        if not chunk_bytes:
                            continue

                        # Measure TTFT on first chunk
                        if ttft is None:
                            ttft = time.time() - start_time

                        chunk_str = chunk_bytes.decode("utf-8")

                        # Skip "data: " prefix and handle [DONE]
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[6:]

                        if chunk_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(chunk_str)
                            if isinstance(chunk_data, dict):
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    choice = chunk_data['choices'][0]
                                    if choice and isinstance(choice, dict):
                                        text = choice.get('text', '')
                                        output_text += text
                                        if text:
                                            token_count += 1

                                # Get prompt tokens from first chunk
                                if prompt_tokens == 0 and 'usage' in chunk_data:
                                    usage = chunk_data.get('usage', {})
                                    if isinstance(usage, dict):
                                        prompt_tokens = usage.get('prompt_tokens', 0)
                        except (json.JSONDecodeError, AttributeError, TypeError) as parse_error:
                            # Skip malformed chunks
                            continue

                    latency = time.time() - start_time

                    # Get queue metrics AFTER completing request
                    queue_after = await self.get_queue_metrics(session)

                    # Use token_count as generated tokens
                    actual_tokens = token_count if token_count > 0 else self.max_tokens
                    tokens_per_sec = actual_tokens / latency if latency > 0 else 0

                    # Calculate TTFT in milliseconds
                    ttft_ms = (ttft * 1000) if ttft else 0

                    # Calculate TBT (Time Between Tokens) - excludes first token
                    if actual_tokens > 1 and ttft:
                        tbt_ms = ((latency - ttft) * 1000) / (actual_tokens - 1)
                    else:
                        tbt_ms = 0

                    # Calculate TPOT (includes first token)
                    tpot_ms = (latency * 1000 / actual_tokens) if actual_tokens > 0 else 0

                    result = {
                        'request_id': request_id,
                        'success': True,
                        'latency': latency,
                        'prompt_tokens': prompt_tokens,
                        'generated_tokens': actual_tokens,
                        'tokens_per_sec': tokens_per_sec,
                        'ttft_ms': ttft_ms,
                        'tbt_ms': tbt_ms,
                        'tpot_ms': tpot_ms,
                        'queue_waiting_before': queue_before['waiting'] if queue_before else None,
                        'queue_running_before': queue_before['running'] if queue_before else None,
                        'queue_waiting_after': queue_after['waiting'] if queue_after else None,
                        'queue_running_after': queue_after['running'] if queue_after else None,
                        'batch_size': queue_after['running'] if queue_after else None,
                        'status_code': 200
                    }

                    async with self.stats_lock:
                        self.successful_requests += 1
                else:
                    latency = time.time() - start_time
                    text = await response.text()

                    # Get queue metrics AFTER getting error response
                    try:
                        queue_after = await self.get_queue_metrics(session)
                    except:
                        queue_after = None

                    result = {
                        'request_id': request_id,
                        'success': False,
                        'latency': latency,
                        'tokens_per_sec': 0,
                        'ttft_ms': 0,
                        'tbt_ms': 0,
                        'tpot_ms': 0,
                        'prompt_tokens': 0,
                        'generated_tokens': 0,
                        'queue_waiting_before': queue_before.get('waiting', 0) if queue_before else 0,
                        'queue_running_before': queue_before.get('running', 0) if queue_before else 0,
                        'queue_waiting_after': queue_after.get('waiting', 0) if queue_after else 0,
                        'queue_running_after': queue_after.get('running', 0) if queue_after else 0,
                        'batch_size': queue_after.get('running', 0) if queue_after else 0,
                        'status_code': response.status,
                        'error': f"HTTP {response.status}: {text[:100]}"
                    }

                    async with self.stats_lock:
                        self.failed_requests += 1

        except Exception as e:
            latency = time.time() - start_time
            try:
                queue_after = await self.get_queue_metrics(session)
            except:
                queue_after = None

            # Add more detailed error info
            error_type = type(e).__name__
            error_msg = f"{error_type}: {str(e)}"

            result = {
                'request_id': request_id,
                'success': False,
                'latency': latency,
                'tokens_per_sec': 0,
                'ttft_ms': 0,
                'tbt_ms': 0,
                'tpot_ms': 0,
                'prompt_tokens': 0,
                'generated_tokens': 0,
                'queue_waiting_before': queue_before.get('waiting', 0) if queue_before else 0,
                'queue_running_before': queue_before.get('running', 0) if queue_before else 0,
                'queue_waiting_after': queue_after.get('waiting', 0) if queue_after else 0,
                'queue_running_after': queue_after.get('running', 0) if queue_after else 0,
                'batch_size': queue_after.get('running', 0) if queue_after else 0,
                'error': error_msg[:200]
            }

            async with self.stats_lock:
                self.failed_requests += 1

        async with self.stats_lock:
            self.in_flight_requests -= 1

        return result

    async def monitor_queue(self, session, duration):
        """Monitor queue"""
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

    async def replay_user(self, session, user_id, timestamps, prompts, start_time, base_timestamp):
        """Replay trace for one user"""
        results = []

        for idx, timestamp in enumerate(timestamps):
            # Calculate target time from Azure trace
            trace_offset = (timestamp - base_timestamp).total_seconds()
            target_time = start_time + (trace_offset / self.speed)

            # Wait until target time
            wait_time = target_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Pick random prompt from ShareGPT
            prompt = prompts[idx % len(prompts)]

            # Send request
            request_id = user_id * len(timestamps) + idx
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
                ttft = result.get('ttft_ms', 0)
                tbt = result.get('tbt_ms', 0)
                batch = result.get('batch_size', 0)

                if result['success']:
                    print(f"[{request_id+1}] {user_prefix} {status} "
                          f"Gen:{result.get('generated_tokens', 0):3d}tok "
                          f"Lat:{result['latency']:5.2f}s "
                          f"TTFT:{ttft:5.1f}ms "
                          f"TBT:{tbt:5.1f}ms "
                          f"Batch:{batch:2.0f} "
                          f"InFlight:{in_flight:3d} {queue_str}")
                else:
                    error_msg = result.get('error', 'Unknown error')[:50]
                    print(f"[{request_id+1}] {user_prefix} {status} "
                          f"Lat:{result['latency']:5.2f}s "
                          f"ERROR: {error_msg} "
                          f"InFlight:{in_flight:3d}")

        return results

    async def replay(self):
        """Main replay logic"""
        timestamps = self.load_azure_timestamps()
        prompts = self.load_sharegpt_prompts()

        if not timestamps or not prompts:
            print("❌ Failed to load data")
            return

        print("================== Hybrid Trace Replay ================")
        print(f"   Target: {self.host}")
        print(f"   Timestamps: {len(timestamps)} (from Azure trace)")
        print(f"   Prompts: {len(prompts)} (from ShareGPT)")
        print(f"   Users: {self.users}")
        print(f"   Speed: {self.speed}x")
        print("-" * 80)

        # Calculate duration
        base_timestamp = timestamps[0]
        last_timestamp = timestamps[-1]
        total_duration = (last_timestamp - base_timestamp).total_seconds() / self.speed

        # Create session
        connector = aiohttp.TCPConnector(limit=2000)
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Store actual start time
            self.actual_start_time = time.time()
            start_time = time.time()

            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor_queue(session, total_duration))

            # Create user tasks
            user_tasks = []
            for user_id in range(self.users):
                task = asyncio.create_task(
                    self.replay_user(session, user_id, timestamps, prompts, start_time, base_timestamp)
                )
                user_tasks.append(task)

            # Wait for completion
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

            # Store actual end time
            self.actual_end_time = time.time()

        print("-" * 80)
        self.print_summary()

    def print_summary(self):
        """Print summary"""
        print("=" * 80)

        # Calculate total execution time
        if hasattr(self, 'actual_start_time') and hasattr(self, 'actual_end_time'):
            total_time_sec = self.actual_end_time - self.actual_start_time
            total_time_min = total_time_sec / 60.0
            print(f"⏱️  Total Execution Time: {total_time_sec:.1f} seconds ({total_time_min:.2f} minutes)")
            print()

        print(f"Total Requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests} ({self.successful_requests/self.total_requests*100:.1f}%)")
        print(f"Failed:              {self.failed_requests} ({self.failed_requests/self.total_requests*100:.1f}%)")

        # Calculate total tokens generated
        if self.successful_requests > 0:
            successful = [r for r in self.results if r['success']]
            total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successful)
            total_generated_tokens = sum(r.get('generated_tokens', 0) for r in successful)
            total_tokens = total_prompt_tokens + total_generated_tokens

            print()
            print(f"📊 Token Statistics:")
            print(f"Total Prompt Tokens:     {total_prompt_tokens:,}")
            print(f"Total Generated Tokens:  {total_generated_tokens:,}")
            print(f"Total Tokens:            {total_tokens:,}")

            # Calculate tokens per second
            if hasattr(self, 'actual_start_time') and hasattr(self, 'actual_end_time'):
                tokens_per_sec = total_generated_tokens / total_time_sec if total_time_sec > 0 else 0
                print(f"Overall Throughput:      {tokens_per_sec:.1f} tokens/sec")

        if self.successful_requests > 0:
            successful = [r for r in self.results if r['success']]
            latencies = [r['latency'] for r in successful]
            throughputs = [r['tokens_per_sec'] for r in successful if r.get('tokens_per_sec', 0) > 0]
            ttfts = [r['ttft_ms'] for r in successful if r.get('ttft_ms', 0) > 0]
            tbts = [r['tbt_ms'] for r in successful if r.get('tbt_ms', 0) > 0]
            tpots = [r['tpot_ms'] for r in successful if r.get('tpot_ms', 0) > 0]

            print(f"\nLatency Statistics")
            print(f"Mean:                {sum(latencies)/len(latencies):.2f}s")
            print(f"Min:                 {min(latencies):.2f}s")
            print(f"Max:                 {max(latencies):.2f}s")
            print(f"p50:                 {sorted(latencies)[len(latencies)//2]:.2f}s")
            print(f"p95:                 {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
            print(f"p99:                 {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")

            if ttfts:
                print(f"\nTTFT Statistics (Time To First Token)")
                print(f"Mean:                {sum(ttfts)/len(ttfts):.2f} ms")
                print(f"Min:                 {min(ttfts):.2f} ms")
                print(f"Max:                 {max(ttfts):.2f} ms")
                print(f"p50:                 {sorted(ttfts)[len(ttfts)//2]:.2f} ms")
                print(f"p95:                 {sorted(ttfts)[int(len(ttfts)*0.95)]:.2f} ms")
                print(f"p99:                 {sorted(ttfts)[int(len(ttfts)*0.99)]:.2f} ms")

            if tbts:
                print(f"\nTBT Statistics (Time Between Tokens)")
                print(f"Mean:                {sum(tbts)/len(tbts):.2f} ms/token")
                print(f"Min:                 {min(tbts):.2f} ms/token")
                print(f"Max:                 {max(tbts):.2f} ms/token")
                print(f"p50:                 {sorted(tbts)[len(tbts)//2]:.2f} ms/token")
                print(f"p95:                 {sorted(tbts)[int(len(tbts)*0.95)]:.2f} ms/token")
                print(f"p99:                 {sorted(tbts)[int(len(tbts)*0.99)]:.2f} ms/token")

                # TBT SLA check (100ms)
                tbt_sla_threshold = 100.0
                tbt_violations = [t for t in tbts if t > tbt_sla_threshold]
                tbt_compliance = (len(tbts) - len(tbt_violations)) / len(tbts) * 100
                print(f"\n📋 TBT SLA Compliance (100ms/token)")
                print(f"Compliant:           {len(tbts) - len(tbt_violations)}/{len(tbts)} ({tbt_compliance:.1f}%)")
                print(f"Violations:          {len(tbt_violations)} ({len(tbt_violations)/len(tbts)*100:.1f}%)")

            if tpots:
                print(f"\n⚡ TPOT Statistics (Time Per Output Token)")
                print(f"Mean:                {sum(tpots)/len(tpots):.2f} ms/token")
                print(f"Min:                 {min(tpots):.2f} ms/token")
                print(f"Max:                 {max(tpots):.2f} ms/token")
                print(f"p50:                 {sorted(tpots)[len(tpots)//2]:.2f} ms/token")
                print(f"p95:                 {sorted(tpots)[int(len(tpots)*0.95)]:.2f} ms/token")
                print(f"p99:                 {sorted(tpots)[int(len(tpots)*0.99)]:.2f} ms/token")

                # SLA check
                sla_threshold = 50.0  # 50ms/token
                sla_violations = [t for t in tpots if t > sla_threshold]
                sla_compliance = (len(tpots) - len(sla_violations)) / len(tpots) * 100
                print(f"\n📋 SLA Compliance (50ms/token)")
                print(f"Compliant:           {len(tpots) - len(sla_violations)}/{len(tpots)} ({sla_compliance:.1f}%)")
                print(f"Violations:          {len(sla_violations)} ({len(sla_violations)/len(tpots)*100:.1f}%)")

            if throughputs:
                print(f"\nThroughput Statistics (tokens/s)")
                print(f"Mean:                {sum(throughputs)/len(throughputs):.1f} tok/s")
                print(f"Min:                 {min(throughputs):.1f} tok/s")
                print(f"Max:                 {max(throughputs):.1f} tok/s")
                print(f"p50:                 {sorted(throughputs)[len(throughputs)//2]:.1f} tok/s")
                print(f"p95:                 {sorted(throughputs)[int(len(throughputs)*0.95)]:.1f} tok/s")

        if self.queue_metrics:
            waiting = [q['waiting'] for q in self.queue_metrics]
            running = [q['running'] for q in self.queue_metrics]

            print(f"\nQueue Statistics")
            print(f"Avg Waiting:         {sum(waiting)/len(waiting):.1f}")
            print(f"Max Waiting:         {max(waiting):.0f}")
            print(f"Avg Running:         {sum(running)/len(running):.1f}")
            print(f"Max Running:         {max(running):.0f}")

    def save_results(self, output_file):
        """Save results"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'request_id', 'success', 'latency',
                'prompt_tokens', 'generated_tokens', 'tokens_per_sec',
                'ttft_ms', 'tbt_ms', 'tpot_ms',
                'queue_waiting_before', 'queue_running_before',
                'queue_waiting_after', 'queue_running_after', 'batch_size'
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
                    'ttft_ms': r.get('ttft_ms', 0),
                    'tbt_ms': r.get('tbt_ms', 0),
                    'tpot_ms': r.get('tpot_ms', 0),
                    'queue_waiting_before': r.get('queue_waiting_before', 0),
                    'queue_running_before': r.get('queue_running_before', 0),
                    'queue_waiting_after': r.get('queue_waiting_after', 0),
                    'queue_running_after': r.get('queue_running_after', 0),
                    'batch_size': r.get('batch_size', 0),
                })

        print(f"\n💾 Results saved to: {output_file}")

        queue_file = output_file.replace('.csv', '_queue.csv')
        with open(queue_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'waiting', 'running'])
            writer.writeheader()
            for q in self.queue_metrics:
                writer.writerow(q)

        print(f"💾 Queue metrics saved to: {queue_file}")


async def main_async():
    parser = argparse.ArgumentParser(description='Hybrid trace replay: Azure timing + ShareGPT content')
    parser.add_argument('--host', default='http://localhost:8000')
    parser.add_argument('--azure-trace',
                        default='/home/skt6g/AI-RAN/KubeSMO/data/AzureLMMInferenceTrace_multimodal.csv')
    parser.add_argument('--sharegpt-file',
                        default='/home/skt6g/AI-RAN/KubeSMO/data/sg_90k_part1.json')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--users', type=int, default=1,
                        help='Number of concurrent users (each replays full trace)')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--output', default='hybrid_trace_results.csv')

    args = parser.parse_args()

    # Check server
    print(f"checking vLLM server at {args.host}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.host}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print(f"Server is healthy")
                else:
                    print(f"Server status {response.status}")
    except Exception as e:
        print(f"Cannot reach server: {e}")
        sys.exit(1)

    # Run replay
    replayer = HybridTraceReplayer(
        args.host, args.azure_trace, args.sharegpt_file,
        args.speed, args.limit, args.users, args.max_tokens
    )
    await replayer.replay()
    replayer.save_results(args.output)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
