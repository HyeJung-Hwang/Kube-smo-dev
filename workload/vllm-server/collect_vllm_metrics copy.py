#!/usr/bin/env python3
"""
Collect vLLM /metrics during benchmark execution

Usage:
    # Start metric collection in background
    python collect_vllm_metrics.py --url http://localhost:8000 --interval 1 --output metrics.csv &

    # Run benchmark
    vllm bench serve --model /path/to/model --request-rate 6 --num-prompts 100

    # Stop collection (Ctrl+C or kill process)
"""
import argparse
import requests
import time
import csv
import sys
from datetime import datetime
from typing import Dict, List, Optional
import signal


class MetricsCollector:
    def __init__(self, base_url: str, interval: float, output_file: str):
        self.base_url = base_url.rstrip('/')
        self.metrics_url = f"{self.base_url}/metrics"
        self.interval = interval
        self.output_file = output_file
        self.running = True
        self.csv_writer = None
        self.csv_file = None
        self.headers_written = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n📊 Received signal {signum}, stopping collection...")
        self.running = False

    def parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """Parse Prometheus metrics text format"""
        metrics = {}
        current_metric_name = None

        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                # Parse HELP/TYPE lines to get metric names
                if line.startswith('# HELP '):
                    current_metric_name = line.split()[2]
                continue

            # Parse metric line: metric_name{labels} value
            if '{' in line:
                # Has labels
                metric_part, rest = line.split('{', 1)
                labels_part, value_part = rest.rsplit('}', 1)
                value = float(value_part.strip())

                # Create full metric name with labels
                full_name = f"{metric_part}{{{labels_part}}}"
                metrics[full_name] = value
            else:
                # No labels
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    try:
                        value = float(parts[1])
                        metrics[metric_name] = value
                    except ValueError:
                        pass

        return metrics

    def extract_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract key vLLM metrics"""
        key_metrics = {}

        # Find model name from any metric
        model_name = None
        for key in metrics.keys():
            if 'model_name=' in key:
                # Extract model name from label
                start = key.find('model_name="') + len('model_name="')
                end = key.find('"', start)
                model_name = key[start:end]
                break

        if not model_name:
            # Fallback: use any model name found
            model_name = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct"

        # Key metrics to extract
        metric_patterns = {
            'num_requests_running': f'vllm:num_requests_running{{engine="0",model_name="{model_name}"}}',
            'num_requests_waiting': f'vllm:num_requests_waiting{{engine="0",model_name="{model_name}"}}',
            'kv_cache_usage_perc': f'vllm:kv_cache_usage_perc{{engine="0",model_name="{model_name}"}}',
            'num_preemptions_total': f'vllm:num_preemptions_total{{engine="0",model_name="{model_name}"}}',
            'prompt_tokens_total': f'vllm:prompt_tokens_total{{engine="0",model_name="{model_name}"}}',
            'generation_tokens_total': f'vllm:generation_tokens_total{{engine="0",model_name="{model_name}"}}',
            'request_success_stop': f'vllm:request_success_total{{engine="0",finished_reason="stop",model_name="{model_name}"}}',
            'request_success_length': f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="{model_name}"}}',
            'request_success_abort': f'vllm:request_success_total{{engine="0",finished_reason="abort",model_name="{model_name}"}}',
        }

        # Add time-to-first-token (TTFT) histogram metrics
        for bucket in ['0.001', '0.005', '0.01', '0.02', '0.04', '0.08', '0.16', '0.32', '0.64', '1.28', '+Inf']:
            key = f'ttft_bucket_le_{bucket.replace("+", "").replace(".", "_")}'
            pattern = f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="{bucket}",model_name="{model_name}"}}'
            metric_patterns[key] = pattern

        # Add TPOT histogram metrics
        for bucket in ['0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5', '+Inf']:
            key = f'tpot_bucket_le_{bucket.replace("+", "").replace(".", "_")}'
            pattern = f'vllm:time_per_output_token_seconds_bucket{{engine="0",le="{bucket}",model_name="{model_name}"}}'
            metric_patterns[key] = pattern

        # Extract metrics
        for key, pattern in metric_patterns.items():
            key_metrics[key] = metrics.get(pattern, 0.0)

        # Add general process metrics
        key_metrics['process_cpu_seconds'] = metrics.get('process_cpu_seconds_total', 0.0)
        key_metrics['process_resident_memory_bytes'] = metrics.get('process_resident_memory_bytes', 0.0)

        return key_metrics

    def collect_once(self) -> Optional[Dict[str, float]]:
        """Collect metrics once"""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()

            # Parse metrics
            all_metrics = self.parse_prometheus_metrics(response.text)
            key_metrics = self.extract_key_metrics(all_metrics)

            # Add timestamp
            key_metrics['timestamp'] = time.time()
            key_metrics['datetime'] = datetime.now().isoformat()

            return key_metrics

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching metrics: {e}", file=sys.stderr)
            return None

    def run(self):
        """Run metric collection loop"""
        print(f"📊 Starting vLLM metrics collection")
        print(f"   URL: {self.metrics_url}")
        print(f"   Interval: {self.interval}s")
        print(f"   Output: {self.output_file}")
        print("="*80)

        # Open CSV file
        self.csv_file = open(self.output_file, 'w', newline='')

        collection_count = 0
        start_time = time.time()

        try:
            while self.running:
                # Collect metrics
                metrics = self.collect_once()

                if metrics:
                    # Write to CSV
                    if not self.headers_written:
                        # Write headers on first successful collection
                        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=sorted(metrics.keys()))
                        self.csv_writer.writeheader()
                        self.headers_written = True

                    self.csv_writer.writerow(metrics)
                    self.csv_file.flush()  # Ensure data is written

                    collection_count += 1

                    # Print status every 10 collections
                    if collection_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = collection_count / elapsed
                        print(f"✓ Collected {collection_count} samples ({rate:.2f} samples/sec)")
                        print(f"  Running: {metrics['num_requests_running']:.0f}, "
                              f"Waiting: {metrics['num_requests_waiting']:.0f}, "
                              f"KV Cache: {metrics['kv_cache_usage_perc']*100:.1f}%, "
                              f"Completed: {metrics['request_success_stop'] + metrics['request_success_length']:.0f}")

                # Wait for next interval
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n📊 Collection interrupted by user")

        finally:
            # Close file
            if self.csv_file:
                self.csv_file.close()

            elapsed = time.time() - start_time
            print("="*80)
            print(f"📊 Collection Summary:")
            print(f"   Total samples: {collection_count}")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Average rate: {collection_count/elapsed:.2f} samples/sec")
            print(f"   Output saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Collect vLLM metrics during benchmark')
    parser.add_argument('--url', default='http://localhost:8000', help='vLLM server URL')
    parser.add_argument('--interval', type=float, default=1.0, help='Collection interval in seconds')
    parser.add_argument('--output', default='vllm_metrics.csv', help='Output CSV file')

    args = parser.parse_args()

    collector = MetricsCollector(args.url, args.interval, args.output)
    collector.run()


if __name__ == "__main__":
    main()
