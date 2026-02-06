#!/usr/bin/env python3
"""
Example Locust tasks for vLLM Server testing

Usage:
    locust -f example-locust-tasks.py \
      --host http://my-vllm-server.default.svc.cluster.local:8000 \
      --users 100 \
      --spawn-rate 10
"""

from locust import FastHttpUser, task, between
import random

# Sample prompts for testing
PROMPTS = [
    "Tell me a short story about AI",
    "Explain quantum computing in simple terms",
    "Write a haiku about programming",
    "What are the benefits of machine learning?",
    "Describe a day in the life of a software engineer",
]


class VLLMUser(FastHttpUser):
    """User class for testing vLLM server"""

    # Wait time between requests (simulates real user behavior)
    wait_time = between(0.1, 1.0)

    @task(weight=3)
    def generate_short(self):
        """Generate short responses (256 tokens)"""
        prompt = random.choice(PROMPTS)

        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            catch_response=True,
            name="generate_short"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(weight=1)
    def generate_long(self):
        """Generate long responses (1024 tokens)"""
        prompt = f"Write a detailed essay about: {random.choice(PROMPTS)}"

        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.8,
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            catch_response=True,
            name="generate_long"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(weight=5)
    def chat_completion(self):
        """Test chat completions endpoint"""
        payload = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [
                {"role": "user", "content": random.choice(PROMPTS)}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
            name="chat_completion"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    def on_start(self):
        """Called when user starts - warm up"""
        print(f"User starting on {self.host}")

    def on_stop(self):
        """Called when user stops"""
        print(f"User stopping")


# For running without Web UI
if __name__ == "__main__":
    import os
    os.system("locust -f example-locust-tasks.py")
