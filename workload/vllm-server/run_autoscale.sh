#!/bin/bash
# Quick start script for auto-scaling benchmark

set -e

# Configuration
MODEL_PATH="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct"
HOST="0.0.0.0"
BASE_PORT=8001
NUM_PROMPTS=1000
INPUT_LEN=512
OUTPUT_LEN=256
REQUEST_RATE=5
CHECK_INTERVAL=5

# Helm configuration
CHART_PATH="/home/skt6g/AI-RAN/KubeSMO/workload/vllm-server"
NODE_NAME="sys-221he-tnr"
GPU_RESOURCE="nvidia.com/mig-1g.12gb"
HF_TOKEN="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"

echo "============================================================"
echo "vLLM Auto-Scaling Benchmark"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Base Port: $BASE_PORT"
echo "Number of Prompts: $NUM_PROMPTS"
echo "Request Rate: $REQUEST_RATE req/s"
echo "============================================================"
echo ""

# Check if first pod is running
echo "Checking if first pod (port $BASE_PORT) is ready..."
if curl -s "http://$HOST:$BASE_PORT/health" > /dev/null 2>&1; then
    echo "✓ First pod is ready"
else
    echo "✗ First pod is not ready at http://$HOST:$BASE_PORT"
    echo ""
    echo "Please deploy the first pod using:"
    echo "  helm install test-1 $CHART_PATH \\"
    echo "    --set nodeName=$NODE_NAME \\"
    echo "    --set gpuResource=$GPU_RESOURCE \\"
    echo "    --set-string secret.hfApiToken=$HF_TOKEN \\"
    echo "    --set server.modelPath=$MODEL_PATH \\"
    echo "    --set server.port=$BASE_PORT \\"
    echo "    --set server.service.port=$BASE_PORT \\"
    echo "    --set server.service.targetPort=$BASE_PORT"
    echo ""
    exit 1
fi

# Check if metrics endpoint is accessible
echo "Checking metrics endpoint..."
if curl -s "http://$HOST:$BASE_PORT/metrics" > /dev/null 2>&1; then
    echo "✓ Metrics endpoint is accessible"
else
    echo "⚠️  Warning: Metrics endpoint not accessible"
    echo "   Continuing anyway..."
fi

echo ""
echo "Starting auto-scaling benchmark..."
echo ""

# Run the benchmark
python3 autoscale_benchmark.py \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --base-port $BASE_PORT \
  --num-prompts $NUM_PROMPTS \
  --random-input-len $INPUT_LEN \
  --random-output-len $OUTPUT_LEN \
  --request-rate $REQUEST_RATE \
  --check-interval $CHECK_INTERVAL \
  --chart-path "$CHART_PATH" \
  --node-name "$NODE_NAME" \
  --gpu-resource "$GPU_RESOURCE" \
  --hf-token "$HF_TOKEN"

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "============================================================"
