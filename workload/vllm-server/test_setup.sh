#!/bin/bash
# Test setup script - verify everything is ready

set -e

MODEL_PATH="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct"
HOST="0.0.0.0"
BASE_PORT=8001

echo "============================================================"
echo "Setup Verification"
echo "============================================================"
echo ""

# 1. Check Python dependencies
echo "1. Checking Python dependencies..."
if python3 -c "import aiohttp, requests, tqdm, vllm" 2>/dev/null; then
    echo "   ✓ All Python packages installed"
else
    echo "   ✗ Missing Python packages"
    echo ""
    echo "   Install with:"
    echo "   pip install aiohttp requests tqdm vllm"
    exit 1
fi

# 2. Check if script exists
echo "2. Checking autoscale_benchmark.py..."
if [ -f "autoscale_benchmark.py" ]; then
    echo "   ✓ Script found"
else
    echo "   ✗ autoscale_benchmark.py not found"
    exit 1
fi

# 3. Check first pod
echo "3. Checking first vLLM pod (port $BASE_PORT)..."
if curl -s "http://$HOST:$BASE_PORT/health" > /dev/null 2>&1; then
    echo "   ✓ Pod is running and healthy"

    # Try to get version
    VERSION=$(curl -s "http://$HOST:$BASE_PORT/version" 2>/dev/null || echo "unknown")
    echo "   Version: $VERSION"
else
    echo "   ✗ Pod is not accessible at http://$HOST:$BASE_PORT"
    echo ""
    echo "   Deploy with:"
    echo "   helm install test-1 /home/skt6g/AI-RAN/KubeSMO/workload/vllm-server \\"
    echo "     --set nodeName=sys-221he-tnr \\"
    echo "     --set gpuResource=nvidia.com/mig-1g.12gb \\"
    echo "     --set-string secret.hfApiToken=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 \\"
    echo "     --set server.modelPath=$MODEL_PATH \\"
    echo "     --set server.port=$BASE_PORT \\"
    echo "     --set server.service.port=$BASE_PORT \\"
    echo "     --set server.service.targetPort=$BASE_PORT"
    exit 1
fi

# 4. Check metrics endpoint
echo "4. Checking vLLM metrics endpoint..."
METRICS_URL="http://$HOST:$BASE_PORT/metrics"
if curl -s "$METRICS_URL" > /dev/null 2>&1; then
    echo "   ✓ Metrics endpoint is accessible"

    # Check if we can find vLLM metrics
    METRICS=$(curl -s "$METRICS_URL")
    if echo "$METRICS" | grep -q "vllm.*num_requests_waiting"; then
        echo "   ✓ vLLM metrics are available"
    else
        echo "   ⚠️  Warning: vLLM metrics not found"
        echo "   The metrics endpoint exists but vLLM metrics may not be enabled"
    fi
else
    echo "   ✗ Cannot access metrics at $METRICS_URL"
    echo ""
    echo "   Metrics endpoint is required for auto-scaling"
    exit 1
fi

# 5. Check Helm
echo "5. Checking Helm..."
if command -v helm &> /dev/null; then
    HELM_VERSION=$(helm version --short 2>/dev/null || echo "unknown")
    echo "   ✓ Helm installed: $HELM_VERSION"

    # Check if chart exists
    if [ -d "/home/skt6g/AI-RAN/KubeSMO/workload/vllm-server" ]; then
        echo "   ✓ Helm chart found"
    else
        echo "   ✗ Helm chart not found at /home/skt6g/AI-RAN/KubeSMO/workload/vllm-server"
        exit 1
    fi
else
    echo "   ✗ Helm not installed"
    exit 1
fi

# 6. Check kubectl
echo "6. Checking kubectl..."
if command -v kubectl &> /dev/null; then
    echo "   ✓ kubectl installed"

    # Check if we can access the cluster
    if kubectl cluster-info &> /dev/null; then
        echo "   ✓ Can access Kubernetes cluster"
    else
        echo "   ✗ Cannot access Kubernetes cluster"
        exit 1
    fi
else
    echo "   ✗ kubectl not installed"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ All checks passed!"
echo "============================================================"
echo ""
echo "You can now run the benchmark with:"
echo ""
echo "  ./run_autoscale.sh"
echo ""
echo "Or manually with:"
echo ""
echo "  python3 autoscale_benchmark.py \\"
echo "    --model $MODEL_PATH \\"
echo "    --host $HOST \\"
echo "    --base-port $BASE_PORT \\"
echo "    --num-prompts 1000 \\"
echo "    --request-rate 5"
echo ""
echo "To monitor metrics in real-time:"
echo ""
echo "  ./monitor_metrics.sh $HOST $BASE_PORT"
echo ""
