#!/bin/bash
# Monitor vLLM metrics in real-time from /metrics endpoint

HOST="${1:-0.0.0.0}"
PORT="${2:-8001}"
INTERVAL="${3:-2}"

METRICS_URL="http://$HOST:$PORT/metrics"

echo "============================================================"
echo "vLLM Metrics Monitor"
echo "============================================================"
echo "Metrics URL: $METRICS_URL"
echo "Refresh interval: ${INTERVAL}s"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Function to extract metric value
get_metric() {
    local metrics_text="$1"
    local metric_name="$2"
    echo "$metrics_text" | grep "^${metric_name}{" | head -1 | awk '{print $2}'
}

while true; do
    clear
    echo "============================================================"
    echo "vLLM Metrics - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # Fetch all metrics
    METRICS=$(curl -s "$METRICS_URL" 2>/dev/null)

    if [ -z "$METRICS" ]; then
        echo "❌ Error: Cannot fetch metrics from $METRICS_URL"
        echo ""
        echo "Please check:"
        echo "  1. Is the vLLM server running?"
        echo "  2. Is the port correct?"
        echo "  3. Try: curl $METRICS_URL"
    else
        # Parse key metrics
        WAITING=$(get_metric "$METRICS" "vllm:num_requests_waiting")
        RUNNING=$(get_metric "$METRICS" "vllm:num_requests_running")
        SWAPPED=$(get_metric "$METRICS" "vllm:num_requests_swapped")
        GPU_CACHE=$(get_metric "$METRICS" "vllm:gpu_cache_usage_perc")
        CPU_CACHE=$(get_metric "$METRICS" "vllm:cpu_cache_usage_perc")

        # If : format doesn't work, try _ format
        if [ -z "$WAITING" ]; then
            WAITING=$(get_metric "$METRICS" "vllm_num_requests_waiting")
            RUNNING=$(get_metric "$METRICS" "vllm_num_requests_running")
            SWAPPED=$(get_metric "$METRICS" "vllm_num_requests_swapped")
            GPU_CACHE=$(get_metric "$METRICS" "vllm_gpu_cache_usage_perc")
            CPU_CACHE=$(get_metric "$METRICS" "vllm_cpu_cache_usage_perc")
        fi

        echo "📊 Request Queue Metrics:"
        echo "   Waiting:  ${WAITING:-0}"
        echo "   Running:  ${RUNNING:-0}"
        echo "   Swapped:  ${SWAPPED:-0}"

        if (( $(echo "${WAITING:-0} >= 1" | bc -l 2>/dev/null || echo 0) )); then
            echo "   ⚠️  WARNING: Queue building up!"
        fi

        echo ""
        echo "💾 Cache Usage:"
        echo "   GPU Cache: ${GPU_CACHE:-N/A}%"
        echo "   CPU Cache: ${CPU_CACHE:-N/A}%"

        echo ""
        echo "📈 Raw metrics available at: $METRICS_URL"
    fi

    echo ""
    echo "============================================================"
    echo "Next refresh in ${INTERVAL}s..."

    sleep $INTERVAL
done
