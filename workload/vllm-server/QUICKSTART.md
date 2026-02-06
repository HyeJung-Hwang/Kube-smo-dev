# Quick Start Guide

## 1분 안에 시작하기

### Step 1: 환경 확인
```bash
cd /home/skt6g/AI-RAN/KubeSMO/workload/vllm-server
./test_setup.sh
```

### Step 2: 첫 번째 Pod 배포 (아직 안 했다면)
```bash
helm install test-1 /home/skt6g/AI-RAN/KubeSMO/workload/vllm-server \
  --set nodeName=sys-221he-tnr \
  --set gpuResource=nvidia.com/mig-1g.12gb \
  --set-string secret.hfApiToken=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 \
  --set server.modelPath=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct \
  --set server.port=8001 \
  --set server.service.port=8001 \
  --set server.service.targetPort=8001
```

### Step 3: 벤치마크 실행
```bash
./run_autoscale.sh
```

끝! 🎉

---

## 상세 설명

### 메트릭 모니터링 (선택사항)

벤치마크 실행 중 다른 터미널에서:

```bash
# 기본 (localhost:8001)
./monitor_metrics.sh

# 또는 커스텀 호스트/포트
./monitor_metrics.sh 0.0.0.0 8001
```

출력 예시:
```
============================================================
vLLM Metrics - 2025-01-15 10:30:00
============================================================

📊 Request Queue Metrics:
   Waiting:  2
   Running:  4
   Swapped:  0
   ⚠️  WARNING: Queue building up!

💾 Cache Usage:
   GPU Cache: 75.5%
   CPU Cache: 12.3%
```

### 수동 실행 (더 많은 옵션)

```bash
python3 autoscale_benchmark.py \
  --model /root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --base-port 8001 \
  --num-prompts 1000 \
  --random-input-len 512 \
  --random-output-len 256 \
  --request-rate 5 \
  --check-interval 5
```

### 파라미터 조정

더 빠르게 스케일 아웃시키려면:

```bash
# 높은 request rate
--request-rate 10

# 더 자주 체크
--check-interval 3

# 더 많은 프롬프트
--num-prompts 2000
```

---

## 실행 흐름

1. **벤치마크 시작** → 첫 번째 pod (port 8001)로 요청 전송
2. **메트릭 모니터링** → 5초마다 `/metrics` 체크
3. **스케일 조건 감지** → `num_requests_waiting >= 1` 이 2번 발생
4. **자동 배포** → 두 번째 pod (port 8002) Helm 배포
5. **계속 실행** → 나머지 요청 처리
6. **결과 출력** → 통계 표시

---

## 예상 출력

```
============================================================
vLLM Auto-Scaling Benchmark
============================================================
Model: /root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct
Base Port: 8001
Number of Prompts: 1000
Request Rate: 5 req/s
============================================================

✓ First pod is ready
✓ Metrics endpoint is accessible

Starting auto-scaling benchmark...

[10:30:00] num_requests_waiting: 0.0
[10:30:05] num_requests_waiting: 1.0
  → Threshold exceeded (1/2)
[10:30:10] num_requests_waiting: 2.0
  → Threshold exceeded (2/2)
  ✓ Scale condition met!

🚀 SCALING OUT at request 250/1000

============================================================
Deploying new pod: test-2 on port 8002
============================================================

Waiting for server at 0.0.0.0:8002 to be ready...
✓ Server at 0.0.0.0:8002 is ready!

============================================================
Single Pod Benchmark Results
============================================================
Duration:             120.45s
Completed requests:   1000
Failed requests:      0
Request throughput:   8.30 req/s
Token throughput:     2123.45 tok/s
============================================================

🎯 Scaled out! Now running on 2 pods

============================================================
Benchmark Complete!
============================================================
```

---

## 문제 해결

### Pod가 Ready 안 됨
```bash
kubectl get pods
kubectl logs test-1-vllm-server-xxx
```

### 메트릭이 안 나옴
```bash
curl http://0.0.0.0:8001/metrics
```

### Helm 배포 실패
```bash
helm list
helm uninstall test-2  # 재시도
```

---

## 정리

벤치마크 완료 후:

```bash
# 모든 릴리스 삭제
helm uninstall test-1
helm uninstall test-2  # 있다면
```

---

더 자세한 내용은 `README_AUTOSCALE.md`를 참조하세요.
