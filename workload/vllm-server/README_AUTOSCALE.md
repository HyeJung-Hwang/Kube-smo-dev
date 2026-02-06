# vLLM Auto-Scaling Benchmark

vLLM 메트릭을 기반으로 자동으로 Pod를 스케일 아웃하는 벤치마크 도구입니다.

## 개요

이 도구는 다음과 같이 동작합니다:

1. **첫 번째 Pod에서 벤치마크 시작**
2. **vLLM 메트릭 모니터링**: `/metrics` 엔드포인트에서 `vllm:num_requests_waiting` 메트릭 직접 확인
3. **스케일 아웃 조건**: 메트릭이 1 이상인 값이 2번 이상 나오면
4. **두 번째 Pod 자동 배포**: Helm으로 새 Pod 배포
5. **트래픽 분산**: 이후 요청이 두 Pod로 분산됨

## 특징

- **Prometheus 서버 불필요**: vLLM pod의 `/metrics` 엔드포인트에서 직접 메트릭 수집
- **간단한 설정**: Pod 포트포워딩만으로 즉시 사용 가능
- **실시간 모니터링**: 5초마다 메트릭 체크 및 자동 스케일 결정

## 사전 준비

### 1. 첫 번째 Pod 배포

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

### 2. 메트릭 엔드포인트 확인

vLLM의 메트릭 엔드포인트가 정상 동작하는지 확인:

```bash
# 메트릭 확인
curl http://0.0.0.0:8001/metrics | grep num_requests_waiting

# 또는 모니터링 스크립트 사용
./monitor_metrics.sh 0.0.0.0 8001
```

## 사용법

### 기본 실행

```bash
python3 autoscale_benchmark.py \
  --model /root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --base-port 8001 \
  --num-prompts 1000 \
  --random-input-len 512 \
  --random-output-len 256 \
  --request-rate 5
```

### 주요 파라미터

#### 벤치마크 설정
- `--model`: 모델 경로 (필수)
- `--host`: vLLM 서버 호스트 (기본: 0.0.0.0)
- `--base-port`: 첫 번째 서버 포트 (기본: 8001)
- `--num-prompts`: 총 프롬프트 수 (기본: 1000)
- `--random-input-len`: 입력 토큰 길이 (기본: 512)
- `--random-output-len`: 출력 토큰 길이 (기본: 256)
- `--request-rate`: 초당 요청 수 (기본: 5)

#### 모니터링 설정
- `--check-interval`: 메트릭 체크 주기 (초, 기본: 5)

#### Helm 배포 설정
- `--chart-path`: Helm 차트 경로
- `--node-name`: Kubernetes 노드 이름
- `--gpu-resource`: GPU 리소스 타입
- `--hf-token`: HuggingFace API 토큰

#### 추가 옵션
- `--run-distributed`: 스케일 아웃 후 분산 벤치마크 추가 실행

### 고급 사용 예제

```bash
# 높은 request rate로 스케일 아웃 테스트
python3 autoscale_benchmark.py \
  --model /root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --base-port 8001 \
  --num-prompts 2000 \
  --random-input-len 1024 \
  --random-output-len 512 \
  --request-rate 10 \
  --check-interval 3 \
  --run-distributed
```

## 실행 흐름

### 1단계: 단일 Pod 벤치마크
```
[10:30:00] num_requests_waiting: 0.0
[10:30:05] num_requests_waiting: 1.0
  → Threshold exceeded (1/2)
[10:30:10] num_requests_waiting: 2.0
  → Threshold exceeded (2/2)
  ✓ Scale condition met!

🚀 SCALING OUT at request 250/1000
```

### 2단계: 두 번째 Pod 배포
```
============================================================
Deploying new pod: test-2 on port 8002
============================================================
Command: helm install test-2 /home/skt6g/AI-RAN/KubeSMO/workload/vllm-server ...

Waiting for server at 0.0.0.0:8002 to be ready...
✓ Server at 0.0.0.0:8002 is ready!
```

### 3단계: 결과 출력
```
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
```

## 모니터링 메트릭

스크립트가 모니터링하는 vLLM 메트릭:

```bash
# /metrics 엔드포인트에서 가져오는 메트릭
vllm:num_requests_waiting{engine="0",model_name="..."}

# 또는
vllm_num_requests_waiting{engine="0",model_name="..."}
```

### 메트릭 확인 방법
```bash
# 직접 확인
curl http://0.0.0.0:8001/metrics | grep num_requests_waiting

# 실시간 모니터링
./monitor_metrics.sh 0.0.0.0 8001
```

### 스케일 아웃 조건
- 메트릭 값 ≥ 1 이 **2번 연속** 발생 시

## 출력 파일

없음 (콘솔 출력만)

## 문제 해결

### 메트릭 엔드포인트 연결 실패
```bash
# vLLM 서버 health 체크
curl http://0.0.0.0:8001/health

# 메트릭 엔드포인트 확인
curl http://0.0.0.0:8001/metrics

# 특정 메트릭 확인
curl http://0.0.0.0:8001/metrics | grep vllm
```

### Helm 배포 실패
```bash
# Helm 설치 확인
helm list

# 이전 릴리스 삭제
helm uninstall test-2

# Kubernetes 권한 확인
kubectl auth can-i create pods
```

### Pod가 Ready 상태가 안 됨
```bash
# Pod 상태 확인
kubectl get pods -l app=vllm-server

# Pod 로그 확인
kubectl logs test-2-vllm-server-xxx

# Health 엔드포인트 직접 확인
curl http://0.0.0.0:8002/health
```

## 코드 구조

```
autoscale_benchmark.py
├── MetricsMonitor         # vLLM /metrics 엔드포인트 모니터링
│   ├── parse_prometheus_metrics()  # Prometheus 형식 파싱
│   ├── query_num_requests_waiting()  # 메트릭 쿼리
│   └── check_should_scale()        # 스케일 조건 체크
├── HelmDeployer           # Helm을 통한 Pod 배포
├── run_benchmark_with_monitoring  # 모니터링하며 벤치마크 실행
├── run_distributed_benchmark      # 분산 벤치마크 실행
└── main                   # 메인 오케스트레이션
```

## 주요 기능

### 1. 실시간 메트릭 모니터링
- vLLM `/metrics` 엔드포인트에서 직접 메트릭 수집
- Prometheus 형식 파싱
- `num_requests_waiting` 값 추적
- 임계값 초과 횟수 카운팅

### 2. 동적 스케일 아웃
- 조건 충족 시 자동으로 Helm 배포
- 새 Pod의 Ready 상태 대기
- 배포 실패 시 에러 핸들링

### 3. 트래픽 분산
- 스케일 아웃 후 남은 요청은 단일 Pod로 계속 전송
- `--run-distributed` 옵션 사용 시 추가로 분산 벤치마크 실행

## 제한사항

1. **현재는 최대 2개 Pod까지 스케일 아웃**
   - 필요시 코드 수정으로 N개까지 확장 가능

2. **스케일 인(Scale In)은 미지원**
   - Pod는 자동으로 제거되지 않음

3. **단순 Round-Robin 분산**
   - 고급 로드 밸런싱 알고리즘 없음

## 향후 개선 사항

- [ ] N개 Pod로 확장 가능하도록 수정
- [ ] 스케일 인 기능 추가
- [ ] 다양한 메트릭 기반 스케일링 (latency, throughput 등)
- [ ] 결과를 JSON 파일로 저장
- [ ] 실시간 대시보드 통합
- [ ] Weighted round-robin 로드 밸런싱
