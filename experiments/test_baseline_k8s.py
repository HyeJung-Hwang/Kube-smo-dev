#!/usr/bin/env python3
"""
Baseline Scheduler - Kubernetes Native Scheduler

쿠버네티스 기본 스케줄러를 사용하는 Baseline 테스트.
- kubectl apply로 간단한 Pod 배포 (busybox 이미지, multi-arch 지원)
- 쿠버네티스 스케줄러가 노드/GPU 배정
- Pod 상태 모니터링으로 메트릭 수집

## 사용법
    python test_baseline_k8s.py
"""
import sys
sys.path.insert(0, '/home/skt6g/AI-RAN/KubeSMO/experiments')

import os
import time
import subprocess
import tempfile
import pandas as pd
import json
import heapq
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ==================== 로깅 설정 ====================
class TeeOutput:
    """stdout을 콘솔과 파일 모두에 출력 (줄바꿈 보존)"""
    def __init__(self, log_file, original_stdout):
        self.log_file = log_file
        self.original_stdout = original_stdout

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        # 줄바꿈 포함하여 그대로 기록 (빈 줄도 유지)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()


def setup_logging(log_path):
    """파일 + 콘솔 로깅 설정 (print 출력도 파일에 기록)"""
    logger = logging.getLogger('baseline_k8s')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    # 로그 파일 열기
    log_file = open(log_path, mode='w', encoding='utf-8')

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # stdout을 Tee로 교체 (print 출력도 파일에 기록)
    sys.stdout = TeeOutput(log_file, sys.__stdout__)

    return logger


# 전역 변수 (KeyboardInterrupt 처리용)
_scheduler = None
_logger = None
_jobs = None

# ==================== 설정 ====================
CSV_PATH = '/home/skt6g/AI-RAN/KubeSMO/data/single_gpu_a100_singleworker_day79_with_mig_capped.csv'

START_HOUR = 9
START_MIN = 40
END_HOUR = 10
END_MIN = 40

SPEED = 1
MAX_TIME = 3600

# 결과 저장 디렉토리
RESULT_DIR = '/home/skt6g/AI-RAN/KubeSMO/experiments'

# 타임스탬프 생성 (실행 시점)
_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 결과 저장 경로 (타임스탬프 포함)
RESULT_PATH = f'{RESULT_DIR}/results_baseline_k8s_{_TIMESTAMP}.json'

# 로그 파일 경로 (타임스탬프 포함)
LOG_PATH = f'{RESULT_DIR}/test_baseline_k8s_{_TIMESTAMP}.log'

# Pod 상태 체크 간격 (초)
POD_CHECK_INTERVAL = 2

# MIG 크기 -> GPU 리소스 이름 매핑 (라운드로빈 분산)
MIG_RESOURCE_MAP = {
    1: ["nvidia.com/mig-1g.12gb"],
    2: ["nvidia.com/mig-2g.24gb"],
    3: ["nvidia.com/mig-3g.48gb", "nvidia.com/mig-3g.47gb"],  # 노드별로 다름
    4: ["nvidia.com/mig-4g.48gb"],
    7: ["nvidia.com/mig-7g.94gb"],
}

# 각 MIG 크기별 라운드로빈 카운터
MIG_ROUND_ROBIN_COUNTER = {size: 0 for size in MIG_RESOURCE_MAP}

# =============================================

# Pod YAML 템플릿 (busybox - multi-arch 지원)
POD_TEMPLATE = """
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  labels:
    app: baseline-workload
    job-id: "{job_id}"
spec:
  restartPolicy: Never
  containers:
  - name: workload
    image: busybox:latest
    command: ["sleep", "infinity"]
    resources:
      limits:
        {gpu_resource}: 1
      requests:
        {gpu_resource}: 1
"""


@dataclass
class JobState:
    """Job 상태 추적"""
    job_id: str
    job_type: str  # 원본 job_type (HP/Spot)
    mig_size: int
    duration_min: float
    submit_time: float  # 시뮬레이션 시간 (초)
    deploy_time: Optional[float] = None  # wall clock (kubectl apply 시점)
    pod_running_time: Optional[float] = None   # wall clock (Pod Running 시점)
    pod_name: Optional[str] = None
    node_name: Optional[str] = None
    gpu_resource: Optional[str] = None
    status: str = "pending"  # pending, submitted, running, completed, failed


@dataclass(order=True)
class Event:
    """이벤트"""
    time: float
    event_type: str = field(compare=False)
    job_id: str = field(compare=False)


class BaselineK8sScheduler:
    """쿠버네티스 기본 스케줄러를 사용하는 Baseline"""

    def __init__(self, jobs: List[dict], speed: float = 1.0):
        self.jobs = jobs
        self.speed = speed
        self.job_states: Dict[str, JobState] = {}
        self.event_queue: List[Event] = []

        self.wall_start = 0.0
        self.start_time = jobs[0]['submit_time'] if jobs else 0.0
        self.current_time = 0.0

        # Job arrival 이벤트 생성
        for job in jobs:
            state = JobState(
                job_id=job['job_id'],
                job_type="Spot",
                mig_size=job['mig_size'],
                duration_min=job['duration_min'],
                submit_time=job['submit_time'],
            )
            self.job_states[job['job_id']] = state

            event = Event(
                time=job['submit_time'],
                event_type='arrival',
                job_id=job['job_id']
            )
            heapq.heappush(self.event_queue, event)

    def run(self, max_time: float):
        """스케줄러 실행"""
        self.wall_start = time.time()

        print("=" * 80)
        print("BASELINE SCHEDULER (Kubernetes Native - Simple Pod)")
        print("=" * 80)
        print()

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if self.current_time > max_time:
                print(f"Max time reached ({max_time}s)")
                break

            # 실제 시간까지 대기
            self._wait_until(event.time)

            if event.event_type == 'arrival':
                self._handle_arrival(event.job_id)

            # 주기적으로 Pod 상태 체크
            self._check_pod_status()

        # 남은 Pod 상태 최종 체크 (모든 Pod가 Running인지 확인)
        print("\nFinal pod status check (querying actual K8s timestamps)...")
        max_checks = 30  # 최대 30번 체크 (60초)
        for i in range(max_checks):
            self._check_pod_status()

            # 모든 submitted pod가 running 상태인지 확인
            pending_pods = [s for s in self.job_states.values()
                          if s.deploy_time is not None and s.pod_running_time is None]
            if not pending_pods:
                print(f"  All pods are running (checked {i+1} times)")
                break

            if i < max_checks - 1:
                time.sleep(POD_CHECK_INTERVAL)

        # 최종 상태 요약
        pending_count = len([s for s in self.job_states.values()
                           if s.deploy_time is not None and s.pod_running_time is None])
        if pending_count > 0:
            print(f"  Warning: {pending_count} pods still pending after final check")

        return self.get_metrics()

    def _wait_until(self, event_time: float):
        """실제 시간까지 대기"""
        wall_elapsed = time.time() - self.wall_start
        sim_elapsed = event_time - self.start_time
        target_wall_time = sim_elapsed / self.speed

        if target_wall_time > wall_elapsed:
            time.sleep(target_wall_time - wall_elapsed)

    def _handle_arrival(self, job_id: str):
        """Job 도착 처리 - kubectl apply"""
        state = self.job_states[job_id]
        time_str = self._format_time(self.current_time)

        print(f"\n [{time_str}] {job_id} arrived ({state.mig_size}g) | {state.job_type}")

        # kubectl apply 시점 기록
        state.deploy_time = time.time()

        success = self._deploy_pod(job_id, state.mig_size)

        if success:
            state.status = "submitted"
            print(f"    Submitted to Kubernetes scheduler")
        else:
            state.status = "failed"
            print(f"    Failed to submit")

    def _deploy_pod(self, job_id: str, mig_size: int) -> bool:
        """kubectl apply로 간단한 Pod 생성"""
        # Pod 이름 생성 (K8s 호환 형식)
        pod_name = f"baseline-{job_id}".lower().replace("_", "-")[:63]
        self.job_states[job_id].pod_name = pod_name

        # MIG 크기에 맞는 GPU 리소스 이름 (라운드로빈으로 선택)
        resource_list = MIG_RESOURCE_MAP.get(mig_size, [f"nvidia.com/mig-{mig_size}g.10gb"])
        counter = MIG_ROUND_ROBIN_COUNTER.get(mig_size, 0)
        gpu_resource = resource_list[counter % len(resource_list)]

        # 카운터 증가
        MIG_ROUND_ROBIN_COUNTER[mig_size] = counter + 1

        self.job_states[job_id].gpu_resource = gpu_resource
        print(f"    GPU resource: {gpu_resource} (round-robin {counter % len(resource_list) + 1}/{len(resource_list)})")

        # Pod YAML 생성
        pod_yaml = POD_TEMPLATE.format(
            pod_name=pod_name,
            job_id=job_id,
            gpu_resource=gpu_resource
        )

        # 임시 파일에 YAML 저장 후 kubectl apply
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(pod_yaml)
                yaml_path = f.name

            result = subprocess.run(
                ["kubectl", "apply", "-f", yaml_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            # 임시 파일 삭제
            os.unlink(yaml_path)

            if result.returncode != 0:
                print(f"    kubectl error: {result.stderr[:200]}")
            return result.returncode == 0

        except Exception as e:
            print(f"    Deploy error: {e}")
            return False

    def _check_pod_status(self):
        """모든 submitted job의 Pod 상태 체크"""
        for job_id, state in self.job_states.items():
            if state.status == "submitted":
                self._check_single_pod(job_id, state)

    def _check_single_pod(self, job_id: str, state: JobState):
        """단일 Pod 상태 체크 (kubectl get pods 사용)"""
        if not state.pod_name:
            return

        try:
            # 간단하게 kubectl get pod로 상태 확인
            cmd = [
                "kubectl", "get", "pod", state.pod_name,
                "-o", "custom-columns=STATUS:.status.phase,NODE:.spec.nodeName",
                "--no-headers"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                # Pod가 아직 없을 수 있음
                return

            if not result.stdout.strip():
                return

            # 파싱: "Running   sys-221he-tnr"
            parts = result.stdout.strip().split()
            phase = parts[0] if len(parts) > 0 else ""
            node_name = parts[1] if len(parts) > 1 else ""

            state.node_name = node_name

            if phase == "Running" and state.pod_running_time is None:
                # Running 감지 시점을 기록 (wall clock)
                state.pod_running_time = time.time()
                state.status = "running"

                k8s_wait = state.pod_running_time - state.deploy_time
                time_str = self._format_time(self.current_time)
                print(f" [{time_str}] {job_id} Running on {node_name} (k8s_wait={k8s_wait:.1f}s)")

            elif phase == "Pending":
                # Pending 상태 - 아직 스케줄링/시작 대기 중
                pending_time = time.time() - state.deploy_time
                # 10초 이상 Pending이면 로그 출력
                if pending_time > 10:
                    time_str = self._format_time(self.current_time)
                    print(f"    {job_id} still Pending ({pending_time:.0f}s)")

            elif phase in ("Succeeded", "Failed"):
                state.status = "completed" if phase == "Succeeded" else "failed"
                time_str = self._format_time(self.current_time)
                print(f" [{time_str}] {job_id} {phase}")

        except Exception as e:
            print(f"    DEBUG: Error checking {job_id}: {e}")

    def _parse_k8s_timestamp(self, timestamp_str: str) -> float:
        """K8s ISO 8601 타임스탬프를 Unix timestamp로 변환

        예: "2024-01-15T10:30:00Z" -> 1705315800.0
        """
        from datetime import datetime, timezone
        try:
            # ISO 8601 형식 파싱 (Z = UTC)
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return time.time()  # fallback

    def _format_time(self, time_seconds: float) -> str:
        """시간 포맷"""
        elapsed = time_seconds - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_metrics(self) -> dict:
        """메트릭 계산 (Running 시점 기준)

        Baseline 특성:
        - 시작 시점: pod_running_time (Pod가 Running 상태가 된 시점)
        - Pending 상태: K8s 스케줄러가 리소스 대기 중 (큐잉 상태)
        - Wait time: submit ~ Running
        """
        total_jobs = len(self.job_states)
        submitted_count = 0
        started_count = 0
        completed_count = 0
        pending_count = 0

        total_wait_time = 0.0
        total_progress = 0.0
        total_jct = 0.0

        current_wall = time.time()

        for job_id, state in self.job_states.items():
            # submit_time을 wall clock으로 변환
            wall_submit_time = self.wall_start + (state.submit_time / self.speed)

            if state.deploy_time is not None:
                submitted_count += 1

            if state.pod_running_time is not None:
                # Running 상태 = 실제 시작됨
                started_count += 1

                # Wait time: submit ~ Running
                wait_time = state.pod_running_time - wall_submit_time

                # 예상 완료 시간: Running + duration
                expected_end_time = state.pod_running_time + (state.duration_min * 60)

                if current_wall >= expected_end_time:
                    # 완료됨 (duration 경과)
                    progress = 100.0
                    completed_count += 1
                    # JCT: submit ~ expected_end
                    jct = expected_end_time - wall_submit_time
                    total_jct += jct
                else:
                    # 실행 중
                    run_time = (current_wall - state.pod_running_time) / 60  # 분
                    progress = min(100.0, (run_time / state.duration_min) * 100) if state.duration_min > 0 else 0

            elif state.deploy_time is not None:
                # Pending 상태 = 큐잉 중 (K8s 스케줄러가 리소스 대기)
                pending_count += 1
                # Wait time: submit ~ current (아직 시작 안 됨)
                wait_time = current_wall - wall_submit_time
                progress = 0.0
            else:
                # 아직 제출 안 됨
                wait_time = current_wall - wall_submit_time
                progress = 0.0

            total_wait_time += wait_time
            total_progress += progress

        return {
            "total_jobs": total_jobs,
            "submitted_jobs": submitted_count,
            "started_jobs": started_count,
            "pending_jobs": pending_count,  # 큐잉 중인 job 수
            "completed_jobs": completed_count,
            "avg_wait_min": (total_wait_time / total_jobs / 60) if total_jobs > 0 else 0,
            "avg_progress": total_progress / total_jobs if total_jobs > 0 else 0,
            "avg_jct_min": (total_jct / completed_count / 60) if completed_count > 0 else 0,
        }

    def get_hp_metrics(self) -> dict:
        """HP job 메트릭 계산 (Running 시점 기준)

        Baseline 특성:
        - 시작 시점: pod_running_time (Pod가 Running 상태가 된 시점)
        - Pending 상태: K8s 스케줄러가 리소스 대기 중 (큐잉 상태)
        - Wait time: submit ~ Running
        """
        hp_count = 0
        hp_started = 0
        hp_completed = 0
        hp_pending = 0
        hp_wait_times = []
        hp_progress_rates = []
        hp_jcts = []

        current_wall = time.time()

        for job_id, state in self.job_states.items():
            if state.job_type != "HP":
                continue

            hp_count += 1

            # submit_time을 wall clock으로 변환
            wall_submit_time = self.wall_start + (state.submit_time / self.speed)

            if state.pod_running_time is not None:
                # Running 상태 = 실제 시작됨
                hp_started += 1

                # Wait time: submit ~ Running
                wait_time = (state.pod_running_time - wall_submit_time) / 60  # 분

                # 예상 완료 시간: Running + duration
                expected_end_time = state.pod_running_time + (state.duration_min * 60)

                if current_wall >= expected_end_time:
                    # 완료됨 (duration 경과)
                    progress = 100.0
                    hp_completed += 1
                    # JCT: submit ~ expected_end
                    jct = (expected_end_time - wall_submit_time) / 60
                    hp_jcts.append(jct)
                else:
                    # 실행 중
                    run_time = (current_wall - state.pod_running_time) / 60
                    progress = min(100.0, (run_time / state.duration_min) * 100) if state.duration_min > 0 else 0

            elif state.deploy_time is not None:
                # Pending 상태 = 큐잉 중
                hp_pending += 1
                wait_time = (current_wall - wall_submit_time) / 60
                progress = 0.0
            else:
                # 아직 제출 안 됨
                wait_time = (current_wall - wall_submit_time) / 60
                progress = 0.0

            hp_wait_times.append(wait_time)
            hp_progress_rates.append(progress)

        return {
            'hp_count': hp_count,
            'hp_started': hp_started,
            'hp_pending': hp_pending,  # 큐잉 중인 HP job 수
            'hp_completed': hp_completed,
            'hp_avg_wait_min': sum(hp_wait_times) / len(hp_wait_times) if hp_wait_times else 0,
            'hp_max_wait_min': max(hp_wait_times) if hp_wait_times else 0,
            'hp_avg_progress': sum(hp_progress_rates) / len(hp_progress_rates) if hp_progress_rates else 0,
            'hp_avg_jct_min': sum(hp_jcts) / len(hp_jcts) if hp_jcts else 0,
        }

    def get_paper_metrics(self) -> dict:
        """
        논문용 5개 메트릭 계산 (HP/Spot/Total 분리)

        메트릭 정의:
        1. Job Completion Time (JCT): 완료된 job의 submit → completion 시간 (분)
        2. Job Queueing Time: submit → 실제 시작 시간 (분)
        3. # of Completed Jobs: 완료된 job 수
        4. Job Progression Rate: (실제 돌아간 시간 / duration) × 100 (%)
        5. Job Processed Time: 실제 클러스터에서 돌아간 시간 (분)

        Returns:
            dict: HP/Spot/Total 별 5개 메트릭 (avg, sum, min, max, std, count, raw_values)
        """
        import numpy as np

        current_wall = time.time()

        # Raw data 수집용 리스트
        hp_data = {
            'jct': [],
            'queueing_time': [],
            'queueing_time_all': [],
            'progression_rate': [],
            'processed_time': [],
            'total_count': 0,
            'completed_count': 0,
            'started_count': 0,
            'running_count': 0,
            'pending_count': 0,
        }

        spot_data = {
            'jct': [],
            'queueing_time': [],
            'queueing_time_all': [],
            'progression_rate': [],
            'processed_time': [],
            'total_count': 0,
            'completed_count': 0,
            'started_count': 0,
            'running_count': 0,
            'pending_count': 0,
        }

        for job_id, state in self.job_states.items():
            is_hp = state.job_type == "HP"
            data = hp_data if is_hp else spot_data
            duration_min = state.duration_min

            data['total_count'] += 1

            # submit_time을 wall clock으로 변환
            wall_submit_time = self.wall_start + (state.submit_time / self.speed)

            # ============================================
            # Job 상태에 따른 메트릭 계산
            # ============================================
            if state.pod_running_time is not None:
                # 시작된 job
                data['started_count'] += 1

                # 2. Queueing Time: submit → Running
                queue_time_min = (state.pod_running_time - wall_submit_time) / 60
                data['queueing_time'].append(queue_time_min)
                data['queueing_time_all'].append(queue_time_min)

                # 예상 완료 시간
                expected_end_time = state.pod_running_time + (duration_min * 60)

                if current_wall >= expected_end_time:
                    # 완료됨
                    data['completed_count'] += 1

                    # 1. JCT: submit → completion
                    jct_min = (expected_end_time - wall_submit_time) / 60
                    data['jct'].append(jct_min)

                    # 4. Progression Rate: 100%
                    progress = 100.0

                    # 5. Processed Time: duration
                    run_time = duration_min
                else:
                    # 실행 중
                    data['running_count'] += 1

                    # 4. Progression Rate
                    run_time = (current_wall - state.pod_running_time) / 60
                    progress = min(100.0, (run_time / duration_min) * 100) if duration_min > 0 else 0.0

                    # 5. Processed Time
                    # run_time은 위에서 계산됨

            elif state.deploy_time is not None:
                # Pending 상태 (K8s 스케줄러가 리소스 대기 중)
                data['pending_count'] += 1

                # 2. Queueing Time (all): submit → current (아직 시작 안 됨)
                queue_time_min = (current_wall - wall_submit_time) / 60
                data['queueing_time_all'].append(queue_time_min)

                # 4. Progression Rate: 0%
                progress = 0.0

                # 5. Processed Time: 0
                run_time = 0.0
            else:
                # 아직 제출 안 됨
                queue_time_min = (current_wall - wall_submit_time) / 60
                data['queueing_time_all'].append(queue_time_min)
                progress = 0.0
                run_time = 0.0

            data['progression_rate'].append(progress)
            data['processed_time'].append(run_time)

        def calc_stats(values: list) -> dict:
            """통계값 계산"""
            if not values:
                return {
                    'avg': 0.0,
                    'sum': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'std': 0.0,
                    'median': 0.0,
                    'count': 0,
                    'raw': []
                }
            arr = np.array(values)
            return {
                'avg': float(np.mean(arr)),
                'sum': float(np.sum(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'count': len(arr),
                'raw': values
            }

        def build_category_metrics(data: dict) -> dict:
            """카테고리별 메트릭 빌드"""
            return {
                'jct': calc_stats(data['jct']),
                'queueing_time': calc_stats(data['queueing_time']),
                'queueing_time_all': calc_stats(data['queueing_time_all']),
                'completed_count': data['completed_count'],
                'progression_rate': calc_stats(data['progression_rate']),
                'processed_time': calc_stats(data['processed_time']),
                'total_count': data['total_count'],
                'started_count': data['started_count'],
                'running_count': data['running_count'],
                'pending_count': data['pending_count'],
            }

        hp_metrics = build_category_metrics(hp_data)
        spot_metrics = build_category_metrics(spot_data)

        # Total 계산
        total_data = {
            'jct': hp_data['jct'] + spot_data['jct'],
            'queueing_time': hp_data['queueing_time'] + spot_data['queueing_time'],
            'queueing_time_all': hp_data['queueing_time_all'] + spot_data['queueing_time_all'],
            'progression_rate': hp_data['progression_rate'] + spot_data['progression_rate'],
            'processed_time': hp_data['processed_time'] + spot_data['processed_time'],
            'total_count': hp_data['total_count'] + spot_data['total_count'],
            'completed_count': hp_data['completed_count'] + spot_data['completed_count'],
            'started_count': hp_data['started_count'] + spot_data['started_count'],
            'running_count': hp_data['running_count'] + spot_data['running_count'],
            'pending_count': hp_data['pending_count'] + spot_data['pending_count'],
        }
        total_metrics = build_category_metrics(total_data)

        return {
            'hp': hp_metrics,
            'spot': spot_metrics,
            'total': total_metrics,
            'metadata': {
                'current_time': self.current_time,
                'wall_elapsed_sec': current_wall - self.wall_start,
            }
        }

    def get_paper_metrics_summary(self) -> dict:
        """논문 테이블용 요약 메트릭 (raw 데이터 제외)"""
        full_metrics = self.get_paper_metrics()

        def summarize(m: dict) -> dict:
            return {
                'jct_avg': m['jct']['avg'],
                'jct_std': m['jct']['std'],
                'jct_min': m['jct']['min'],
                'jct_max': m['jct']['max'],
                'jct_median': m['jct']['median'],
                'queueing_time_avg': m['queueing_time']['avg'],
                'queueing_time_std': m['queueing_time']['std'],
                'queueing_time_min': m['queueing_time']['min'],
                'queueing_time_max': m['queueing_time']['max'],
                'queueing_time_median': m['queueing_time']['median'],
                'completed_count': m['completed_count'],
                'progression_rate_avg': m['progression_rate']['avg'],
                'progression_rate_std': m['progression_rate']['std'],
                'progression_rate_min': m['progression_rate']['min'],
                'progression_rate_max': m['progression_rate']['max'],
                'processed_time_avg': m['processed_time']['avg'],
                'processed_time_sum': m['processed_time']['sum'],
                'processed_time_std': m['processed_time']['std'],
                'total_count': m['total_count'],
                'started_count': m['started_count'],
                'running_count': m['running_count'],
                'pending_count': m['pending_count'],
            }

        return {
            'hp': summarize(full_metrics['hp']),
            'spot': summarize(full_metrics['spot']),
            'total': summarize(full_metrics['total']),
            'metadata': full_metrics['metadata']
        }

    def print_paper_metrics(self):
        """논문용 5개 메트릭 출력"""
        m = self.get_paper_metrics_summary()

        print()
        print("=" * 80)
        print("PAPER METRICS (5 Metrics × HP/Spot/Total)")
        print("=" * 80)
        print()

        print(f"{'Metric':<35} {'HP':>12} {'Spot':>12} {'Total':>12}")
        print("-" * 75)

        # 1. JCT
        print(f"{'1. JCT (min) - Completed Jobs Only':<35}")
        print(f"{'   avg':<35} {m['hp']['jct_avg']:>12.2f} {m['spot']['jct_avg']:>12.2f} {m['total']['jct_avg']:>12.2f}")
        print(f"{'   std':<35} {m['hp']['jct_std']:>12.2f} {m['spot']['jct_std']:>12.2f} {m['total']['jct_std']:>12.2f}")
        print(f"{'   min':<35} {m['hp']['jct_min']:>12.2f} {m['spot']['jct_min']:>12.2f} {m['total']['jct_min']:>12.2f}")
        print(f"{'   max':<35} {m['hp']['jct_max']:>12.2f} {m['spot']['jct_max']:>12.2f} {m['total']['jct_max']:>12.2f}")
        print(f"{'   median':<35} {m['hp']['jct_median']:>12.2f} {m['spot']['jct_median']:>12.2f} {m['total']['jct_median']:>12.2f}")
        print()

        # 2. Queueing Time
        print(f"{'2. Queueing Time (min) - Started Jobs':<35}")
        print(f"{'   avg':<35} {m['hp']['queueing_time_avg']:>12.2f} {m['spot']['queueing_time_avg']:>12.2f} {m['total']['queueing_time_avg']:>12.2f}")
        print(f"{'   std':<35} {m['hp']['queueing_time_std']:>12.2f} {m['spot']['queueing_time_std']:>12.2f} {m['total']['queueing_time_std']:>12.2f}")
        print(f"{'   min':<35} {m['hp']['queueing_time_min']:>12.2f} {m['spot']['queueing_time_min']:>12.2f} {m['total']['queueing_time_min']:>12.2f}")
        print(f"{'   max':<35} {m['hp']['queueing_time_max']:>12.2f} {m['spot']['queueing_time_max']:>12.2f} {m['total']['queueing_time_max']:>12.2f}")
        print(f"{'   median':<35} {m['hp']['queueing_time_median']:>12.2f} {m['spot']['queueing_time_median']:>12.2f} {m['total']['queueing_time_median']:>12.2f}")
        print()

        # 3. Completed Count
        print(f"{'3. # of Completed Jobs':<35} {m['hp']['completed_count']:>12} {m['spot']['completed_count']:>12} {m['total']['completed_count']:>12}")
        print(f"{'   (total submitted)':<35} {m['hp']['total_count']:>12} {m['spot']['total_count']:>12} {m['total']['total_count']:>12}")
        print(f"{'   (started)':<35} {m['hp']['started_count']:>12} {m['spot']['started_count']:>12} {m['total']['started_count']:>12}")
        print(f"{'   (running)':<35} {m['hp']['running_count']:>12} {m['spot']['running_count']:>12} {m['total']['running_count']:>12}")
        print(f"{'   (pending)':<35} {m['hp']['pending_count']:>12} {m['spot']['pending_count']:>12} {m['total']['pending_count']:>12}")
        print()

        # 4. Progression Rate
        print(f"{'4. Progression Rate (%) - All Jobs':<35}")
        print(f"{'   avg':<35} {m['hp']['progression_rate_avg']:>12.2f} {m['spot']['progression_rate_avg']:>12.2f} {m['total']['progression_rate_avg']:>12.2f}")
        print(f"{'   std':<35} {m['hp']['progression_rate_std']:>12.2f} {m['spot']['progression_rate_std']:>12.2f} {m['total']['progression_rate_std']:>12.2f}")
        print(f"{'   min':<35} {m['hp']['progression_rate_min']:>12.2f} {m['spot']['progression_rate_min']:>12.2f} {m['total']['progression_rate_min']:>12.2f}")
        print(f"{'   max':<35} {m['hp']['progression_rate_max']:>12.2f} {m['spot']['progression_rate_max']:>12.2f} {m['total']['progression_rate_max']:>12.2f}")
        print()

        # 5. Processed Time
        print(f"{'5. Processed Time (min) - All Jobs':<35}")
        print(f"{'   avg':<35} {m['hp']['processed_time_avg']:>12.2f} {m['spot']['processed_time_avg']:>12.2f} {m['total']['processed_time_avg']:>12.2f}")
        print(f"{'   sum':<35} {m['hp']['processed_time_sum']:>12.2f} {m['spot']['processed_time_sum']:>12.2f} {m['total']['processed_time_sum']:>12.2f}")
        print(f"{'   std':<35} {m['hp']['processed_time_std']:>12.2f} {m['spot']['processed_time_std']:>12.2f} {m['total']['processed_time_std']:>12.2f}")
        print()

        print("-" * 75)
        wall_elapsed_min = m['metadata']['wall_elapsed_sec'] / 60
        print(f"Wall Time Elapsed: {wall_elapsed_min:.2f} min")
        print()

    def cleanup(self):
        """모든 Pod 정리"""
        print("\nCleaning up Pods...")
        for job_id, state in self.job_states.items():
            if state.pod_name:
                try:
                    subprocess.run(
                        ["kubectl", "delete", "pod", state.pod_name, "--grace-period=0", "--force"],
                        capture_output=True,
                        timeout=30
                    )
                except:
                    pass

        # 라벨로 일괄 삭제 (혹시 남은 것들)
        try:
            subprocess.run(
                ["kubectl", "delete", "pod", "-l", "app=baseline-workload", "--grace-period=0", "--force"],
                capture_output=True,
                timeout=60
            )
        except:
            pass


def load_jobs_from_csv(csv_path, start_hour, start_min, end_hour, end_min,
                       exclude_multi_gpu=True, exclude_a100=True):
    """CSV에서 job 로드"""
    df = pd.read_csv(csv_path)

    if start_hour == end_hour:
        filtered = df[
            (df['hour_of_day'] == start_hour) &
            (df['minute_of_hour'] >= start_min) &
            (df['minute_of_hour'] < end_min)
        ]
    else:
        filtered = df[
            ((df['hour_of_day'] == start_hour) & (df['minute_of_hour'] >= start_min)) |
            ((df['hour_of_day'] > start_hour) & (df['hour_of_day'] < end_hour)) |
            ((df['hour_of_day'] == end_hour) & (df['minute_of_hour'] < end_min))
        ]

    if exclude_multi_gpu and 'gpu_request' in filtered.columns:
        filtered = filtered[filtered['gpu_request'] <= 1]

    if exclude_a100 and 'gpu_model' in filtered.columns:
        filtered = filtered[~filtered['gpu_model'].str.contains('A100', case=False, na=False)]

    if len(filtered) == 0:
        return []

    base_time = start_hour * 3600 + start_min * 60

    jobs = []
    for idx, row in filtered.iterrows():
        submit_time = (row['hour_of_day'] * 3600 +
                      row['minute_of_hour'] * 60 +
                      row['second_of_minute']) - base_time
        submit_time = max(0, submit_time)

        jobs.append({
            'job_id': f"{row['job_type'].lower()}-{row['job_name']}",
            'job_type': row['job_type'],
            'mig_size': int(row['mig_size']),
            'duration_min': row['duration'] / 60.0,
            'submit_time': submit_time,
        })

    return sorted(jobs, key=lambda j: j['submit_time'])


def print_results(metrics, hp_metrics):
    """결과 출력"""
    print()
    print("=" * 80)
    print("BASELINE (K8s Native - Simple Pod) RESULTS")
    print("=" * 80)
    print()

    print("Overall Metrics:")
    print(f"   Total Jobs: {metrics['total_jobs']}")
    print(f"   Submitted Jobs: {metrics.get('submitted_jobs', 0)}")
    print(f"   Pending Jobs: {metrics.get('pending_jobs', 0)} (queued, waiting for resources)")
    print(f"   Started Jobs: {metrics.get('started_jobs', 0)} (Running)")
    print(f"   Completed Jobs: {metrics['completed_jobs']}")
    print(f"   Avg Wait: {metrics['avg_wait_min']:.2f} min (submit -> Running)")
    print(f"   Avg Progress: {metrics.get('avg_progress', 0):.1f}%")
    print()

    print("HP Job Metrics:")
    print(f"   HP Total: {hp_metrics['hp_count']}")
    print(f"   HP Pending: {hp_metrics.get('hp_pending', 0)} (queued)")
    print(f"   HP Started: {hp_metrics['hp_started']} (Running)")
    print(f"   HP Completed: {hp_metrics['hp_completed']}")
    print(f"   HP Avg Wait: {hp_metrics['hp_avg_wait_min']:.2f} min")
    print(f"   HP Max Wait: {hp_metrics['hp_max_wait_min']:.2f} min")
    print(f"   HP Avg Progress: {hp_metrics['hp_avg_progress']:.1f}%")
    print()


def save_results(metrics, hp_metrics, result_path):
    """결과 저장 (Legacy)"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'scheduler_type': 'baseline_k8s',
        'config': {
            'csv_path': CSV_PATH,
            'start_hour': START_HOUR,
            'start_min': START_MIN,
            'end_hour': END_HOUR,
            'end_min': END_MIN,
            'max_time': MAX_TIME,
            'speed': SPEED,
        },
        'overall_metrics': metrics,
        'hp_metrics': hp_metrics,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {result_path}")


def save_paper_results(scheduler, result_path):
    """
    논문용 5개 메트릭 결과를 JSON 파일로 저장

    5개 메트릭 (HP/Spot/Total 분리):
    1. Job Completion Time (JCT)
    2. Job Queueing Time
    3. # of Completed Jobs
    4. Job Progression Rate
    5. Job Processed Time
    """
    paper_metrics = scheduler.get_paper_metrics_summary()

    result = {
        'timestamp': datetime.now().isoformat(),
        'scheduler_type': 'baseline_k8s',
        'config': {
            'csv_path': CSV_PATH,
            'start_hour': START_HOUR,
            'start_min': START_MIN,
            'end_hour': END_HOUR,
            'end_min': END_MIN,
            'max_time': MAX_TIME,
            'speed': SPEED,
        },
        'paper_metrics': paper_metrics,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Paper metrics saved to: {result_path}")


def save_on_interrupt(scheduler, logger):
    """KeyboardInterrupt 시 결과 저장"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("KeyboardInterrupt detected! Saving results...")
    logger.info("=" * 60)

    try:
        metrics = scheduler.get_metrics()
        hp_metrics = scheduler.get_hp_metrics()

        print_results(metrics, hp_metrics)
        scheduler.print_paper_metrics()

        save_results(metrics, hp_metrics, RESULT_PATH)
        save_paper_results(scheduler, RESULT_PATH.replace('.json', '_paper.json'))

        logger.info("Results saved successfully!")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    global _scheduler, _logger, _jobs

    # 로깅 설정
    logger = setup_logging(LOG_PATH)
    _logger = logger

    logger.info("=" * 80)
    logger.info("Baseline Scheduler Test (Kubernetes Native - Simple Pod)")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"CSV: {CSV_PATH}")
    logger.info(f"Time: {START_HOUR:02d}:{START_MIN:02d} ~ {END_HOUR:02d}:{END_MIN:02d}")
    logger.info(f"Speed: {SPEED}x")
    logger.info(f"Max time: {MAX_TIME}s")
    logger.info(f"Log file: {LOG_PATH}")
    logger.info("")

    # CSV에서 job 로드
    logger.info("Loading jobs from CSV...")
    jobs = load_jobs_from_csv(CSV_PATH, START_HOUR, START_MIN, END_HOUR, END_MIN,
                              exclude_multi_gpu=False, exclude_a100=False)
    _jobs = jobs

    if not jobs:
        logger.info("No jobs found!")
        return

    # submit_time 정규화: 첫 번째 job을 0초에 시작
    if jobs:
        first_submit_time = jobs[0]['submit_time']
        for job in jobs:
            job['submit_time'] = job['submit_time'] - first_submit_time
        logger.info(f"  Submit times normalized (offset: -{first_submit_time:.1f}s)")

    hp_jobs = [j for j in jobs if j['job_type'] == "HP"]
    spot_jobs = [j for j in jobs if j['job_type'] == "Spot"]

    logger.info(f"Job Stats:")
    logger.info(f"   Total: {len(jobs)} jobs")
    logger.info(f"   HP: {len(hp_jobs)} jobs")
    logger.info(f"   Spot: {len(spot_jobs)} jobs")
    logger.info("")

    # 첫 10개 job 출력
    logger.info("Jobs (first 10):")
    for job in jobs[:10]:
        logger.info(f"   T={job['submit_time']:>5.0f}s | {job['job_id']:20s} | {job['job_type']:5s} | {job['mig_size']}g")
    if len(jobs) > 10:
        logger.info(f"   ... (total {len(jobs)})")
    logger.info("")

    # Baseline Scheduler 실행
    scheduler = BaselineK8sScheduler(jobs, speed=SPEED)
    _scheduler = scheduler

    try:
        metrics = scheduler.run(max_time=MAX_TIME)
        hp_metrics = scheduler.get_hp_metrics()

        # Legacy 결과 출력
        print_results(metrics, hp_metrics)

        # ========================================
        # 논문용 5개 메트릭 출력 및 저장
        # ========================================
        scheduler.print_paper_metrics()

        # 결과 저장 (Legacy + Paper metrics)
        save_results(metrics, hp_metrics, RESULT_PATH)
        save_paper_results(scheduler, RESULT_PATH.replace('.json', '_paper.json'))

        logger.info("")
        logger.info("=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        if _scheduler is not None:
            save_on_interrupt(_scheduler, logger)
        raise

    finally:
        # Cleanup 옵션
        try:
            cleanup = input("\nCleanup Pods? (y/n): ").strip().lower()
            if cleanup == 'y':
                scheduler.cleanup()
        except (KeyboardInterrupt, EOFError):
            logger.info("\nSkipping cleanup prompt...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if _logger:
            _logger.info("\nExiting...")
        print("\nExiting...")
