import sys
sys.path.insert(0, '/home/skt6g/AI-RAN/KubeSMO/experiments')

from test_dynamic_endpoint import EndpointScheduler, JobRequest, JobResponse, load_ran_model, setup_logging

import time
import threading
import heapq
import logging
import json
import signal
import os
import subprocess
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
import realtime_deploy as deploy

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import cloudpickle

from job import Job, RANJob, Event, GPUNode
from realtime_slot_scheduler import JobState, RealTimeSlotScheduler


# 전역 변수 
logger = logging.getLogger('dynamic_endpoint')
HOST = "0.0.0.0"
PORT = 8001
SPEED = 1  # 시뮬레이션 속도
DRY_RUN = True # True: 실제 배포 안함, False: 실제 배포


# RAN Prediction Model
MODEL_PATH = "/home/skt6g/AI-RAN/KubeSMO/backend/control_plane/models/enhanced_lgb_model.pkl"
CELL_DATA_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/cell_5678_data.csv"
AI_INFERENCE_DATA_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/Azure_0518_inference.csv"
_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_DIR = '/home/skt6g/AI-RAN/KubeSMO/experiments'
LOG_PATH = f'{RESULT_DIR}/endpoint_{_TIMESTAMP}.log'

# Global model variable
_MODEL = None
_log_file = None
scheduler: Optional[EndpointScheduler] = None
scheduler_thread: Optional[threading.Thread] = None
scheduler_lock = threading.Lock()
stop_event = threading.Event()
RAN_IP_PORT_MAP = {
    "aerial-l1": 50052,
    "aerial-l1-1": 50053,
}
RAN_METRICS_PORT_MAP = {
    "aerial-l1": 8085,
    "aerial-l1-1": 8086,
}


def create_initial_nodes():
    return [
        GPUNode(name="skt-6gtb-ars", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=1),
    ]

class BinPackScheduler(EndpointScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replica_counter = {}
        self.hp_replicas = {}
        self._consolidation_in_progress = False

    def _compute_size_index(self, node, target_slice):
        return sum(1 for s in node.slices[:target_slice['id']] if s['size'] == target_slice['size'])

    def _exec_in_pod(self, pod_name, container, command):
        cmd = ['kubectl', 'exec', '-i', pod_name, '-c', container, '--', '/bin/bash', '-c', command]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}

    def _get_pod_ip(self, pod_name):
        cmd = ['kubectl', 'get', 'pod', pod_name, '-o', 'jsonpath={.status.podIP}']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
        return "127.0.0.1"

    def _wait_pod_running(self, pod_name, timeout=300):
        cmd = ['kubectl', 'wait', f'--for=jsonpath={{.status.phase}}=Running', f'pod/{pod_name}', f'--timeout={timeout}s']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
            return result.returncode == 0
        except:
            return False

    def _wait_app_ready(self, pod_name, container="aerial-testmac-ctr", timeout=300):
        for attempt in range(timeout):
            cmd = ['kubectl', 'logs', f'pod/{pod_name}', '-c', container, '--tail', '50']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if "Thread oam_thread_func on CPU" in result.stdout and "initialized fmtlog" in result.stdout:
                    print(f"[RAN-Migration] {pod_name} app ready (attempt {attempt + 1})")
                    return True
            except:
                pass
            time.sleep(1)
        print(f"[RAN-Migration] {pod_name} app not ready after {timeout}s")
        return False

    def _run_cell_cmd(self, pod_name, container, ip, port, num_cells, cmd_id):
        command = f'''
    cd /opt/nvidia/cuBB/cuPHY-CP/cuphyoam/examples
    for i in $(seq 0 {num_cells - 1}); do
        python3 aerial_cell_ctrl_cmd.py --server_ip {ip} --cell_id $i --port {port} --cmd {cmd_id} || exit 1
    done
    '''
        print(f"[RAN-Migration] cell cmd {cmd_id} on {pod_name} ({num_cells} cells, ip={ip}, port={port})")
        print(command)
        result = self._exec_in_pod(pod_name, container, command)
        print(f"[RAN-Migration] cell cmd {cmd_id} result: success={result['success']}")
        return result

    def handle_arrival(self, job):
        # 메트릭 측정등을 위함 — 최초 도착 시에만 JobState 생성 (re-queue 시 submit_time 보존)
        if job.job_id not in self.job_states:
            self.job_states[job.job_id] = JobState(
                job_id=job.job_id,
                original_duration=job.duration,
                remaining_duration=job.duration,
                job_type=job.job_type,
                workload_type=getattr(job, 'workload_type', 'AI'),
                submit_time=self.current_time
            )
        # 배포할 노드 , GPU, 인스턴스 결정 [1~4]
        N = job.req
        # 1. 배포 대상이 되는 노드 리스트 결정 (Job Type에 따라 다름)
        if job.job_type == "migration" and job.workload_type == "AI":
            deploy_target_nodes = [n for n in self.nodes if n is not job.source_node]
        # TODO: SYS에서 ARS 돌리는게 가능해지면 제외해도되는 조건문
        elif job.workload_type == "RAN":
            deploy_target_nodes = [n for n in self.nodes if n.name == "skt-6gtb-ars"]
        else:
            deploy_target_nodes = self.nodes

        # 2. 노드&GPU 조합에서 빈공간 찾기
        # candidate_list: [(node, [{'id':0,'size':3,'used':0,'jobs':[]}, ...])]
        if job.workload_type == "RAN" and   job.job_type == "migration":
            # migration의 경우 source node 제외
            # deploy_target_nodes = [n for n in deploy_target_nodes if n != job.source_node]
            candidate_list = []
            for node in deploy_target_nodes:    
                free_slices = node.get_all_slice_list()
                if free_slices:
                    candidate_list.append((node, free_slices))
            print(f"\n[BinPack-Migration] Job {job.job_id} (req={N}g, type={job.job_type})")
            print(f"[BinPack-Migration] Candidates: {len(candidate_list)} nodes")
            for node, slices in candidate_list:
                sizes = [s['size'] for s in slices]
                print(f"  {node.name} GPU{node.gpu_index}: all_slices={sizes} (total={sum(sizes)}g)")
        else:
            candidate_list = []
            for node in deploy_target_nodes:
                free_slices = node.get_free_slice_list()
                if free_slices:
                    candidate_list.append((node, free_slices))
            print(f"\n[BinPack] Job {job.job_id} (req={N}g, type={job.job_type})")
            print(f"[BinPack] Candidates: {len(candidate_list)} nodes")
            for node, slices in candidate_list:
                sizes = [s['size'] for s in slices]
                print(f"  {node.name} GPU{node.gpu_index}: free_slices={sizes} (total_free={sum(sizes)}g)")

        # 3. Placement strategy에 따라 노드/슬라이스 선택
        best_node = None
        best_slice = None
        strategy = getattr(self, 'placement_strategy', 'best_fit')

        # 배치 가능한 candidate 필터링
        candidate_with_fit = [
            (node, slices) for node, slices in candidate_list
            if any(s['size'] >= N for s in slices)
        ]

        if candidate_with_fit:
            if strategy == 'best_fit':
                # Best-Fit: exact fit 우선, 없으면 min waste
                candidate_with_exact_fit = [
                    (node, slices) for node, slices in candidate_with_fit
                    if any(s['size'] == N for s in slices)
                ]
                if candidate_with_exact_fit:
                    print(f"[BinPack] Exact fit ({N}g) found in {len(candidate_with_exact_fit)} nodes:")
                    for node, slices in candidate_with_exact_fit:
                        total_free = sum(s['size'] for s in slices)
                        print(f"  {node.name} GPU{node.gpu_index}: free={total_free}g, waste={total_free - N}g")
                    best_node, _ = min(
                        candidate_with_exact_fit,
                        key=lambda x: sum(s['size'] for s in x[1]) - N
                    )
                    is_migration = (job.job_type == "migration")
                    slice_list = best_node.get_all_slice_list() if is_migration else best_node.get_free_slice_list()
                    best_slice = next(s for s in slice_list if s['size'] == N)
                    print(f"[BinPack] Selected: {best_node.name} GPU{best_node.gpu_index} "
                          f"slice[{best_slice['id']}] ({best_slice['size']}g) -- exact fit")
                else:
                    print(f"[BinPack] No exact fit. Larger fit (>={N}g) in {len(candidate_with_fit)} nodes:")
                    for node, slices in candidate_with_fit:
                        fits = [s['size'] for s in slices if s['size'] >= N]
                        min_waste = min(s - N for s in fits)
                        print(f"  {node.name} GPU{node.gpu_index}: fits={fits}, min_waste={min_waste}g")
                    best_node, _ = min(
                        candidate_with_fit,
                        key=lambda x: min(s['size'] - N for s in x[1] if s['size'] >= N)
                    )
                    is_migration = (job.job_type == "migration")
                    slice_list = best_node.get_all_slice_list() if is_migration else best_node.get_free_slice_list()
                    best_slice = min(
                        (s for s in slice_list if s['size'] >= N),
                        key=lambda s: s['size'] - N
                    )
                    print(f"[BinPack] Selected: {best_node.name} GPU{best_node.gpu_index} "
                          f"slice[{best_slice['id']}] ({best_slice['size']}g) -- waste={best_slice['size'] - N}g")

            elif strategy == 'least_allocated':
                # Least Allocated (K8s default): exact fit만 허용, 빈 공간이 가장 많은 노드
                # K8s는 mig-1g.12gb 요청하면 1g 슬롯에만 배치됨 (3g에 배치 안됨)
                candidate_with_exact = [
                    (node, slices) for node, slices in candidate_with_fit
                    if any(s['size'] == N for s in slices)
                ]
                if candidate_with_exact:
                    best_node, _ = max(
                        candidate_with_exact,
                        key=lambda x: sum(s['size'] for s in x[1])
                    )
                    is_migration = (job.job_type == "migration")
                    slice_list = best_node.get_all_slice_list() if is_migration else best_node.get_free_slice_list()
                    best_slice = next(s for s in slice_list if s['size'] == N)
                    total_free = sum(s['size'] for s in best_node.get_free_slice_list())
                    print(f"[LeastAlloc] Selected: {best_node.name} GPU{best_node.gpu_index} "
                          f"slice[{best_slice['id']}] ({best_slice['size']}g) -- node_free={total_free}g")
                else:
                    # exact fit 없음 → 배치 불가 (K8s처럼 Pending)
                    best_node = None
                    best_slice = None
                    print(f"[LeastAlloc] No exact {N}g slot available")

            elif strategy == 'most_allocated':
                # Most Allocated: exact fit만 허용, 빈 공간이 가장 적은 노드
                candidate_with_exact = [
                    (node, slices) for node, slices in candidate_with_fit
                    if any(s['size'] == N for s in slices)
                ]
                if candidate_with_exact:
                    best_node, _ = min(
                        candidate_with_exact,
                        key=lambda x: sum(s['size'] for s in x[1])
                    )
                    is_migration = (job.job_type == "migration")
                    slice_list = best_node.get_all_slice_list() if is_migration else best_node.get_free_slice_list()
                    best_slice = next(s for s in slice_list if s['size'] == N)
                    total_free = sum(s['size'] for s in best_node.get_free_slice_list())
                    print(f"[MostAlloc] Selected: {best_node.name} GPU{best_node.gpu_index} "
                          f"slice[{best_slice['id']}] ({best_slice['size']}g) -- node_free={total_free}g")
                else:
                    # exact fit 없음 → 배치 불가
                    best_node = None
                    best_slice = None
                    print(f"[MostAlloc] No exact {N}g slot available")
        else:
            print(f"[BinPack] No node can fit {N}g")

        # 4. best_node에 job 배포/ best node 없는 경우 다시 큐잉
        if best_node is None:
            # HP-AI deploy인데 빈 슬라이스가 없는 경우 → ILP로 재계획
            if job.job_type in ("HP", "HP-deploy") and getattr(job, 'workload_type', 'AI') == "AI":
                target_node = self._select_node_for_hp_preempt(job) #  ILP가 트리거되는 시점 = HP가 도착했는데 자리가 없을 때 
                if target_node:
                    print(f"[BinPack] HP-AI deploy {job.job_id} → ILP preemption on {target_node.name}")
                    self._hp_preempt_with_ilp(job, target_node)
                    return

            # 다음 completion 직후에 재시도 (+1초로 completion이 먼저 처리되도록 보장)
            next_completion = min(
                (e.time for e in self.event_queue
                 if e.event_type == 'completion' and not e.cancelled),
                default=self.current_time + 60
            )
            retry_time = next_completion + 1
            arrival_event = Event(
                time=retry_time,
                event_type='arrival',
                job=job
            )
            heapq.heappush(self.event_queue, arrival_event)
        else: # 있을 떄 (정상)
            self.deploy_job_to_node(job, best_node, best_slice)

    def deploy_job_to_node(self, job, node, target_slice):
        logger.info(
            f"Deploying job {job.job_id} ({job.req}g) to node {node.name} "
            f"GPU {node.gpu_index} slice {target_slice['id']} ({target_slice['size']}g)"
        )

        if job.job_type == "deploy":
            # 1. 상태 처리 GPU Node & Job Status 업데이트
            target_slice['used'] = job.req
            target_slice['jobs'].append(job.job_id)
            job.allocated_size = target_slice['size']

            actual_time = (time.time() - self.wall_start) * self.speed + self.start_time
            job.start_time = actual_time

            if job.job_id in self.job_states:
                remaining_duration = self.job_states[job.job_id].remaining_duration
                self.job_states[job.job_id].actual_start_time = actual_time
            else:
                remaining_duration = job.duration   
            job.end_time = actual_time + (remaining_duration * 60)

            self.running_jobs[job.job_id] = job
            self.job_to_node[job.job_id] = node

            # 노드 running list에 추가 (deallocate_job 및 consolidation에서 사용)
            if job.job_type in ("HP", "HP-scale-out"):
                node.hp_running_jobs.append(job)
            else:
                node.spot_running_jobs.append(job)

            completion_event = Event(
                time=job.end_time,
                event_type='completion',
                job=job
            )
            heapq.heappush(self.event_queue, completion_event)
            # 2. 실제 배포
            if isinstance(job, RANJob):
                release = job.name
                gpu_resource = deploy.get_gpu_resource(target_slice['size'], node.name)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                repo_root = os.path.dirname(script_dir)
                helm_chart = os.path.join(repo_root, "workload", "heng", release)
                launch_pattern = getattr(job, 'launch_pattern', 'F08 1C 59') # default 로 1C 처리
                oam_port = RAN_IP_PORT_MAP.get(release, 50053)
                metrics_port = RAN_METRICS_PORT_MAP.get(release, 8086)
                helm_cmd = [
                    "helm", "install", release, helm_chart,
                    "--set", f"nodeName={node.name}",
                    "--set", f"gpuResource={gpu_resource}",
                    "--set", f"launchPattern={launch_pattern}",
                    "--set", f"aerial_metrics_backend_address=127.0.0.1:{metrics_port}",
                    "--set", f"oam_server_addr=127.0.0.2:{oam_port}",
                ]
                print(f"[RAN-Deploy] Running: {' '.join(helm_cmd)}")
                try:
                    result = subprocess.run(helm_cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        print(f"[RAN-Deploy] Installed {release}")
                    else:
                        print(f"[RAN-Deploy] Install failed: {result.stderr}")
                        return
                except Exception as e:
                    print(f"[RAN-Deploy] Install error: {e}")
                    return

                mig_info = deploy.get_mig_instance_uuids(node.name, node.gpu_index, include_size=True)
                if mig_info:
                    matching = [uuid for sz, uuid in mig_info if sz == target_slice['size']]
                    si = self._compute_size_index(node, target_slice)
                    if matching and si < len(matching):
                        job.mig_uuid = matching[si]
                    elif matching:
                        job.mig_uuid = matching[0]
                    elif target_slice['id'] < len(mig_info):
                        job.mig_uuid = mig_info[target_slice['id']][1]  # fallback to positional

                container = "aerial-testmac-ctr"
                print(f"[RAN-Deploy] Waiting for {release} running...")
                self._wait_pod_running(release)

                print(f"[RAN-Deploy] Waiting for {release} app ready...")
                self._wait_app_ready(release, container)

                print(f"[RAN-Deploy] Activating cells on {release} (cmd 1)")
                ip = "127.0.0.2"
                port = RAN_IP_PORT_MAP.get(release, 50053)
                self._run_cell_cmd(release, container, ip, port, getattr(job, 'cell_group_num', 1), cmd_id=3)
                self._run_cell_cmd(release, container, ip, port, getattr(job, 'cell_group_num', 1), cmd_id=1)

                print(f"[RAN-Deploy] {release} deploy complete")
            else:
                mig_uuid = deploy.deploy_job(
                        job=job,
                        node_name=node.name,
                        gpu_g=target_slice['size'],
                        gpu_index=node.gpu_index,
                        slice_index=target_slice['id'],
                        size_index=self._compute_size_index(node, target_slice),
                )
                if mig_uuid:
                    job.mig_uuid = mig_uuid

        elif job.job_type == "scale-out":
            # o.repliac의 경우 타겟이 있는지 검사해야함
            target_job_id = job.target_job_id

            if target_job_id not in self.running_jobs:
                print(f"    Error: Target job {target_job_id} not running")
                return

            target_job = self.running_jobs[target_job_id]

            # 1. 상태 처리 GPU Node & Job Status 업데이트 -> TODO: 이 클래스의 메소드로 뺴면 좋을 듯
            target_slice['used'] = job.req
            target_slice['jobs'].append(job.job_id)
            job.allocated_size = target_slice['size']

            actual_time = (time.time() - self.wall_start) * self.speed + self.start_time
            job.start_time = actual_time

            if job.job_id in self.job_states:
                remaining_duration = self.job_states[job.job_id].remaining_duration
                self.job_states[job.job_id].actual_start_time = actual_time
            else:
                remaining_duration = job.duration
            job.end_time = actual_time + (remaining_duration * 60)

            self.running_jobs[job.job_id] = job
            self.job_to_node[job.job_id] = node

            # 노드 running list에 추가
            if job.job_type in ("HP", "HP-scale-out"):
                node.hp_running_jobs.append(job)
            else:
                node.spot_running_jobs.append(job)

            completion_event = Event(
                time=job.end_time,
                event_type='completion',
                job=job
            )
            heapq.heappush(self.event_queue, completion_event)

            # Replica counter 초기화 (처음 scale-out인 경우)
            if target_job_id not in self.replica_counter:
                self.replica_counter[target_job_id] = 0
                self.hp_replicas[target_job_id] = []

            # Replica ID 생성
            self.replica_counter[target_job_id] += 1
            replica_id = f"{target_job_id}-replica-{self.replica_counter[target_job_id]}"
            self.hp_replicas[target_job_id].append(replica_id)
            # 2. 실제 배포
            mig_uuid = deploy.deploy_job_replica(
                replica_id=replica_id,
                target_job_id=job.target_job_id,  # 원본 job의 service에 연결
                node_name=node.name,
                gpu_g=target_slice['size'],
                gpu_index=node.gpu_index,
                slice_index=target_slice['id'],
                size_index=self._compute_size_index(node, target_slice),
            )
            if mig_uuid:
                job.mig_uuid = mig_uuid
        elif job.job_type == "migration" and job.workload_type == "AI":
            pass
        elif job.job_type == "migration" and job.workload_type == "RAN":
            actual_time = (time.time() - self.wall_start) * self.speed + self.start_time

            # 0. target_slice에 기존 job이 있으면 preempt
            if target_slice['jobs']:
                evict_job_id = target_slice['jobs'][0]
                print(f"[RAN-Migration] Preempting job {evict_job_id} on slice[{target_slice['id']}] ({target_slice['size']}g)")

                if evict_job_id in self.running_jobs:
                    evict_job = self.running_jobs[evict_job_id]

                    # completion event 취소
                    for ev in self.event_queue:
                        if ev.event_type == 'completion' and ev.job and ev.job.job_id == evict_job_id:
                            ev.cancelled = True
                            print(f"[RAN-Migration] Cancelled completion event for {evict_job_id}")
                            break

                    # remaining duration 계산
                    elapsed_min = (actual_time - evict_job.start_time) / 60 if evict_job.start_time else 0
                    if evict_job_id in self.job_states:
                        self.job_states[evict_job_id].remaining_duration = max(0, self.job_states[evict_job_id].remaining_duration - elapsed_min)
                        new_remaining = self.job_states[evict_job_id].remaining_duration
                    else:
                        new_remaining = max(0, evict_job.duration - elapsed_min)
                    evict_job.duration = new_remaining
                    print(f"[RAN-Migration] {evict_job_id} elapsed={elapsed_min:.1f}min, remaining={new_remaining:.1f}min")

                    # 실제 undeploy + pod 삭제 대기
                    if not self.dry_run:
                        deploy.undeploy_job(evict_job_id)
                        deploy.wait_for_pod_deletion([evict_job_id], max_wait_sec=60)

                    # slice 상태 초기화
                    target_slice['used'] = 0
                    target_slice['jobs'].clear()

                    # running 상태 제거
                    del self.running_jobs[evict_job_id]
                    if evict_job_id in self.job_to_node:
                        del self.job_to_node[evict_job_id]

                    # 재큐잉
                    evict_job.start_time = None
                    evict_job.end_time = None
                    evict_job.mig_uuid = None
                    evict_job.allocated_size = None
                    evict_job.job_type = "deploy"
                    requeue_event = Event(
                        time=actual_time,
                        event_type='arrival',
                        job=evict_job
                    )
                    heapq.heappush(self.event_queue, requeue_event)
                    print(f"[RAN-Migration] Re-queued {evict_job_id} as deploy to allocate RAN") # 걱정된느 부분 현재 진행중인 RAN MIgration event랑 안 겹치나??

            # src/dst 정보
            src_job_id = job.target_job_id
            if src_job_id not in self.running_jobs:
                print(f"[RAN-Migration] Error: Source job {src_job_id} not running")
                return
            src_job = self.running_jobs[src_job_id]
            src_pod = src_job.name       # "aerial-l1" or "aerial-l1-1"
            dst_pod = job.name           # "aerial-l1-1" or "aerial-l1"
            container = "aerial-testmac-ctr"
            src_cells = getattr(src_job, 'cell_group_num', 1)
            dst_cells = getattr(job, 'cell_group_num', 1)
            # i aerial-l1 50052, aerial-l1-1 50053
            src_port = RAN_IP_PORT_MAP.get(src_pod, 50053)
            dst_port = RAN_IP_PORT_MAP.get(dst_pod, 50053)
            launch_pattern = getattr(job, 'launch_pattern', 'F08 1C 59')
            print(f"[RAN-Migration] {src_pod} -> {dst_pod} (src_cells={src_cells}, dst_cells={dst_cells})")

            # 상태 처리
            target_slice['used'] = job.req
            target_slice['jobs'].append(job.job_id)
            job.allocated_size = target_slice['size']
            job.start_time = actual_time
            if job.job_id in self.job_states:
                remaining_duration = self.job_states[job.job_id].remaining_duration
                self.job_states[job.job_id].actual_start_time = actual_time
            else:
                remaining_duration = job.duration
            job.end_time = actual_time + (remaining_duration * 60)
            self.running_jobs[job.job_id] = job
            self.job_to_node[job.job_id] = node
            heapq.heappush(self.event_queue, Event(time=job.end_time, event_type='completion', job=job))

            # Step 1: Deactivate Source Cells (cmd 0)
            # print(f"[RAN-Migration] Step 1: Deactivate src cells on {src_pod}")
            src_ip = "127.0.0.2"
            # self._run_cell_cmd(src_pod, container, src_ip, src_port, src_cells, cmd_id=0)

            # delay 에상
            # Step 2: Helm Install Dst Pod
            print(f"[RAN-Migration] Step 2: Install dst pod {dst_pod}")
            gpu_resource = deploy.get_gpu_resource(target_slice['size'], node.name)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            helm_chart = os.path.join(repo_root, "workload", dst_pod) # 백업 버전으로 테스트 
            dst_metrics_port = RAN_METRICS_PORT_MAP.get(dst_pod, 8086)
            dst_oam_port = RAN_IP_PORT_MAP.get(dst_pod, 50053)
            helm_cmd = [
                "helm", "install", dst_pod, helm_chart,
                "--set", f"nodeName={node.name}",
                "--set", f"gpuResource={gpu_resource}",
                "--set", f"launchPattern={launch_pattern}",
                "--set", f"aerial_metrics_backend_address=127.0.0.1:{dst_metrics_port}",
                "--set", f"oam_server_addr=127.0.0.2:{dst_oam_port}",
            ]
            try:
                result = subprocess.run(helm_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"[RAN-Migration] Dst pod {dst_pod} installed")
                else:
                    print(f"[RAN-Migration] Dst pod install failed: {result.stderr}")
                    return
            except Exception as e:
                print(f"[RAN-Migration] Dst pod install error: {e}")
                return

            mig_info = deploy.get_mig_instance_uuids(node.name, node.gpu_index, include_size=True)
            if mig_info:
                matching = [uuid for sz, uuid in mig_info if sz == target_slice['size']]
                si = self._compute_size_index(node, target_slice)
                if matching and si < len(matching):
                    job.mig_uuid = matching[si]
                elif matching:
                    job.mig_uuid = matching[0]
                elif target_slice['id'] < len(mig_info):
                    job.mig_uuid = mig_info[target_slice['id']][1]  # fallback to positional

            # Step 3: Wait for Dst Pod Running
            print(f"[RAN-Migration] Step 3: Wait for {dst_pod} running")
            if not self._wait_pod_running(dst_pod):
                print(f"[RAN-Migration] Dst pod not running, aborting")
                return
            time.sleep(1)

            # Step 4: Turn on Traffic on Src Pod (cmd 1)
            # print(f"[RAN-Migration] Step 4: Activate src cells on {src_pod}")
            # self._run_cell_cmd(src_pod, container, src_ip, src_port, src_cells, cmd_id=1)

            # Step 5: Wait for Dst Pod App Ready (log check)
            print(f"[RAN-Migration] Step 5: Wait for {dst_pod} app ready")
            if not self._wait_app_ready(dst_pod, container):
                print(f"[RAN-Migration] Dst app not ready, aborting")
                return
            time.sleep(20)

            # Step 6: Turn off Traffic on Src Pod (cmd 0)
            print(f"[RAN-Migration] Step 6: Deactivate src cells on {src_pod}")
            self._run_cell_cmd(src_pod, container, src_ip, src_port, src_cells, cmd_id=0)
            t0 = time.time()
            elapsed = time.time() - t0
            print(f"[RAN-Migration] Step 6→7 sleep: {elapsed:.3f}s")

            # Step 7: Configure Destination Cells (cmd 3)
            print(f"[RAN-Migration] Step 7: Configure dst cells on {dst_pod}")
            dst_ip = "127.0.0.2"
            self._run_cell_cmd(dst_pod, container, dst_ip, dst_port, dst_cells, cmd_id=3)

            # Step 8: Activate Destination Cells (cmd 1)
            print(f"[RAN-Migration] Step 8: Activate dst cells on {dst_pod}")
            self._run_cell_cmd(dst_pod, container, dst_ip, dst_port, dst_cells, cmd_id=1)

            # Step 9: Uninstall Src Pod
            print(f"[RAN-Migration] Step 9: Uninstall src pod {src_pod}")
            try:
                subprocess.run(["helm", "uninstall", src_pod], capture_output=True, text=True, timeout=30)
            except Exception as e:
                print(f"[RAN-Migration] Src uninstall error: {e}")

            # src job 상태 정리
            for ev in self.event_queue:
                if ev.event_type == 'completion' and ev.job and ev.job.job_id == src_job_id:
                    ev.cancelled = True
                    break
            freed_slice = None
            src_node = self.job_to_node.get(src_job_id)
            if src_node:
                for s in src_node.slices:
                    if src_job_id in s['jobs']:
                        s['used'] = 0
                        s['jobs'].remove(src_job_id)
                        freed_slice = s
                        break
            if src_job_id in self.running_jobs:
                del self.running_jobs[src_job_id]
            if src_job_id in self.job_to_node:
                del self.job_to_node[src_job_id]

            print(f"[RAN-Migration] Complete: {src_pod} -> {dst_pod}")

            # RAN Migration 완료 후: 빈 슬라이스에 대기 Job 배치 → Consolidation
            if freed_slice and src_node:
                self._try_schedule_waiting(src_node)
                # 여전히 비어있으면 consolidation
                if len(freed_slice['jobs']) == 0 and not self._consolidation_in_progress and src_node.current_plan is None:
                    self.try_consolidate_ai(src_node, freed_slice)

    # =========================================================================
    # handle_completion override (Phase 1: Consolidation 트리거)
    # =========================================================================
    def handle_completion(self, job):
        """Job 완료 처리 + 대기 Job 스케줄링 + AI Consolidation 트리거"""
        # 완료된 job의 노드 정보 저장 (삭제 전)
        completed_node = None
        if job.job_id in self.job_to_node:
            completed_node = self.job_to_node[job.job_id]

        # 중복 완료 방지: 이미 완료된 job이면 early return
        already_completed = job.job_id in {j.job_id for j in self.completed_jobs}
        if already_completed:
            # 이미 완료된 job의 stale completion event → 무시
            return

        if job.job_id in self.running_jobs:
            if job.job_id in self.job_states:
                self.update_job_state_on_completion(job.job_id, self.current_time)

            if not self.dry_run:
                self.undeploy(job)

            if completed_node:
                completed_node.deallocate_job(job)
                del self.job_to_node[job.job_id]

            del self.running_jobs[job.job_id]

            if not self.dry_run:
                if not deploy.wait_for_pod_deletion([job.job_id], max_wait_sec=60):
                    print(f"    Warning: Pod {job.job_id} may still be running!")

        # Snapshot completion_time in job_states (immune to Job object aliasing)
        # Set even if not in running_jobs — ILP may have removed it but the event still fires
        if job.job_id in self.job_states and self.job_states[job.job_id].completion_time is None:
            self.job_states[job.job_id].completion_time = self.current_time

        # Add to completed_jobs (duplicate check already done above)
        if job.job_id not in {j.job_id for j in self.completed_jobs}:
            self.completed_jobs.append(job)
            self.total_jct += job.jct()
            self.total_wait_time += job.wait_time()

        time_str = self.format_time(self.current_time)
        if completed_node:
            print(f" [{time_str}] {job.job_id} completed on {completed_node.name} ({job.req}g freed)")
            print(f"  {completed_node.get_slice_info()}")
        else:
            print(f" [{time_str}] {job.job_id} completed ({job.req}g freed)")

        # BinPack 고유 로직: 빈 슬라이스에 대기 Job 우선 배치 → Consolidation
        if completed_node:
            # Slot plan이 활성화된 경우: 조기 slot 전환 체크
            if completed_node.current_plan is not None:
                self.check_slot_transition(completed_node)
                return

            # freed_slice 식별 (deallocate 후 비어있는 슬라이스)
            freed_slices = [s for s in completed_node.slices if len(s['jobs']) == 0]

            # 1. 대기 Job 우선 배치
            if freed_slices:
                self._try_schedule_waiting(completed_node)

            # 2. Consolidation (빈 슬라이스가 여전히 남아있을 때)
            remaining_free = [s for s in completed_node.slices if len(s['jobs']) == 0]
            if (remaining_free
                    and not self._consolidation_in_progress
                    and completed_node.current_plan is None):
                for freed_slice in remaining_free:
                    self.try_consolidate_ai(completed_node, freed_slice)
                    # 슬라이스가 채워졌으면 다음 슬라이스로
                    if len(freed_slice['jobs']) > 0:
                        continue

    # =========================================================================
    # _try_schedule_waiting: 빈 슬라이스에 대기 Job 배치
    # =========================================================================
    def _try_schedule_waiting(self, node):
        """node의 빈 슬라이스에 맞는 대기 중인 arrival job을 즉시 배치"""
        free_slices = node.get_free_slice_list()
        if not free_slices:
            return

        free_sizes = {s['size'] for s in free_slices}

        # event_queue에서 arrival 이벤트 중 이 노드에 맞는 job 찾기
        waiting_arrivals = []
        for i, ev in enumerate(self.event_queue):
            if (ev.event_type == 'arrival'
                    and not ev.cancelled
                    and ev.job is not None
                    and ev.job.req in free_sizes):
                waiting_arrivals.append((i, ev))

        # 매칭되는 job을 즉시 스케줄링 (가장 빠른 arrival 먼저)
        scheduled_indices = []
        for idx, ev in waiting_arrivals:
            # 아직 빈 슬라이스가 있는지 재확인
            current_free = node.get_free_slice_list()
            if not current_free:
                break
            matching = [s for s in current_free if s['size'] >= ev.job.req]
            if matching:
                ev.cancelled = True
                scheduled_indices.append(idx)
                print(f"  [Consolidation] Scheduling waiting job {ev.job.job_id} ({ev.job.req}g) to freed slot")
                self.handle_arrival(ev.job)

    # =========================================================================
    # try_consolidate_ai: Consolidation 후보 탐색 + 실행
    # =========================================================================
    def try_consolidate_ai(self, target_node, freed_slice):
        """빈 슬라이스에 다른 노드의 AI Spot job을 이동하여 utilization 개선"""
        # Guard: target에 Slot Plan이 있으면 skip
        if target_node.current_plan is not None:
            return

        # Guard: 이미 채워졌으면 skip
        if len(freed_slice['jobs']) > 0:
            return

        S = freed_slice['size']
        target_used = sum(s['used'] for s in target_node.slices)

        best_candidate = None
        best_score = 0.1  # threshold

        for node in self.nodes:
            if node is target_node:
                continue
            # Guard: source에 Slot Plan이 있으면 skip
            if node.current_plan is not None:
                continue

            source_used = sum(s['used'] for s in node.slices)

            for spot_job in list(node.spot_running_jobs):
                # AI Spot job만 대상
                if getattr(spot_job, 'workload_type', 'AI') != "AI":
                    continue
                # 크기가 맞아야 함
                if spot_job.req != S:
                    continue

                # Scoring
                target_util_after = (target_used + S) / target_node.total_capacity
                source_util_after = (source_used - S) / node.total_capacity
                score = target_util_after - source_util_after

                # source slice 찾기
                src_slice = None
                for s in node.slices:
                    if spot_job.job_id in s['jobs']:
                        src_slice = s
                        break

                if src_slice and score > best_score:
                    best_score = score
                    best_candidate = (spot_job, node, src_slice)

        if best_candidate:
            job, src_node, src_slice = best_candidate
            print(f"  [Consolidation] Moving {job.job_id} ({job.req}g) "
                  f"from {src_node.name} GPU{src_node.gpu_index} → "
                  f"{target_node.name} GPU{target_node.gpu_index} (score={best_score:.2f})")
            self._execute_ai_consolidation(job, src_node, src_slice, target_node, freed_slice)

    # =========================================================================
    # _execute_ai_consolidation: Source에서 제거 + Target에 배포
    # =========================================================================
    def _execute_ai_consolidation(self, job, src_node, src_slice, dst_node, dst_slice):
        """AI Spot job을 src_node에서 dst_node로 이동"""
        self._consolidation_in_progress = True
        try:
            # --- Phase A: Source에서 제거 ---
            print(f"    [Consolidation] Phase A: Remove {job.job_id} from {src_node.name}")

            # 1. Completion event 취소
            self._remove_completion_event(job.job_id)

            # 2. Remaining duration 계산
            if not self.dry_run:
                current_time = (time.time() - self.wall_start) * self.speed + self.start_time
            else:
                current_time = self.current_time

            remaining_min = job.duration  # fallback
            if job.job_id in self.job_states:
                state = self.job_states[job.job_id]
                if state.actual_start_time is not None:
                    elapsed_min = (current_time - state.actual_start_time) / 60
                    state.total_run_time += elapsed_min
                    remaining_min = max(0, state.remaining_duration - elapsed_min)
                    state.remaining_duration = remaining_min
                else:
                    remaining_min = state.remaining_duration
                state.actual_start_time = None

            # 3. Undeploy (실제 모드에서만)
            if not self.dry_run:
                deploy.undeploy_job(job.job_id)
                deploy.wait_for_pod_deletion([job.job_id], max_wait_sec=60)

            # 4. Source slice 상태 초기화
            src_slice['used'] = 0
            src_slice['jobs'].clear()
            if job in src_node.spot_running_jobs:
                src_node.spot_running_jobs.remove(job)

            # 5. Running 상태 정리
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            if job.job_id in self.job_to_node:
                del self.job_to_node[job.job_id]

            # MIG 상태 초기화
            job.mig_uuid = None
            job.allocated_size = None

            # --- Phase B: Target에 배포 ---
            print(f"    [Consolidation] Phase B: Deploy {job.job_id} to {dst_node.name} slice[{dst_slice['id']}] ({dst_slice['size']}g)")

            # 1. Dst slice 상태 업데이트
            dst_slice['used'] = job.req
            dst_slice['jobs'].append(job.job_id)
            job.allocated_size = dst_slice['size']

            # 2. Timing 설정
            job.start_time = current_time
            job.duration = remaining_min
            job.end_time = current_time + (remaining_min * 60)

            if job.job_id in self.job_states:
                self.job_states[job.job_id].actual_start_time = current_time

            # 3. Running 상태 등록
            self.running_jobs[job.job_id] = job
            self.job_to_node[job.job_id] = dst_node
            dst_node.spot_running_jobs.append(job)

            # 4. Completion event 생성
            completion_event = Event(
                time=job.end_time,
                event_type='completion',
                job=job
            )
            heapq.heappush(self.event_queue, completion_event)

            # 5. 실제 배포
            if not self.dry_run:
                mig_uuid = deploy.deploy_job(
                    job=job,
                    node_name=dst_node.name,
                    gpu_g=dst_slice['size'],
                    gpu_index=dst_node.gpu_index,
                    slice_index=dst_slice['id'],
                    size_index=self._compute_size_index(dst_node, dst_slice),
                )
                if mig_uuid:
                    job.mig_uuid = mig_uuid

            print(f"    [Consolidation] Done: {job.job_id} moved, remaining={remaining_min:.1f}min")

        finally:
            self._consolidation_in_progress = False

    # =========================================================================
    # HP-AI Preemption with ILP (Phase 2)
    # =========================================================================
    def _select_node_for_hp_preempt(self, hp_job):
        # """기준을 다르게 하면 성능이 달라질 )"""
        best_node = None
        max_spot_count = 0
        for node in self.nodes:
            spot_count = len(node.spot_running_jobs)
            if spot_count > max_spot_count:
                max_spot_count = spot_count
                best_node = node
        return best_node

    def _hp_preempt_with_ilp(self, hp_job, node):
        """HP-AI deploy를 위해 노드의 모든 job을 중지 → ILP → slot plan 기반 재배포

        기존 realtime_slot_scheduler의 slot 메커니즘을 활용하여
        ILP의 multi-slot reconfig 계획을 실제로 실행한다.
        """
        from bin_pack import run_bin_pack

        log_header = f"\n{'='*60}\n  [ILP-Preempt] HP-AI deploy {hp_job.job_id} ({hp_job.req}g)\n  [ILP-Preempt] Target: {node.name} GPU{node.gpu_index}\n  [ILP-Preempt] Before profile: {node.mig_profile}\n  [ILP-Preempt] Before slices: {node.get_slice_info()}\n{'='*60}"
        print(log_header)
        logger.info(log_header)

        # 1. Save: 현재 노드의 모든 running jobs
        running_hp_jobs = list(node.hp_running_jobs)
        running_spot_jobs = list(node.spot_running_jobs)
        all_running = running_hp_jobs + running_spot_jobs
        all_running_ids = [j.job_id for j in all_running]

        log_running = f"  [ILP-Preempt] Running HP: {[j.job_id for j in running_hp_jobs]}\n  [ILP-Preempt] Running Spot: {[j.job_id for j in running_spot_jobs]}"
        print(log_running)
        logger.info(log_running)

        # 1.5 기존 spot_waiting_queue flush → arrival 이벤트로 re-queue
        # (새 plan이 old plan을 덮어쓰므로, old plan의 waiting job들이 stuck되는 것을 방지)
        if node.spot_waiting_queue:
            orphan_jobs = list(node.spot_waiting_queue)
            node.spot_waiting_queue.clear()
            print(f"  [ILP-Preempt] Re-queuing {len(orphan_jobs)} orphaned waiting jobs")
            for j in orphan_jobs:
                if j.job_id not in all_running_ids and j.job_id != hp_job.job_id:
                    requeue_event = Event(time=self.current_time + 1, event_type='arrival', job=j)
                    heapq.heappush(self.event_queue, requeue_event)

        # 2. Calculate remaining_duration
        if not self.dry_run:
            current_time = (time.time() - self.wall_start) * self.speed + self.start_time
        else:
            current_time = self.current_time

        job_remaining = {}
        for j in all_running:
            remaining = j.duration
            if j.job_id in self.job_states and self.job_states[j.job_id].actual_start_time is not None:
                state = self.job_states[j.job_id]
                elapsed_min = (current_time - state.actual_start_time) / 60
                state.total_run_time += elapsed_min
                remaining = max(0, state.remaining_duration - elapsed_min)
                state.remaining_duration = remaining
                state.actual_start_time = None
            job_remaining[j.job_id] = remaining

        # Cancel all completion events
        for j in all_running:
            self._remove_completion_event(j.job_id)

        # 3. Undeploy ALL
        if not self.dry_run and all_running_ids:
            print(f"  [ILP-Preempt] Undeploying ALL {len(all_running_ids)} jobs...")
            success_count, failed = deploy.undeploy_jobs_batch(all_running_ids)
            print(f"  [ILP-Preempt] Undeployed: {success_count}/{len(all_running_ids)}")
            deploy.wait_for_pod_deletion(all_running_ids, max_wait_sec=120)

        # Clear node state
        for j in all_running:
            j.mig_uuid = None
            j.allocated_size = None
            j.start_time = None
            if j.job_id in self.running_jobs:
                del self.running_jobs[j.job_id]
            if j.job_id in self.job_to_node:
                del self.job_to_node[j.job_id]
        node.hp_running_jobs.clear()
        node.spot_running_jobs.clear()
        for s in node.slices:
            s['used'] = 0
            s['jobs'].clear()

        # 4. ILP 호출: 전체 7g available, 모든 job 포함
        slot_minutes = 30  # HP preemption은 빠른 계획 필요 → 큰 slot
        all_jobs_for_ilp = running_spot_jobs + running_hp_jobs + [hp_job]
        ilp_job_list = []
        for j in all_jobs_for_ilp:
            if j.job_id == hp_job.job_id:
                dur = hp_job.duration
            else:
                dur = job_remaining.get(j.job_id, j.duration)
            ilp_job_list.append({"name": j.job_id, "size": j.req, "duration": dur})

        # HP job 이름 수집 (ILP에서 HP 우선 배치용)
        hp_job_names = {hp_job.job_id} | {j.job_id for j in running_hp_jobs}

        ilp_input_log = f"  [ILP-Preempt] ILP input: {len(ilp_job_list)} jobs (HP: {len(hp_job_names)})\n"
        for jd in ilp_job_list:
            is_hp = jd['name'] in hp_job_names
            hp_mark = " [HP]" if is_hp else ""
            ilp_input_log += f"    {jd['name']}: {jd['size']}g, {jd['duration']:.1f}min{hp_mark}\n"
        print(ilp_input_log.rstrip())
        logger.info(ilp_input_log.rstrip())

        ilp_result = None
        try:
            ilp_result = run_bin_pack(
                init_config="1111111",
                job_list=ilp_job_list,
                slot_minutes=slot_minutes,
                dry_run=True,  # ILP는 항상 static config 사용
                hp_job_names=hp_job_names,  # HP 우선 배치
            )
        except Exception as e:
            print(f"  [ILP-Preempt] ILP failed: {e}")

        # 5. ILP 결과 처리
        if not (ilp_result and 'chosen_cfg' in ilp_result and ilp_result['chosen_cfg']):
            # Fallback: HP job의 req 크기와 나머지를 1g로 채움
            hp_size = hp_job.req
            remaining_g = 7 - hp_size
            new_profile = str(hp_size) + "1" * remaining_g
            print(f"  [ILP-Preempt] ILP failed, fallback profile: {new_profile}")

            if not self.dry_run:
                success = deploy.hard_reconfigure_mig(node.name, node.gpu_index, new_profile)
                if not success:
                    print(f"  [ILP-Preempt] Reconfig failed!")
                    return
            node.reconfigure(new_profile)

            # HP만 배치, Spot은 다음 completion 시점에 재시도
            self._deploy_hp_jobs_manual([hp_job] + running_hp_jobs, job_remaining, node, current_time)

            # Spot jobs: 다음 completion 시점에 재큐잉 (무한루프 방지)
            next_completion = min(
                (e.time for e in self.event_queue if e.event_type == 'completion' and not e.cancelled),
                default=current_time + 60
            )
            for j in running_spot_jobs:
                j.start_time = None
                j.end_time = None
                j.job_type = "deploy"
                j.duration = job_remaining.get(j.job_id, j.duration)
                requeue_event = Event(time=next_completion, event_type='arrival', job=j)
                heapq.heappush(self.event_queue, requeue_event)
                print(f"  [ILP-Preempt] Re-queued Spot {j.job_id} at t={next_completion:.0f}s")
            return

        chosen_cfg = ilp_result['chosen_cfg']
        slot_jobs_map = ilp_result.get('slot_jobs', {})
        reconfigs = ilp_result.get('reconfigs', [])
        tmax_slot = ilp_result.get('Tmax_slot', len(chosen_cfg))

        # 6. 첫 slot의 full profile로 reconfig
        before_profile = node.mig_profile
        first_profile = chosen_cfg.get(1, chosen_cfg.get(0, ""))
        reconfig_log = f"  [ILP-Preempt] Reconfiguring: {before_profile} → {first_profile}"
        print(reconfig_log)
        logger.info(reconfig_log)
        if not self.dry_run:
            success = deploy.hard_reconfigure_mig(node.name, node.gpu_index, first_profile)
            if not success:
                print(f"  [ILP-Preempt] Reconfig failed! Reverting...")
                return
        node.reconfigure(first_profile)

        # 7. Slot-1 HP jobs vs Later-slot HP jobs 구분
        all_hp_jobs = [hp_job] + running_hp_jobs
        all_hp_job_ids = {j.job_id for j in all_hp_jobs}
        slot1_job_ids = set(slot_jobs_map.get(1, []))
        hp_slot1_ids = all_hp_job_ids & slot1_job_ids
        hp_later_ids = all_hp_job_ids - hp_slot1_ids

        hp_slot1_jobs = [j for j in all_hp_jobs if j.job_id in hp_slot1_ids]
        hp_later_jobs = [j for j in all_hp_jobs if j.job_id in hp_later_ids]

        print(f"  [ILP-Preempt] HP slot-1: {[j.job_id for j in hp_slot1_jobs]}")
        if hp_later_jobs:
            print(f"  [ILP-Preempt] HP later-slot: {[j.job_id for j in hp_later_jobs]}")

        # 8. Slot-1 HP jobs 즉시 배치
        self._deploy_hp_jobs_manual(hp_slot1_jobs, job_remaining, node, current_time)

        # 9. Later-slot HP jobs → spot_waiting_queue (deploy 타입으로 변환)
        for j in hp_later_jobs:
            j.start_time = None
            j.end_time = None
            j.job_type = "deploy"
            j.duration = job_remaining.get(j.job_id, j.duration)
            node.spot_waiting_queue.append(j)

        # 10. Spot jobs → spot_waiting_queue에 넣기
        for j in running_spot_jobs:
            j.start_time = None
            j.end_time = None
            j.job_type = "deploy"
            j.duration = job_remaining.get(j.job_id, j.duration)
            node.spot_waiting_queue.append(j)

        # 11. Spot-only plan 생성 (slot-1 HP만 제거, later-slot HP는 유지)
        spot_chosen_cfg = {}
        for t, cfg in chosen_cfg.items():
            # Slot-1 HP 중 이 slot에서 아직 active한 것만 strip
            active_hp_at_t = [j for j in hp_slot1_jobs if j.job_id in slot_jobs_map.get(t, [])]
            spot_chosen_cfg[t] = self._strip_hp_from_profile(cfg, active_hp_at_t)

        spot_slot_jobs = {}
        for t, jobs_list in slot_jobs_map.items():
            # Slot-1 HP만 제외 (이미 수동 배치됨), later-slot HP와 Spot은 유지
            spot_slot_jobs[t] = [jid for jid in jobs_list if jid not in hp_slot1_ids]

        # Slot 1은 이미 reconfig 했으므로 제거
        adjusted_reconfigs = [r for r in reconfigs if r != 1]

        plan = {
            'status': ilp_result.get('status', 'Optimal'),
            'chosen_cfg': spot_chosen_cfg,
            'slot_jobs': spot_slot_jobs,
            'reconfigs': adjusted_reconfigs,
            'Tmax_slot': tmax_slot,
            'reserved_profile': None,
            'cleanup_mode': False,
        }

        node.current_plan = plan
        node.current_slot = 1
        node.plan_start_time = current_time

        slot_plan_log = f"  [ILP-Preempt] Slot plan created: Tmax={tmax_slot}, reconfigs={adjusted_reconfigs}\n"
        for t in range(1, min(tmax_slot + 1, 4)):
            slot_plan_log += f"    Slot {t}: profile={spot_chosen_cfg.get(t, '?')}, jobs={spot_slot_jobs.get(t, [])}\n"
        if tmax_slot > 3:
            slot_plan_log += f"    ... ({tmax_slot - 3} more slots)\n"
        print(slot_plan_log.rstrip())
        logger.info(slot_plan_log.rstrip())

        # 12. Slot timeout 이벤트 스케줄링
        slot_duration_sec = slot_minutes * 60
        for slot_idx in range(1, tmax_slot + 1):
            timeout_time = node.plan_start_time + slot_idx * slot_duration_sec
            timeout_event = Event(
                time=timeout_time,
                event_type='slot_timeout',
                job=None,
                node=node,
                slot_idx=slot_idx,
                plan=plan,
            )
            heapq.heappush(self.event_queue, timeout_event)

        # 13. 첫 slot의 Spot 배포 (기존 slot 메커니즘 활용)
        self._execute_current_slot(node)

        done_log = f"  [ILP-Preempt] Done. Profile: {node.mig_profile}\n  [ILP-Preempt] {node.get_slice_info()}"
        print(done_log)
        logger.info(done_log)

    def _deploy_hp_jobs_manual(self, hp_jobs_to_deploy, job_remaining, node, current_time):
        """HP jobs를 수동 best-fit으로 배치 (slot plan과 독립적으로 유지)"""
        # Job state 생성 (없는 경우)
        for j in hp_jobs_to_deploy:
            if j.job_id not in self.job_states:
                self.job_states[j.job_id] = JobState(
                    job_id=j.job_id,
                    original_duration=j.duration,
                    remaining_duration=j.duration,
                    job_type=j.job_type,
                    workload_type=getattr(j, 'workload_type', 'AI'),
                    submit_time=current_time
                )

        for j in hp_jobs_to_deploy:
            best_slice = None
            min_waste = float('inf')
            for s in node.slices:
                if len(s['jobs']) == 0 and s['size'] >= j.req:
                    waste = s['size'] - j.req
                    if waste < min_waste:
                        min_waste = waste
                        best_slice = s

            if not best_slice:
                print(f"  [ILP-Preempt] Cannot fit HP {j.job_id} ({j.req}g)")
                continue

            best_slice['used'] = j.req
            best_slice['jobs'].append(j.job_id)
            j.allocated_size = best_slice['size']
            node.hp_running_jobs.append(j)

            remaining_dur = job_remaining.get(j.job_id, j.duration)
            j.start_time = current_time
            j.duration = remaining_dur
            j.end_time = current_time + (remaining_dur * 60)

            if j.job_id in self.job_states:
                self.job_states[j.job_id].actual_start_time = current_time

            self.running_jobs[j.job_id] = j
            self.job_to_node[j.job_id] = node

            heapq.heappush(self.event_queue, Event(time=j.end_time, event_type='completion', job=j))

            if not self.dry_run:
                mig_uuid = deploy.deploy_job(
                    job=j, node_name=node.name,
                    gpu_g=best_slice['size'],
                    gpu_index=node.gpu_index,
                    slice_index=best_slice['id'],
                    size_index=self._compute_size_index(node, best_slice),
                )
                if mig_uuid:
                    j.mig_uuid = mig_uuid

            print(f"  [ILP-Preempt] Deployed HP {j.job_id} ({j.req}g)")

    @staticmethod
    def _strip_hp_from_profile(full_profile, hp_jobs):
        """ILP full profile에서 HP가 차지하는 슬라이스를 제거하여 Spot-only profile 반환

        예: full_profile="31111", hp_jobs=[Job(req=1)] → "3111"
            full_profile="2221", hp_jobs=[Job(req=2)] → "21" (X) → "221" (제거 1개)
        """
        chars = list(full_profile)
        for j in hp_jobs:
            size_char = str(j.req)
            if size_char in chars:
                chars.remove(size_char)
        return "".join(chars)

    def handle_slot_timeout(self, event):
        """Slot timeout 처리 + plan 종료 시 spot_waiting_queue 재큐잉"""
        super().handle_slot_timeout(event)

        # Plan 종료 후 spot_waiting_queue에 남은 job들을 arrival 이벤트로 복구
        node = event.node
        if node.current_plan is None and len(node.spot_waiting_queue) > 0:
            print(f"  [ILP-Preempt] Plan finished, re-queuing {len(node.spot_waiting_queue)} Spot jobs")
            for j in list(node.spot_waiting_queue):
                node.spot_waiting_queue.remove(j)
                requeue_event = Event(time=self.current_time, event_type='arrival', job=j)
                heapq.heappush(self.event_queue, requeue_event)

    def check_slot_transition(self, node):
        """Slot 전환 + plan 종료 시 spot_waiting_queue 재큐잉"""
        super().check_slot_transition(node)

        # Plan 종료 후 spot_waiting_queue에 남은 job들을 arrival 이벤트로 복구
        if node.current_plan is None and len(node.spot_waiting_queue) > 0:
            print(f"  [ILP-Preempt] Plan finished, re-queuing {len(node.spot_waiting_queue)} Spot jobs")
            for j in list(node.spot_waiting_queue):
                node.spot_waiting_queue.remove(j)
                requeue_event = Event(time=self.current_time, event_type='arrival', job=j)
                heapq.heappush(self.event_queue, requeue_event)


def run_scheduler():
    global scheduler

    nodes = create_initial_nodes()
    scheduler = BinPackScheduler(
        nodes=nodes,
        node_selection_strategy="bin_pack",
        speed=1,
        dry_run=False
    )
    logger.info(f"Nodes: {len(nodes)}")
    for node in nodes:
        logger.info(f"  {node.name} GPU {node.gpu_index}: {node.mig_profile}")

    # 무한 루프 실행 (stop_event가 set되면 종료)
    scheduler.run_forever(stop_event)
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 스케줄러 관리"""
    global scheduler_thread

    # 시작: 스케줄러 스레드 시작
    logger.info("Starting scheduler thread...")
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # 스케줄러가 초기화될 때까지 대기
    while scheduler is None:
        time.sleep(0.1)

    # RAN prediction model 로드
    try:
        load_ran_model(MODEL_PATH)
    except Exception as e:
        logger.warning(f"Failed to load RAN model: {e}")

    logger.info(f"Server ready at http://{HOST}:{PORT}")

    yield

    # 종료: 스케줄러 정지
    logger.info("Stopping scheduler...")
    stop_event.set()
    if scheduler_thread:
        scheduler_thread.join(timeout=5)


app = FastAPI(
    title="Dynamic Scheduler API",
    description="HTTP 엔드포인트로 job을 받아서 스케줄링합니다.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_results_on_exit():
    """종료 시 결과 저장"""
    global scheduler, _log_file

    if scheduler is not None:
        print("\n" + "=" * 60)
        print("Saving results before exit...")
        print("=" * 60)

        try:
            # 메트릭 출력
            metrics = scheduler.get_metrics()
            print(f"\nFinal Metrics:")
            print(f"  Total Jobs: {metrics.get('total_jobs', 0)}")
            print(f"  Completed Jobs: {metrics.get('completed_jobs', 0)}")
            print(f"  Avg Wait: {metrics.get('avg_wait_min', 0):.2f} min")

            # 논문 메트릭 출력
            if hasattr(scheduler, 'print_paper_metrics'):
                scheduler.print_paper_metrics()

            # JSON 결과 저장
            import json
            result_path = LOG_PATH.replace('.log', '_results.json')
            result = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
            }
            if hasattr(scheduler, 'get_paper_metrics_summary'):
                result['paper_metrics'] = scheduler.get_paper_metrics_summary()

            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {result_path}")

        except Exception as e:
            print(f"Error saving results: {e}")

    if _log_file:
        print(f"\nLog saved to: {LOG_PATH}")
        _log_file.close()


# 백엔드 엔드포인트
@app.post("/submit", response_model=JobResponse)
async def submit_job(job_req: JobRequest):
    """
    Job 제출

    - job_id: 고유 식별자
    - job_type: deploy( 일반 배포)
    - req: MIG 크기 (1, 2, 3, 4, 7)
    - duration: 실행 시간 (분)
    """
    if scheduler is None:
        return JobResponse(
            status="error",
            job_id=job_req.job_id,
            submit_time=0,
            message="Scheduler not started"
        )

    # 스케쥴러에서 사용하는 job 객체 생성 
    # 현제는 workload_type (RAN, 일반 (AI)에 따라 구분)
    if job_req.workload_type == "RAN":
        job = RANJob(
            job_id=job_req.job_id,
            name=job_req.name,
            job_type=job_req.job_type,
            workload_type=job_req.workload_type,
            req=job_req.req,
            duration=job_req.duration,
            submit_time=scheduler.current_time,
            launch_pattern=job_req.launch_pattern or "F08 1C 59",
            cell_group_num=job_req.cell_group_num or 1,
        )
    else:
        job = Job(
            job_id=job_req.job_id,
            name=job_req.name,
            job_type=job_req.job_type,
            workload_type=job_req.workload_type,
            req=job_req.req,
            duration=job_req.duration,
            submit_time=scheduler.current_time,
        )

    # job이 scale-up/scale-down 인 경우, target 정보 객체에 반영
    if job_req.target_job_id:
        job.target_job_id = job_req.target_job_id

    # 스케줄러에 job 추가 (thread-safe)
    with scheduler_lock:
        scheduler.add_job(job)

    return JobResponse(
        status="submitted",
        job_id=job.job_id,
        submit_time=scheduler.current_time,
        message=f"Job submitted at T={scheduler.current_time:.1f}s"
    )

def main_endpoint():
    """기존 HTTP 엔드포인트 모드"""
    global logger

    # 로깅 설정
    logger = setup_logging(LOG_PATH)

    logger.info("=" * 60)
    logger.info("Dynamic Scheduler with HTTP Endpoint")
    logger.info("=" * 60)
    logger.info(f"Log file: {LOG_PATH}")
    logger.info(f"Speed: {SPEED}x")
    logger.info(f"Dry run: {DRY_RUN}")
    logger.info("")

    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received")
    finally:
        save_results_on_exit()


# =============================================================================
# Phase 3: CSV 기반 비교실험
# =============================================================================

class SimBinPackScheduler(BinPackScheduler):
    """시뮬레이션용 BinPackScheduler (dry_run, event-driven)"""

    def __init__(self, jobs, nodes, consolidation_enabled=True, ilp_replan_enabled=True,
                 placement_strategy='best_fit'):
        # EndpointScheduler.__init__는 빈 job 리스트로 초기화
        super().__init__(
            nodes=nodes,
            node_selection_strategy="bin_pack",
            speed=1,
            dry_run=True,
        )
        self.consolidation_enabled = consolidation_enabled
        self.ilp_replan_enabled = ilp_replan_enabled
        self.placement_strategy = placement_strategy  # 'best_fit', 'least_allocated', 'most_allocated'
        self.start_time = jobs[0].submit_time if jobs else 0.0

        # Job arrival events 로드
        for job in jobs:
            arrival_event = Event(
                time=job.submit_time,
                event_type='arrival',
                job=job
            )
            heapq.heappush(self.event_queue, arrival_event)

    def handle_completion(self, job):
        """Consolidation을 조건부로 실행"""
        if self.consolidation_enabled:
            # 부모(BinPackScheduler)의 handle_completion 사용 (consolidation 포함)
            super().handle_completion(job)
        else:
            # consolidation 없는 기본 완료 처리
            completed_node = None
            if job.job_id in self.job_to_node:
                completed_node = self.job_to_node[job.job_id]

            if job.job_id in self.running_jobs:
                if job.job_id in self.job_states:
                    self.update_job_state_on_completion(job.job_id, self.current_time)
                if completed_node:
                    completed_node.deallocate_job(job)
                    del self.job_to_node[job.job_id]
                del self.running_jobs[job.job_id]

            # Snapshot completion_time in job_states
            if job.job_id in self.job_states and self.job_states[job.job_id].completion_time is None:
                self.job_states[job.job_id].completion_time = self.current_time

            # Prevent duplicate completion entries
            if job.job_id not in {j.job_id for j in self.completed_jobs}:
                self.completed_jobs.append(job)
                self.total_jct += job.jct()
                self.total_wait_time += job.wait_time()

            time_str = self.format_time(self.current_time)
            if completed_node:
                print(f" [{time_str}] {job.job_id} completed on {completed_node.name} ({job.req}g freed)")

    def handle_arrival(self, job):
        """ILP replan을 조건부로 비활성화!!!"""
        if not self.ilp_replan_enabled:
            # ILP 비활성화: HP-AI deploy도 일반 best-fit으로 처리
            original_type = job.job_type
            if job.job_type in ("HP", "HP-deploy"):
                job.job_type = "deploy"
            super().handle_arrival(job)
            if job.job_type == "deploy" and original_type in ("HP", "HP-deploy"):
                job.job_type = original_type
        else:
            super().handle_arrival(job)

    def deploy_job_to_node(self, job, node, target_slice):
        """dry_run 시뮬레이션: 상태 업데이트만 수행"""
        # deploy 타입만 처리 (시뮬레이션에서는 migration/scale-out 없음)
        if job.job_type in ("deploy", "HP", "HP-deploy"):
            target_slice['used'] = job.req
            target_slice['jobs'].append(job.job_id)
            job.allocated_size = target_slice['size']

            job.start_time = self.current_time

            if job.job_id in self.job_states:
                remaining_duration = self.job_states[job.job_id].remaining_duration
                self.job_states[job.job_id].actual_start_time = self.current_time
            else:
                remaining_duration = job.duration
            job.end_time = self.current_time + (remaining_duration * 60)

            self.running_jobs[job.job_id] = job
            self.job_to_node[job.job_id] = node

            # job_type에 따라 올바른 running list에 추가
            # (deallocate_job이 job_type 기반으로 제거하므로 일치시켜야 함)
            if job.job_type in ("HP", "HP-scale-out"):
                node.hp_running_jobs.append(job)
            else:
                node.spot_running_jobs.append(job)

            completion_event = Event(
                time=job.end_time,
                event_type='completion',
                job=job
            )
            heapq.heappush(self.event_queue, completion_event)
        else:
            # 나머지 job_type은 부모 로직 (dry_run이므로 kubectl 호출 안됨)
            super().deploy_job_to_node(job, node, target_slice)

    def run(self, max_time=None):
        """이벤트 기반 시뮬레이션 루프 (wall clock 없이 event time 기준)"""
        self.wall_start = time.time()

        while self.event_queue:
            event = heapq.heappop(self.event_queue)

            if event.cancelled:
                continue

            self.current_time = event.time

            if max_time is not None and self.current_time > max_time:
                break

            if event.event_type == 'arrival':
                self.handle_arrival(event.job)
            elif event.event_type == 'completion':
                self.handle_completion(event.job)
            elif event.event_type == 'slot_timeout':
                self.handle_slot_timeout(event)

        return self.get_metrics()


def load_jobs_from_csv(csv_path, start_hour, start_min, end_hour, end_min,
                       exclude_multi_gpu=True, exclude_a100=True):
    """CSV에서 job 로드 (test_baseline.py 패턴 재사용)"""
    df = pd.read_csv(csv_path)

    # 시간대 필터링
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

        duration_min = row['duration'] / 60.0
        job_type = row['job_type']

        job = Job(
            job_id=f"{job_type.lower()}-{row['job_name']}",
            name=f"workload-{row['organization']}",
            job_type=job_type,
            req=int(row['mig_size']),
            duration=duration_min,
            submit_time=submit_time
        )
        jobs.append(job)

    jobs.sort(key=lambda j: j.submit_time)
    return jobs


def run_comparison(csv_path, start_hour, start_min, end_hour, end_min, max_time=1800):
    """3가지 전략 비교실험"""
    import copy

    print("=" * 80)
    print("COMPARISON EXPERIMENT: Baseline vs BestFit+Consol vs BestFit+Consol+ILP")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Time: {start_hour:02d}:{start_min:02d} ~ {end_hour:02d}:{end_min:02d}")
    print(f"Max simulation time: {max_time}s")
    print()

    # Job 로드
    jobs = load_jobs_from_csv(csv_path, start_hour, start_min, end_hour, end_min)
    if not jobs:
        print("No jobs found!")
        return

    hp_jobs = [j for j in jobs if j.job_type == "HP"]
    spot_jobs = [j for j in jobs if j.job_type == "Spot"]
    mig_sizes = {}
    for j in jobs:
        mig_sizes[j.req] = mig_sizes.get(j.req, 0) + 1

    print(f"Total: {len(jobs)} jobs (HP={len(hp_jobs)}, Spot={len(spot_jobs)})")
    print(f"MIG sizes: {mig_sizes}")
    print()

    results = {}

    # ===== Strategy 1: Baseline (고정 profile 3211 × 3, HP 구분 없음) =====
    print("\n" + "=" * 60)
    print("Strategy 1: BASELINE (Fixed 3211 × 3, No HP Priority)")
    print("=" * 60)

    baseline_jobs = copy.deepcopy(jobs)
    for j in baseline_jobs:
        j.job_type = "deploy"  # 모든 job을 deploy로 (BinPack best-fit 사용)

    baseline_nodes = [
        GPUNode(name="skt-6gtb-ars", mig_profile="3211", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="3211", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="3211", gpu_index=1),
    ]

    baseline_sched = SimBinPackScheduler(
        jobs=baseline_jobs,
        nodes=baseline_nodes,
        consolidation_enabled=False,
        ilp_replan_enabled=False,
    )
    results['baseline'] = baseline_sched.run(max_time=max_time)

    # ===== Strategy 2: BestFit + Consolidation (ILP 없음) =====
    print("\n" + "=" * 60)
    print("Strategy 2: BESTFIT + CONSOLIDATION (Dynamic Profile, No ILP)")
    print("=" * 60)

    bestfit_jobs = copy.deepcopy(jobs)
    for j in bestfit_jobs:
        j.job_type = "deploy"  # BinPackScheduler는 deploy 타입 사용

    bestfit_nodes = [
        GPUNode(name="skt-6gtb-ars", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=1),
    ]

    bestfit_sched = SimBinPackScheduler(
        jobs=bestfit_jobs,
        nodes=bestfit_nodes,
        consolidation_enabled=True,
        ilp_replan_enabled=False,
    )
    results['bestfit_consol'] = bestfit_sched.run(max_time=max_time)

    # ===== Strategy 3: BestFit + Consolidation + ILP =====
    print("\n" + "=" * 60)
    print("Strategy 3: BESTFIT + CONSOLIDATION + ILP (Full)")
    print("=" * 60)

    full_jobs = copy.deepcopy(jobs)
    for j in full_jobs:
        if j.job_type == "Spot":
            j.job_type = "deploy"
        # HP jobs는 job_type="HP" 유지 (ILP preemption 트리거)

    full_nodes = [
        GPUNode(name="skt-6gtb-ars", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=1),
    ]

    full_sched = SimBinPackScheduler(
        jobs=full_jobs,
        nodes=full_nodes,
        consolidation_enabled=True,
        ilp_replan_enabled=True,
    )
    results['bestfit_consol_ilp'] = full_sched.run(max_time=max_time)

    # ===== 비교표 출력 =====
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    header = f"{'Metric':<25} {'Baseline':>15} {'BF+Consol':>15} {'BF+Consol+ILP':>15}"
    print(header)
    print("-" * len(header))

    metrics_keys = [
        ('total_jobs', 'Total Jobs'),
        ('completed_jobs', 'Completed Jobs'),
        ('started_jobs', 'Started Jobs'),
        ('avg_wait_min', 'Avg Wait (min)'),
        ('avg_jct_min', 'Avg JCT (min)'),
        ('avg_progress', 'Avg Progress (%)'),
    ]

    for key, label in metrics_keys:
        vals = []
        for strategy in ['baseline', 'bestfit_consol', 'bestfit_consol_ilp']:
            v = results[strategy].get(key, 0)
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(f"{label:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print()

    # JSON 결과 저장
    result_path = os.path.join(RESULT_DIR, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    comparison_result = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'csv_path': csv_path,
            'start_hour': start_hour, 'start_min': start_min,
            'end_hour': end_hour, 'end_min': end_min,
            'max_time': max_time,
        },
        'results': results,
    }
    with open(result_path, 'w') as f:
        json.dump(comparison_result, f, indent=2)
    print(f"Results saved to: {result_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BinPack Scheduler")
    parser.add_argument('--mode', choices=['endpoint', 'compare'], default='endpoint',
                        help='endpoint: HTTP 서버 모드, compare: CSV 비교실험 모드')
    parser.add_argument('--csv', default='/home/skt6g/AI-RAN/KubeSMO/data/single_gpu_a100_singleworker_day79_with_mig_capped.csv')
    parser.add_argument('--start-hour', type=int, default=10)
    parser.add_argument('--start-min', type=int, default=30)
    parser.add_argument('--end-hour', type=int, default=11)
    parser.add_argument('--end-min', type=int, default=0)
    parser.add_argument('--max-time', type=int, default=1800)
    args = parser.parse_args()

    if args.mode == 'compare':
        run_comparison(
            csv_path=args.csv,
            start_hour=args.start_hour,
            start_min=args.start_min,
            end_hour=args.end_hour,
            end_min=args.end_min,
            max_time=args.max_time,
        )
    else:
        main_endpoint()


if __name__ == "__main__":
    main()
