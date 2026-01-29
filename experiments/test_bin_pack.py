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

from job import Job, Event, GPUNode
from realtime_slot_scheduler import JobState, RealTimeSlotScheduler


# 전역 변수 
logger = logging.getLogger('dynamic_endpoint')
HOST = "0.0.0.0"
PORT = 8000
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



def create_initial_nodes():
    return [
        GPUNode(name="skt-6gtb-ars", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="34", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="33", gpu_index=1),
    ]

class BinPackScheduler(EndpointScheduler):
    # Binpacking 로직 포함해야함 
    def handle_job_arrival(self, job):
        # 메트릭 측정등으 위함
        self.job_states[job.job_id] = JobState(
            job_id=job.job_id,
            original_duration=job.duration,
            remaining_duration=job.duration,
            job_type=job.job_type,
            workload_type=getattr(job, 'workload_type', 'AI'),
            submit_time=self.current_time
        )
        # 배포할 노드 , GPU, 인스턴스 결정
        N = job.req
        # 1. 배포 대상이 되는 노드 리스트 결정 (Job Type에 따라 다름)
        if job.job_type == "migration":
            deploy_target_nodes = [n for n in self.nodes if n is not job.source_node]
        else:
            deploy_target_nodes = self.nodes
        # 2. 노드&GPU 조합에서 빈공간 찾기
        # candidate_list: [(node, [{'id':0,'size':3,'used':0,'jobs':[]}, ...])]
        candidate_list = []
        for node in deploy_target_nodes:
            free_slices = node.get_free_slice_list()
            if free_slices:
                candidate_list.append((node, free_slices))

        # 3. candidate 중에서 N 이 free slice size에 있는 경우 중에 총 빈공간이 가장 작은 노드 선택
        best_node = None
        best_slice = None
        candidate_with_exact_fit = [
            (node, slices) for node, slices in candidate_list
            if any(s['size'] == N for s in slices)
        ]
        if candidate_with_exact_fit:
            best_node, _ = min(
                candidate_with_exact_fit,
                key=lambda x: sum(s['size'] for s in x[1]) - N
            )
            best_slice = next(s for s in best_node.get_free_slice_list() if s['size'] == N)

        else:
            candidate_with_larger_fit = [
                (node, slices) for node, slices in candidate_list
                if any(s['size'] >= N for s in slices)
            ]
            if candidate_with_larger_fit:
                best_node, _ = min(
                    candidate_with_larger_fit,
                    key=lambda x: min(s['size'] - N for s in x[1] if s['size'] >= N)
                )
                best_slice = min(
                    (s for s in best_node.get_free_slice_list() if s['size'] >= N),
                    key=lambda s: s['size'] - N
                )

        # 4. best_node에 job 배포/ best node 없는 경우 다시 큐잉
        if best_node is None: # 없을 떄
            logger.info(f"No suitable node found for job {job.job_id} ({job.req}g). Re-queuing.")
            arrival_event = Event(
                time=self.current_time + 1,
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

            completion_event = Event(
                time=job.end_time,
                event_type='completion',
                job=job
            )
            heapq.heappush(self.event_queue, completion_event)

            mig_uuid = deploy.deploy_job(
                    job=job,
                    node_name=node.name,
                    gpu_g=target_slice['size'],
                    gpu_index=node.gpu_index,
                    slice_index=target_slice['id'],
            )
            if mig_uuid:
                job.mig_uuid = mig_uuid
        elif job.job_type == "migration":
            # 같은type 하위에서는 job_tyepe에 따라 분기
            mig_uuid = deploy.deploy_migration(
                        job.job_id,
                        node.name   ,
                        gpu_g=target_slice['size'],
                        gpu_index=node.gpu_index,
                        slice_index=target_slice['id'],
                        avoid_uuids= None
            ) # Migration 처리 후에 bin)pack 해야함. + completion. 사용할 로직은 빈자리가 생긴 사이즈와 노드를 찾고, 해당 사이즈의 다른 고세 배포된 워크로드들에 대해 migration 전후 노드 score mgain 게산해서 가장 큰애 옮기기
            if mig_uuid:
                job.mig_uuid = mig_uuid



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
    - job_type: HP, Spot, HP-scale-out, HP-scale-in
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
    job = Job(
        job_id=job_req.job_id,
        name=job_req.name,
        job_type=job_req.job_type,
        workload_type=job_req.workload_type,
        req=job_req.req,
        duration=job_req.duration,
        submit_time=scheduler.current_time
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

def main():
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


if __name__ == "__main__":
    main()
