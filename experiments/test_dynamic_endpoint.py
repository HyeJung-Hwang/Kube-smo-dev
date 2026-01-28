#!/usr/bin/env python3
"""
Dynamic Scheduler with HTTP Endpoint

기존 test_dynamic.py의 시간 기반 이벤트 발송 대신,
HTTP 엔드포인트로 job을 받아서 스케줄러에 전달합니다.

## 사용법
    # 서버 시작
    python test_dynamic_endpoint.py

    # Job 제출 (HP)
    curl -X POST http://localhost:8000/submit \
      -H "Content-Type: application/json" \
      -d '{"job_id": "hp-job-001", "job_type": "HP", "req": 2, "duration": 5.0}'

    # Job 제출 (Spot)
    curl -X POST http://localhost:8000/submit \
      -H "Content-Type: application/json" \
      -d '{"job_id": "spot-job-001", "job_type": "Spot", "req": 1, "duration": 10.0}'

    # 상태 조회
    curl http://localhost:8000/status

    # 메트릭 조회
    curl http://localhost:8000/metrics

    # 스케줄러 종료
    curl -X POST http://localhost:8000/stop
"""
import sys
sys.path.insert(0, '/home/skt6g/AI-RAN/KubeSMO/experiments')

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import cloudpickle

from job import Job, Event, GPUNode
from realtime_slot_scheduler import RealTimeSlotScheduler


# ==================== 로깅 설정 ====================
class TeeOutput:
    """stdout을 콘솔과 파일 모두에 출력"""
    def __init__(self, log_file, original_stdout):
        self.log_file = log_file
        self.original_stdout = original_stdout

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        if not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.original_stdout.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def isatty(self):
        return self.original_stdout.isatty()

    def fileno(self):
        return self.original_stdout.fileno()


# 타임스탬프 생성 (실행 시점)
_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_DIR = '/home/skt6g/AI-RAN/KubeSMO/experiments'
LOG_PATH = f'{RESULT_DIR}/endpoint_{_TIMESTAMP}.log'

# 전역 로그 파일 핸들
_log_file = None


def setup_logging(log_path):
    """파일 + 콘솔 로깅 설정"""
    global _log_file

    # 로그 파일 열기
    _log_file = open(log_path, mode='w', encoding='utf-8')

    # stdout을 Tee로 교체 (print 출력도 파일에 기록)
    sys.stdout = TeeOutput(_log_file, sys.__stdout__)

    # 로거 설정
    logger = logging.getLogger('dynamic_endpoint')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

    # 파일 핸들러
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 부모 로거로 전파 방지 (중복 출력 방지)
    logger.propagate = False

    return logger


logger = logging.getLogger('dynamic_endpoint')


# ==================== 설정 ====================
HOST = "0.0.0.0"
PORT = 8000
SPEED = 1  # 시뮬레이션 속도
DRY_RUN = True # True: 실제 배포 안함, False: 실제 배포

# RAN Prediction Model
MODEL_PATH = "/home/skt6g/AI-RAN/KubeSMO/backend/control_plane/models/enhanced_lgb_model.pkl"
CELL_DATA_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/cell_5678_data.csv"
AI_INFERENCE_DATA_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/Azure_0518_inference.csv"

# Global model variable
_MODEL = None


def load_ran_model(model_path: str):
    """Load RAN prediction model"""
    global _MODEL
    from pathlib import Path

    # Add ml_models path for pickle to find 'model' module
    ml_models_path = "/home/skt6g/AI-RAN/KubeSMO/backend/control_plane/ml_models"
    if ml_models_path not in sys.path:
        sys.path.insert(0, ml_models_path)

    p = Path(model_path)
    if not p.exists():
        logger.warning(f"Model not found: {p}")
        return None
    with p.open("rb") as f:
        _MODEL = cloudpickle.load(f)
    logger.info(f"RAN prediction model loaded from {model_path}")
    return _MODEL


# ==================== 전역 변수 ====================
scheduler: Optional[RealTimeSlotScheduler] = None
scheduler_thread: Optional[threading.Thread] = None
scheduler_lock = threading.Lock()
stop_event = threading.Event()


# ==================== Pydantic 모델 ====================
class JobRequest(BaseModel):
    """Job 제출 요청"""
    job_id: str
    name: str = "workload"
    job_type: str = "Spot"  # HP, Spot, HP-scale-out, HP-scale-in, HP-scale-up, HP-scale-down
    workload_type: str = "AI"  # AI, RAN (HP job에서만 사용)
    req: int = 1  # MIG size (1, 2, 3, 4, 7)
    duration: float = 10.0  # 분 단위
    target_job_id: Optional[str] = None  # scale-out/in/up/down용


class JobResponse(BaseModel):
    """Job 제출 응답"""
    status: str
    job_id: str
    submit_time: float
    message: str = ""


class FeaturesInput(BaseModel):
    """RAN Prediction 입력"""
    features: List[Optional[float]]


# ==================== RAN Prediction Helper Functions ====================
def mig_scale_decision_ran(prediction_result: dict, b=1.5, a=8.5) -> int:
    """MIG scale decision based on prediction"""
    mean_pred = prediction_result.get('mean_pred', None)[0]
    scaled_num_of_cell = b * mean_pred + a
    if 8 <= scaled_num_of_cell < 9:
        return 1
    elif 9 <= scaled_num_of_cell < 10:
        return 2
    else:
        return 2  # edge case -> default to 2


def convert_numpy_arrays(prediction: dict) -> dict:
    """Convert numpy arrays to lists for JSON serialization"""
    result = {}

    mean_pred = prediction.get('mean_pred')
    if mean_pred is not None:
        result['mean_pred'] = mean_pred.tolist() if hasattr(mean_pred, 'tolist') else mean_pred
    else:
        result['mean_pred'] = None

    std_pred = prediction.get('std_pred')
    if std_pred is not None:
        result['std_pred'] = std_pred.tolist() if hasattr(std_pred, 'tolist') else std_pred
    else:
        result['std_pred'] = None

    quantiles = prediction.get('quantiles')
    if quantiles is not None and isinstance(quantiles, dict):
        quantiles_converted = {}
        keys = ['q0.01', 'q0.03', 'q0.05', 'q0.10', 'q0.25', 'q0.50',
                'q0.75', 'q0.90', 'q0.95', 'q0.97', 'q0.99']
        for k in keys:
            v = quantiles.get(k)
            if v is not None:
                quantiles_converted[k] = v.tolist() if hasattr(v, 'tolist') else v
        result['quantiles'] = quantiles_converted
    else:
        result['quantiles'] = None

    return result


# ==================== 노드 설정 ====================
def create_nodes():
    """GPU 노드 생성"""
    return [
        GPUNode(name="skt-6gtb-ars", mig_profile="3211", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="3211", gpu_index=0),
        GPUNode(name="sys-221he-tnr", mig_profile="3211", gpu_index=1),
    ]


# ==================== 스케줄러 루프 ====================
class EndpointScheduler(RealTimeSlotScheduler):
    """HTTP 엔드포인트용 스케줄러 (무한 루프 지원)"""

    def __init__(self, nodes, node_selection_strategy, speed, dry_run=True):
        # 빈 job 리스트로 초기화
        super().__init__(
            jobs=[],
            nodes=nodes,
            node_selection_strategy=node_selection_strategy,
            speed=speed,
            dry_run=dry_run
        )
        # 빈 리스트로 시작하므로 start_time을 0으로 설정
        self.start_time = 0.0

    def add_job(self, job: Job):
        """외부에서 job 추가 (thread-safe)"""
        arrival_event = Event(
            time=self.current_time,
            event_type='arrival',
            job=job
        )
        heapq.heappush(self.event_queue, arrival_event)
        logger.info(f"Job added to queue: {job.job_id} ({job.job_type}, {job.req}g, {job.duration}min)")

    def run_forever(self, stop_event: threading.Event, poll_interval: float = 0.1):
        """
        무한 루프로 스케줄러 실행 (event_queue가 비어도 계속 대기)

        Args:
            stop_event: 종료 신호
            poll_interval: 이벤트 폴링 간격 (초)
        """
        self.wall_start = time.time()
        logger.info("Scheduler started (waiting for jobs...)")

        while not stop_event.is_set():
            # 현재 시간 업데이트 (wall clock 기준)
            wall_elapsed = time.time() - self.wall_start
            self.current_time = wall_elapsed * self.speed

            # 이벤트 큐에서 처리할 이벤트가 있는지 확인
            while self.event_queue:
                # peek (pop하지 않고 확인)
                next_event = self.event_queue[0]

                # 아직 처리할 시간이 안 됐으면 대기
                if next_event.time > self.current_time:
                    break

                # 이벤트 처리
                event = heapq.heappop(self.event_queue)

                # Cancelled event는 skip
                if event.cancelled:
                    continue

                self.current_time = max(self.current_time, event.time)

                if event.event_type == 'arrival':
                    self.handle_arrival(event.job)
                elif event.event_type == 'completion':
                    self.handle_completion(event.job)
                elif event.event_type == 'slot_timeout':
                    self.handle_slot_timeout(event)

            # 잠시 대기 후 다음 폴링
            time.sleep(poll_interval)

        logger.info("Scheduler stopped")
        return self.get_metrics()


def run_scheduler():
    """스케줄러를 별도 스레드에서 실행"""
    global scheduler

    nodes = create_nodes()
    scheduler = EndpointScheduler(
        nodes=nodes,
        node_selection_strategy="least_loaded",
        speed=SPEED,
        dry_run=DRY_RUN
    )

    logger.info(f"Nodes: {len(nodes)}")
    for node in nodes:
        logger.info(f"  {node.name} GPU {node.gpu_index}: {node.mig_profile}")

    # 무한 루프 실행 (stop_event가 set되면 종료)
    scheduler.run_forever(stop_event)


# ==================== FastAPI 앱 ====================
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


# ==================== API 엔드포인트 ====================
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

    # Job 객체 생성
    job = Job(
        job_id=job_req.job_id,
        name=job_req.name,
        job_type=job_req.job_type,
        workload_type=job_req.workload_type,
        req=job_req.req,
        duration=job_req.duration,
        submit_time=scheduler.current_time
    )

    # scale-out/in용 target_job_id 설정
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


@app.get("/status")
async def get_status():
    """스케줄러 상태 조회"""
    if scheduler is None:
        return {"status": "not_started"}

    # 노드별 상태
    node_status = []
    for node in scheduler.nodes:
        node_status.append({
            "name": node.name,
            "gpu_index": node.gpu_index,
            "mig_profile": node.mig_profile,
            "hp_running": [j.job_id for j in node.hp_running_jobs],
            "spot_running": [j.job_id for j in node.spot_running_jobs],
            "hp_waiting": [j.job_id for j in node.hp_waiting_queue],
            "spot_waiting": [j.job_id for j in node.spot_waiting_queue],
        })

    return {
        "status": "running",
        "current_time": round(scheduler.current_time, 2),
        "current_time_formatted": scheduler.format_time(scheduler.current_time),
        "running_jobs": list(scheduler.running_jobs.keys()),
        "completed_jobs": len(scheduler.completed_jobs),
        "queue_size": len(scheduler.event_queue),
        "nodes": node_status
    }


@app.get("/jobs")
async def get_jobs():
    """모든 job 상태 조회"""
    if scheduler is None:
        return {"error": "Scheduler not started"}

    jobs = []
    for job_id, state in scheduler.job_states.items():
        jobs.append({
            "job_id": job_id,
            "job_type": state.job_type,
            "original_duration": state.original_duration,
            "remaining_duration": state.remaining_duration,
            "submit_time": state.submit_time,
            "actual_start_time": state.actual_start_time,
            "completion_time": state.completion_time,
            "times_preempted": state.times_preempted,
            "total_run_time": state.total_run_time,
            "status": "completed" if state.completion_time else (
                "running" if state.actual_start_time else "waiting"
            )
        })

    return {"jobs": jobs}


@app.get("/metrics")
async def get_metrics():
    """메트릭 조회"""
    if scheduler is None:
        return {"error": "Scheduler not started"}

    return scheduler.get_metrics()


@app.get("/paper-metrics")
async def get_paper_metrics():
    """논문용 메트릭 조회"""
    if scheduler is None:
        return {"error": "Scheduler not started"}

    return scheduler.get_paper_metrics_summary()


@app.post("/stop")
async def stop_scheduler():
    """스케줄러 종료"""
    stop_event.set()
    return {"status": "stopping", "message": "Scheduler is stopping..."}


@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Dynamic Scheduler API",
        "version": "1.0.0",
        "endpoints": {
            "POST /submit": "Job 제출",
            "GET /status": "스케줄러 상태 조회",
            "GET /jobs": "모든 job 상태 조회",
            "GET /metrics": "메트릭 조회",
            "GET /paper-metrics": "논문용 메트릭 조회",
            "POST /stop": "스케줄러 종료",
            "POST /ran/predict": "RAN throughput 예측",
            "GET /ran/cell-data": "RAN cell 데이터 조회",
            "GET /ai/inference-data": "AI inference 데이터 조회",
        }
    }


# ==================== RAN Prediction Endpoints ====================
@app.post("/ran/predict")
async def predict_ran(input: FeaturesInput):
    """
    RAN Throughput 예측

    Args:
        input: features 리스트 (12개 feature)

    Returns:
        prediction: 예측 결과
        mig_scale_decision: MIG scale 결정 (1 또는 2 cells)
    """
    global _MODEL

    if _MODEL is None:
        raise HTTPException(status_code=500, detail="RAN model not loaded")

    try:
        features_array = np.array(input.features)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        prediction = _MODEL.predict(features_array)
        prediction_jsonable = convert_numpy_arrays(prediction)
        decision_result = mig_scale_decision_ran(prediction_jsonable)

        return {
            "prediction": prediction_jsonable,
            "mig_scale_decision": decision_result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ran/cell-data")
async def get_cell_data(
    current_time: str = "2023-08-17 10:10:00",
    future_time: str = "2023-08-17 10:25:00"
):
    """
    현재 시점 기준 이전 10개 데이터와 미래 예측 타겟 데이터를 반환

    Args:
        current_time: 현재 시점 (예: "2023-08-17 10:10:00")
        future_time: 예측할 미래 시점 (예: "2023-08-17 10:25:00")

    Returns:
        historical: 현재 이전 10개 데이터
        current: 현재 시점 데이터
        future: 예측 대상 시점 데이터 (features 포함)
    """
    try:
        if not os.path.exists(CELL_DATA_PATH):
            raise HTTPException(404, f"CSV file not found: {CELL_DATA_PATH}")

        df = pd.read_csv(CELL_DATA_PATH)
        df['Time'] = pd.to_datetime(df['Time'])

        current_dt = pd.to_datetime(current_time)
        future_dt = pd.to_datetime(future_time)

        feature_columns = ['lag_1', 'ma_1', 'std_1', 'ema_1',
                          'lag_2', 'ma_2', 'std_2', 'ema_2',
                          'lag_3', 'ma_3', 'std_3', 'ema_3']

        # 1. 현재 시점 이전 10개 데이터
        before_current = df[df['Time'] < current_dt].sort_values('Time', ascending=False)
        last_10 = before_current.head(10).sort_values('Time', ascending=True)

        historical_data = []
        for _, row in last_10.iterrows():
            historical_data.append({
                "time": row['Time'].isoformat(),
                "throughput": float(row['MAC Throughput (Kbps) DL-Normal']) if pd.notna(row['MAC Throughput (Kbps) DL-Normal']) else None
            })

        # 2. 현재 시점 데이터
        current_row = df[df['Time'] == current_dt]
        if current_row.empty:
            current_row = df.iloc[(df['Time'] - current_dt).abs().argsort()[:1]]

        current_data = None
        if not current_row.empty:
            row = current_row.iloc[0]
            current_data = {
                "time": row['Time'].isoformat(),
                "throughput": float(row['MAC Throughput (Kbps) DL-Normal']) if pd.notna(row['MAC Throughput (Kbps) DL-Normal']) else None
            }

        # 3. 미래 시점 데이터 (예측 대상)
        future_row = df[df['Time'] == future_dt]
        if future_row.empty:
            future_row = df.iloc[(df['Time'] - future_dt).abs().argsort()[:1]]

        future_data = None
        if not future_row.empty:
            row = future_row.iloc[0]
            features = []
            for col in feature_columns:
                val = row[col]
                if pd.isna(val):
                    features.append(None)
                else:
                    features.append(float(val))

            future_data = {
                "time": row['Time'].isoformat(),
                "throughput": float(row['MAC Throughput (Kbps) DL-Normal']) if pd.notna(row['MAC Throughput (Kbps) DL-Normal']) else None,
                "features": features
            }

        return {
            "historical": historical_data,
            "current": current_data,
            "future": future_data
        }

    except Exception as e:
        raise HTTPException(500, f"Error reading cell data: {str(e)}")


# ==================== AI Inference Data Endpoint ====================
@app.get("/ai/inference-data")
async def get_ai_inference_data():
    """
    AI Inference 데이터 반환 (Azure trace)

    Returns:
        list of {timestamp, actual, predicted}
    """
    try:
        if not os.path.exists(AI_INFERENCE_DATA_PATH):
            raise HTTPException(404, f"CSV file not found: {AI_INFERENCE_DATA_PATH}")

        df = pd.read_csv(AI_INFERENCE_DATA_PATH)

        data = []
        for _, row in df.iterrows():
            timestamp = row['TIMESTAMP']
            # Remove timezone info for simplicity
            if '+' in str(timestamp):
                timestamp = str(timestamp).split('+')[0]

            data.append({
                "timestamp": timestamp,
                "actual": round(float(row['y_test_scaled']), 3) if pd.notna(row['y_test_scaled']) else None,
                "predicted": round(float(row['y_pred_scaled']), 3) if pd.notna(row['y_pred_scaled']) else None
            })

        return data

    except Exception as e:
        raise HTTPException(500, f"Error reading AI inference data: {str(e)}")


# ==================== 메인 ====================
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
