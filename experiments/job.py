from __future__ import annotations
# 필요한 라이브러리 import
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
import numpy as np

print("Libraries loaded successfully")

@dataclass
class Job:
    """Job 정의"""
    job_id: str
    name: str
    job_type: str  # 논문 HP, Spot, HP-scale-out, HP-scale-in, HP-scale-up, HP-scale-down // # 시연용 deploy, scale-out , scale-in, migration
    req: int  # GPU 크기 (1, 2, 3, 7) 또는 scale delta (+1, -1)
    duration: float  # 실행 시간 (분) - scale-out/in의 경우 무시됨
    submit_time: float  # 제출 시간 (초)

    # Workload 타입 (HP job용)
    workload_type: str = "AI"  # AI, RAN 

    # Scale 관련 (scale-out/scale-in용)
    target_job_id: Optional[str] = None  # scale 대상이 되는 원본 job ID

    # 런타임 정보
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # MIG 관련
    mig_uuid: Optional[str] = None  # 배포된 MIG 인스턴스 UUID
    allocated_size: Optional[int] = None  # 실제 할당된 슬라이스 크기 (req와 다를 수 있음)

    ## Binpacking Schefuler에 만 필요
    source_node: Optional[GPUNode] = None
    # Migration POd이 경우 필요
    src_pod_name: Optional[str] = None

    def jct(self) -> float:
        """Job Completion Time (초)"""
        if self.end_time is None or self.submit_time is None:
            return 0.0
        return self.end_time - self.submit_time

    def wait_time(self) -> float:
        """대기 시간 (초)"""
        if self.start_time is None or self.submit_time is None:
            return 0.0
        return self.start_time - self.submit_time

@dataclass
class RANJob(Job):
    launch_pattern: str = "F08 1C 59"
    cell_group_num: int = 1


@dataclass
class Event:
    """시뮬레이션 이벤트"""
    time: float
    event_type: str  # 'arrival', 'completion', 'slot_timeout'
    job: Optional['Job'] = None  # Optional (slot_timeout에는 job 없음)

    # Slot timeout용 추가 필드
    node: Optional['GPUNode'] = None  # 어느 노드의 timeout인지
    slot_idx: Optional[int] = None    # 어느 slot의 timeout인지
    plan: Optional[Dict] = None       # 어느 plan의 timeout인지 (검증용)

    # Preemption용 필드
    cancelled: bool = False  # Event가 취소되었는지 (preempt 시 completion event 취소)

    def __lt__(self, other):
        return self.time < other.time

@dataclass
class GPUNode:
    """GPU 노드 - MIG 슬라이스 기반 할당 (Multi-GPU 지원)"""
    name: str
    # allocated_list: List[int]
    mig_profile: str
    gpu_index: int = 0  # GPU index for nvidia-smi -i <index>
    total_capacity: int = field(init=False)
    slices: List[Dict] = field(init=False)  # MIG 슬라이스 목록
    current_plan: Optional[Dict] = None  # 현재 MIG 재구성 계획
    current_slot: Optional[int] = None  # 현재 실행 중인 slot 번호
    plan_start_time: Optional[float] = None  # Plan 시작 시간 (시뮬레이션 시간)
    hp_waiting_queue: List[Job] = field(default_factory=list)  # HP job 대기 큐
    spot_waiting_queue: List[Job] = field(default_factory=list)  # Spot
    hp_running_jobs: List[Job] = field(default_factory=list)  # 실행 중인 HP job
    spot_running_jobs: List[Job] = field(default_factory=list) # 실행

    def __post_init__(self):
        self.total_capacity = sum(int(c) for c in self.mig_profile)
        self._init_slices()

    def _init_slices(self):
        """MIG 슬라이스 초기화"""
        self.slices = []
        for i, size_char in enumerate(self.mig_profile):
            size = int(size_char)
            self.slices.append({
                'id': i,
                'size': size,
                'used': 0,
                'jobs': []  # 할당된 job ID 리스트
            })
    def get_free_slice_list(self) -> list:
        # allcoated_list [ 3,0] <-> mig_profile_list [3,4]
        free_size_list = [slice for slice in self.slices if slice["used"] ==0 ]
        return free_size_list # [  {'id': 0, 'size': 3, 'used': 0, 'jobs': []},  {'id': 0, 'size': 3, 'used': 0, 'jobs': []} ]

    def get_all_slice_list(self) -> list:
        return self.slices

    def get_capacity(self) -> int:
        """총 용량 반환"""
        return self.total_capacity

    def get_available_capacity(self) -> int:
        """실제 할당 가능한 용량 계산 (MIG 슬라이스 제약 고려)"""
        # 빈 슬라이스들의 크기만 합산 (각 슬라이스는 1개 job만 할당 가능)
        return sum(s['size'] for s in self.slices if len(s['jobs']) == 0)

    def get_free_capacity(self) -> int:
        """빈 슬라이스들의 총 용량 반환 (get_available_capacity와 동일)"""
        return self.get_available_capacity()

    def can_allocate(self, job_size: int) -> bool:
        """job을 할당할 수 있는 빈 슬라이스가 있는지 확인"""
        # 빈 슬라이스 중에서 job_size 이상인 것이 있는지 확인
        for s in self.slices:
            if len(s['jobs']) == 0 and s['size'] >= job_size:
                return True
        return False

    def allocate_job(self, job):
        """
        job을 MIG 슬라이스에 할당 (1 슬라이스 = 1 job만)

        Returns:
            int: 할당된 slice index (0, 1, 2, ...)
            None: 할당 실패
        """
        # Best-fit: 낭비가 가장 적은 빈 슬라이스에 할당
        best_slice = None
        best_slice_index = None
        min_waste = float('inf')
        job_id = job.job_id
        job_size = job.req
        for i, s in enumerate(self.slices):
            # 슬라이스가 비어있고, job을 수용할 수 있는 경우만
            if len(s['jobs']) == 0 and s['size'] >= job_size:
                waste = s['size'] - job_size
                if waste < min_waste:
                    min_waste = waste
                    best_slice = s
                    best_slice_index = i

        if best_slice:
            best_slice['used'] = job_size  # 전체 슬라이스가 점유됨
            best_slice['jobs'].append(job_id)
            # 실제 할당된 슬라이스 크기 저장 (req와 다를 수 있음)
            job.allocated_size = best_slice['size']
            if job.job_type in ("HP", "HP-scale-out"):
                self.hp_running_jobs.append(job)
                self.hp_waiting_queue.remove(job)
            else:
                self.spot_running_jobs.append(job)
                self.spot_waiting_queue.remove(job)
            return best_slice_index
        return None

    def deallocate_job(self, job) -> bool:
        """job을 MIG 슬라이스에서 해제 (running list에서도 자동 제거)"""
        # 1. Slice에서 제거
        found = False
        job_id = job.job_id
        for s in self.slices:
            if job_id in s['jobs']:
                s['used'] = 0  # 슬라이스 전체 해제
                s['jobs'].remove(job_id)
                found = True
                break

        if not found:
            return False
        if job.job_type in ("HP", "HP-scale-out"):
            self.hp_running_jobs.remove(job)
        else:
            self.spot_running_jobs.remove(job)

        return True  # Slice에서는 제거했지만 running list에 없음 (정상 케이스)

    def get_slice_info(self) -> str:
        """슬라이스 정보 문자열 반환 (각 슬라이스는 1 job만 할당 가능)"""
        info = []
        for s in self.slices:
            if len(s['jobs']) == 0:
                info.append(f"{s['size']}g:empty")
            else:
                job_id = s['jobs'][0]
                info.append(f"{s['size']}g:{job_id}")
        return f"[{', '.join(info)}]"

    def reconfigure(self, new_profile: str):
        """MIG 프로파일 재구성 (기존 HP jobs 보존)"""
        # 기존 HP jobs와 그들의 실제 할당 크기 저장
        hp_jobs_to_restore = []
        for job in self.hp_running_jobs:
            # allocated_size 사용 (실제 할당된 슬라이스 크기)
            actual_size = job.allocated_size if job.allocated_size else job.req
            hp_jobs_to_restore.append((job, actual_size))
            print(f"    [reconfigure] HP job {job.job_id}: req={job.req}, allocated_size={job.allocated_size} → restore to {actual_size}g")

        # 새 프로파일로 슬라이스 초기화
        self.mig_profile = new_profile
        self.total_capacity = sum(int(c) for c in new_profile)
        self._init_slices()

        # HP jobs를 새 슬라이스에 다시 할당 (큰 것부터)
        hp_jobs_to_restore.sort(key=lambda x: -x[1])  # 큰 슬라이스부터 할당

        for job, actual_size in hp_jobs_to_restore:
            # 같은 크기의 빈 슬라이스 찾기
            restored = False
            for s in self.slices:
                if s['size'] == actual_size and len(s['jobs']) == 0:
                    s['jobs'].append(job.job_id)
                    s['used'] = actual_size
                    print(f"    [reconfigure] Restored HP job {job.job_id} to {actual_size}g slice ")
                    restored = True
                    break

            if not restored:
                print(f"    [reconfigure] WARNING: Could not restore HP job {job.job_id} to {actual_size}g slice!")
    
    def get_running_hp_jobs(self) -> List[Job]:
        """현재 실행 중인 HP job 리스트 반환

        Returns:
            List[Job]: 이 노드에서 실행 중인 HP job 리스트
        """
        return self.hp_running_jobs

    def get_running_spot_jobs(self) -> List[Job]:
        """현재 실행 중인 Spot job 리스트 반환

        Returns:
            List[Job]: 이 노드에서 실행 중인 Spot job 리스트
        """
        return self.spot_running_jobs
     
    def get_available_profile(self) -> str:
        """현재 HP job을 제외한 사용 가능한 MIG 프로파일 반환

        Returns:
            str: HP가 없는 슬라이스들로 구성된 프로파일 (예: "1111")

        설명:
            - 빈 슬라이스: available 
            - HP job 슬라이스: 제외 (보호됨)
            - Spot job 슬라이스: available (preempt 가능)
        """
        available_profile = ""

        # HP job ID들을 미리 set으로 추출 (효율적인 검색을 위해)
        hp_job_ids = {job.job_id for job in self.hp_running_jobs}

        for s in self.slices:
            # 빈 슬라이스는 항상 available
            if len(s['jobs']) == 0:
                available_profile += str(s['size'])
            else:
                # Job이 있는 슬라이스: HP job이 없으면 available (Spot job은 preempt 가능)
                has_hp = False
                for job_id in s['jobs']:
                    # HP job 체크
                    if job_id in hp_job_ids:
                        has_hp = True
                        break

                # HP job이 없으면 available (Spot job은 preempt 가능하므로)
                if not has_hp:
                    available_profile += str(s['size'])

        # Profile이 비어있으면 fallback (모든 슬라이스가 HP로 차있는 경우)
        if not available_profile:
            # HP가 차지하는 용량만큼 뺀 나머지를 1g 슬라이스로
            hp_capacity = sum(job.req for job in self.hp_running_jobs)
            available_capacity = self.total_capacity - hp_capacity
            available_profile = "1" * max(0, available_capacity)

        return available_profile
    
    def get_jobs_for_ilp(self) -> List[Job]:
        """ILP에 사용할 job 리스트 반환 (실행 중인 Spot job + 대기 중인 Spot job)

        Returns:
            List[Job]: ILP에 포함할 job 리스트

        설명:
            - 실행 중인 Spot job: preempt 후 재배치 가능
            - 대기 중인 Spot job: 새로 배치
            - HP job은 제외 (보호됨)
        """
        running_spot_jobs = self.get_running_spot_jobs()
        return running_spot_jobs + self.spot_waiting_queue


print("Data structures defined")