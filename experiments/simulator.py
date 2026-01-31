# 필요한 라이브러리 import
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt

from bin_pack import mock_bin_pack, run_bin_pack
from job import Job, Event, GPUNode
from scheduler_utils import (
    merge_profiles,
    calculate_available_capacity,
    extract_initial_profile_from_bin_packing,
    calculate_reserved_profile
)
import numpy as np


class Simulator:
    """GPU 스케줄러 시뮬레이터"""

    def __init__(self, jobs: List[Job],node_name: str="skt-6gtb-ars",profile: str = "1111111",  mode: str = "baseline"):
        """
        Args:
            jobs: Job 리스트
            mode: 'baseline', 'non-preemptive', 'preemptive'
        """
        self.jobs = sorted(jobs, key=lambda j: j.submit_time)
        self.mode = mode

        # 이벤트 큐
        self.event_queue: List[Event] = []

        # GPU 노드 - Baseline은 7G 고정, 나머지는 1111111 시작
        if mode == "baseline":
            self.node = GPUNode(node_name, mig_profile="7")
        else:
            print(f"Initializing GPUNode with profile: {profile}")
            self.node = GPUNode(node_name, mig_profile=profile)
            print(f"Initial MIG slices: {self.node.get_slice_info()}")
            print(f"Total capacity: {self.node.get_capacity()}g")

        # 실행 중인 job들 "jov_id": Job 객체
        """
            self.running_jobs = {                                                                                                                  
        "hp-001": Job(job_id="hp-001", name="vllm", job_type="deploy", req=3, ...),                                                        
        "spot-002": Job(job_id="spot-002", name="bert", job_type="deploy", req=1, ...),                                                    
        } """
        self.running_jobs: Dict[str, Job] = {}

        # 대기 큐
        self.waiting_queue: deque[Job] = deque()

        # 완료된 job들
        self.completed_jobs: List[Job] = []

        # 메트릭
        self.total_wait_time = 0.0
        self.total_jct = 0.0
        self.mig_reconfig_count = 0
        self.mig_reconfig_overhead = 0.0
        self.preemption_count = 0

        # 현재 시간
        self.current_time = 0.0

        # 시뮬레이션 시작 시간 (첫 job의 submit_time을 기준으로)
        self.start_time = self.jobs[0].submit_time if self.jobs else 0.0

        # 초기 이벤트 생성
        for job in self.jobs:
            heapq.heappush(self.event_queue, Event(job.submit_time, 'arrival', job))

    def format_time(self, time_seconds: float) -> str:
        """시간을 HH:MM:SS 형식으로 변환 (시뮬레이션 시작 시간 기준)"""
        elapsed = time_seconds - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def get_used_capacity(self) -> int:
        return sum(job.req for job in self.running_jobs.values())

    def get_available_capacity(self) -> int:
        """MIG 슬라이스 제약을 고려한 사용 가능한 용량"""
        return self.node.get_available_capacity()

    def is_node_idle(self) -> bool:
        return len(self.running_jobs) == 0

    def try_schedule(self) -> bool:
        """대기 큐에서 job 스케줄링 시도"""
        scheduled_any = False
        jobs_to_remove = []

        for job in list(self.waiting_queue):
            # MIG 슬라이스에 할당 가능한지 확인
            if self.node.can_allocate(job.req):
                # 실제로 MIG 슬라이스에 할당
                if self.node.allocate_job(job.job_id, job.req):
                    job.start_time = self.current_time
                    job.end_time = self.current_time + (job.duration * 60)

                    self.running_jobs[job.job_id] = job
                    jobs_to_remove.append(job)

                    slice_info = self.node.get_slice_info()
                    time_str = self.format_time(self.current_time)
                    print(f"[{time_str}] Job {job.job_id} started (req: {job.req}g, duration: {job.duration:.0f}min)")
                    print(f"  MIG slices: {slice_info}")

                    heapq.heappush(self.event_queue, Event(job.end_time, 'completion', job))
                    scheduled_any = True
            else:
                available = self.get_available_capacity()
                time_str = self.format_time(self.current_time)
                print(f"[{time_str}] Job {job.job_id} waiting (req: {job.req}g, available capacity: {available}g but no suitable MIG slice)")
                break

        for job in jobs_to_remove:
            self.waiting_queue.remove(job)

        return scheduled_any

    def calculate_current_expected_jct(self) -> float:
        """현재 상태에서 대기 중인 job들의 예상 평균 JCT"""
        if not self.waiting_queue:
            return 0.0

        # 실행 중인 job 중 가장 늦게 끝나는 시간
        max_end_time = max(
            (job.end_time for job in self.running_jobs.values()),
            default=self.current_time
        )

        # 대기 중인 job들이 순차 실행된다고 가정
        total_jct = 0
        cumulative_time = max_end_time

        for job in self.waiting_queue:
            job_duration = job.duration * 60
            completion_time = cumulative_time + job_duration
            jct = completion_time - job.submit_time
            total_jct += jct
            cumulative_time += job_duration

        return total_jct / len(self.waiting_queue)

    def try_bin_packing(self):
        """Bin packing 시도"""
        queue_threshold = 1  # 실험용: 1개만 대기해도 시도

        # HP job이 큐에 있으면 즉시 bin packing 실행
        has_hp_job = any(job.job_type == "HP" for job in self.waiting_queue)

        if not has_hp_job and len(self.waiting_queue) < queue_threshold:
            return

        if has_hp_job:
            print(f"  HP job detected in queue - Triggering immediate bin packing!")

        # Non-Preemptive: 노드가 idle이어야만
        if self.mode == "non-preemptive" and not self.is_node_idle():
            return

        # ===== Backend 방식: Reserved Profile 적용 =====

        # 1. 실행 중인 job 목록 조회
        running_jobs_list = list(self.running_jobs.values())

        # 2. HP jobs와 Spot jobs 분리
        hp_jobs = [j for j in running_jobs_list if j.job_type == "HP"]
        spot_jobs = [j for j in running_jobs_list if j.job_type == "Spot"]

        print(f"  [bin_packing] HP jobs (reserved): {len(hp_jobs)}")
        print(f"  [bin_packing] Spot jobs: {len(spot_jobs)}")

        # 3. HP jobs가 사용하는 용량 계산 (Reserved Profile)
        reserved_capacity = sum(j.req for j in hp_jobs)
        reserved_profile_list = sorted([j.req for j in hp_jobs], reverse=True)
        reserved_profile = ''.join(str(s) for s in reserved_profile_list) if reserved_profile_list else ""

        print(f"  [bin_packing] Reserved capacity (HP jobs): {reserved_capacity}g")
        print(f"  [bin_packing] Reserved profile: {reserved_profile}")

        # 4. Available capacity 계산 (전체 7g 가정)
        total_capacity = 7
        available_capacity = total_capacity - reserved_capacity
        available_init_config = "1" * available_capacity if available_capacity > 0 else ""

        print(f"  [bin_packing] Available capacity for bin packing: {available_capacity}g")
        print(f"  [bin_packing] Available init_config: {available_init_config}")

        # 5. Spot jobs + Queued jobs만 bin packing에 포함
        all_jobs_for_bp = []

        # Spot jobs만 포함 (HP jobs는 제외)
        for sj in spot_jobs:
            # 남은 실행 시간 계산 (초 -> 분)
            remaining_duration = max(0, (sj.end_time - self.current_time) / 60)
            all_jobs_for_bp.append({
                "name": sj.job_id,
                "size": sj.req,
                "duration": int(remaining_duration)
            })

        # 큐에 있는 job 추가
        for job in self.waiting_queue:
            all_jobs_for_bp.append({
                "name": job.job_id,
                "size": job.req,
                "duration": int(job.duration)
            })

        print(f"  [bin_packing] Total jobs for bin packing (Spot + Queued): {len(all_jobs_for_bp)}")
        print(f"  [bin_packing]   - HP jobs: {len(hp_jobs)} (protected, not in bin packing)")
        print(f"  [bin_packing]   - Spot jobs: {len(spot_jobs)}")
        print(f"  [bin_packing]   - Queued jobs: {len(self.waiting_queue)}")

        # 6. Bin Packing 실행 (Available capacity만 사용)
        bp_result = None

        # Available capacity가 없거나 job이 없으면 bin packing 스킵
        if available_capacity <= 0:
            print(f"  [bin_packing] No available capacity (all reserved by HP jobs)")
            return
        elif not all_jobs_for_bp:
            print(f"  [bin_packing] No jobs to schedule (only HP jobs running)")
            return
        else:
            # Bin packing 실행
            bp_result = run_bin_pack(available_init_config, all_jobs_for_bp)
            # bp_result = mock_bin_pack(available_init_config, all_jobs_for_bp)

        

        if not bp_result:
            return

        # 7. Profile Merging (Backend 방식)
        # Bin packing 결과에 HP jobs 자리(reserved profile)를 추가
        chosen_cfg = bp_result.get("chosen_cfg", {})
        bp_profile = chosen_cfg.get(1, "")  # Bin packing이 만든 profile

        # Reserved profile + Bin packing profile = Final profile
        new_profile = merge_profiles(reserved_profile, bp_profile)

        print(f"  [bin_packing] Bin packing profile: {bp_profile}")
        print(f"  [bin_packing] Reserved profile (HP): {reserved_profile}")
        print(f"  [bin_packing] Final merged profile: {new_profile}")

        current_profile = self.node.mig_profile

        # Preemptive: JCT improvement 계산
        if self.mode == "preemptive":
            bp_avg_jct = bp_result.get("avg_jct", float('inf')) * 60  # 분->초
            current_jct = self.calculate_current_expected_jct()

            improvement_threshold = 0.95  # 실험용: 5% 이상 개선

            if bp_avg_jct < current_jct * improvement_threshold:
                # Preemption 수행 (이미 위에서 hp_jobs, spot_jobs 분리됨)
                improvement_pct = (1 - bp_avg_jct / current_jct) * 100

                print(f"  [preemption] Improvement: {improvement_pct:.1f}%")

                terminated_jobs = []

                # Spot job만 종료 (HP job은 보호됨)
                for job in spot_jobs:
                    self.node.deallocate_job(job.job_id, job.req)  # MIG 슬라이스에서 해제
                    del self.running_jobs[job.job_id]
                    self.waiting_queue.appendleft(job)  # 재큐잉
                    terminated_jobs.append(job)
                    self.preemption_count += 1

                # HP job은 절대 preempt하지 않음 (보호됨)
                if hp_jobs:
                    print(f"  HP jobs protected from preemption: {[j.job_id for j in hp_jobs]}")

                if terminated_jobs:
                    # MIG reconfiguration
                    if new_profile != current_profile:
                        time_str = self.format_time(self.current_time)
                        print(f"[{time_str}] MIG Reconfiguration {current_profile} -> {new_profile} (Improvement: {improvement_pct:.1f}%)")
                        self.node.reconfigure(new_profile)
                        self.mig_reconfig_count += 1

                        reconfig_overhead = 5.0  # 실험용: 5초
                        self.mig_reconfig_overhead += reconfig_overhead
                        self.current_time += reconfig_overhead
                        time_str = self.format_time(self.current_time)
                        print(f"[{time_str}] Reconfiguration completed (overhead: {reconfig_overhead:.0f}s)")

                    # 재스케줄링
                    self.try_schedule()
        else:
            # Non-Preemptive: 단순 reconfiguration
            if new_profile != current_profile:
                time_str = self.format_time(self.current_time)
                print(f"[{time_str}] MIG Reconfiguration {current_profile} -> {new_profile}")
                self.node.reconfigure(new_profile)
                self.mig_reconfig_count += 1

                reconfig_overhead = 5.0  # 실험용: 5초
                self.mig_reconfig_overhead += reconfig_overhead
                self.current_time += reconfig_overhead
                time_str = self.format_time(self.current_time)
                print(f"[{time_str}] Reconfiguration completed (overhead: {reconfig_overhead:.0f}s)")

    def handle_arrival(self, job: Job):
        self.waiting_queue.append(job)
        slice_info = self.node.get_slice_info()
        available = self.get_available_capacity()
        used = self.get_used_capacity()
        total = self.node.get_capacity()
        print(f"  MIG slices: {slice_info}, Available: {available}g / Total: {total}g (Used: {used}g)")
        self.try_schedule()

        if self.mode in ["non-preemptive", "preemptive"]:
            self.try_bin_packing()

    def handle_completion(self, job: Job):
        if job.job_id in self.running_jobs:
            # MIG 슬라이스에서 할당 해제
            self.node.deallocate_job(job.job_id, job.req)
            del self.running_jobs[job.job_id]

        self.completed_jobs.append(job)
        self.total_jct += job.jct()
        self.total_wait_time += job.wait_time()

        time_str = self.format_time(self.current_time)
        print(f"[{time_str}] Job {job.job_id} completed (freed: {job.req}g)")
        slice_info = self.node.get_slice_info()
        available = self.get_available_capacity()
        used = self.get_used_capacity()
        total = self.node.get_capacity()
        print(f"  MIG slices: {slice_info}")
        print(f"  Available: {available}g / Total: {total}g (Used: {used}g)")

        self.try_schedule()

        if self.mode in ["non-preemptive", "preemptive"] and self.is_node_idle():
            self.try_bin_packing()

    def run(self, verbose: bool = False):
        """시뮬레이션 실행"""
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == 'arrival':
                time_str = self.format_time(self.current_time)
                print(f"\n[{time_str}] Job {event.job.job_id} arrived (req: {event.job.req}g)")
                self.handle_arrival(event.job)
            elif event.event_type == 'completion':
                self.handle_completion(event.job)

        if verbose:
            self.print_results()

        return self.get_metrics()

    def get_metrics(self) -> dict:
        """메트릭 반환"""
        completed_jobs = len(self.completed_jobs)
        avg_jct = self.total_jct / completed_jobs if completed_jobs > 0 else 0
        avg_wait = self.total_wait_time / completed_jobs if completed_jobs > 0 else 0

        jcts = [job.jct() / 60 for job in self.completed_jobs]

        return {
            "mode": self.mode,
            "total_jobs": len(self.jobs),
            "completed_jobs": completed_jobs,
            "avg_jct_min": avg_jct / 60,
            "avg_wait_min": avg_wait / 60,
            "mig_reconfigs": self.mig_reconfig_count,
            "reconfig_overhead_sec": self.mig_reconfig_overhead,
            "preemptions": self.preemption_count,
            "final_profile": self.node.mig_profile,
            "jcts": jcts
        }

    def print_results(self):
        metrics = self.get_metrics()
        print(f"\n{'='*80}")
        print(f"Results - {metrics['mode'].upper()}")
        print(f"{'='*80}")
        print(f"Total jobs: {metrics['total_jobs']}")
        print(f"Completed jobs: {metrics['completed_jobs']}")
        print(f"Average JCT: {metrics['avg_jct_min']:.2f} minutes")
        print(f"Average wait: {metrics['avg_wait_min']:.2f} minutes")
        print(f"MIG reconfigurations: {metrics['mig_reconfigs']}")
        print(f"Preemptions: {metrics['preemptions']}")
        print(f"Final MIG profile: {metrics['final_profile']}")

print("Simulator class defined")