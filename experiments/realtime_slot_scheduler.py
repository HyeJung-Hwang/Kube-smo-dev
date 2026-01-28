import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import time
import heapq
import subprocess
from dataclasses import dataclass
from typing import List, Optional
from simulator import Simulator
from job import Event, GPUNode
import realtime_deploy as deploy

from bin_pack import run_bin_pack
from mig_dynamic_config import VALID_MIG_PROFILES, is_valid_mig_profile

@dataclass
class JobState:
    """Job의 실행 상태를 추적"""
    job_id: str
    original_duration: float  # 원래 요청된 duration (분)
    remaining_duration: float  # 남은 duration (분)
    job_type: str = "Spot"  # HP, Spot, HP-scale-out, HP-scale-in, HP-scale-up, HP-scale-down
    workload_type: str = "AI"  # AI, RAN (HP job에서만 사용)
    submit_time: float = 0.0  # job 도착 시간 (시뮬레이션 시간)
    actual_start_time: Optional[float] = None  # 실제 배포 시작 시간 (시뮬레이션 시간)
    completion_time: Optional[float] = None  # 완료 시간 (시뮬레이션 시간)
    times_preempted: int = 0  # 몇 번 preempt 되었는지
    total_run_time: float = 0.0  # 총 실행 시간 (분)

class RealTimeSlotScheduler(Simulator):
    def __init__(self, jobs, nodes, node_selection_strategy, speed, dry_run=True):
        # 기본 설정
        self.nodes = nodes
        self.node_selection_strategy = node_selection_strategy
        self.speed = speed
        self.dry_run = dry_run
        self.multi_node = len(nodes) > 1

        # 시간 관련
        self.current_time = 0.0
        self.wall_start = 0.0
        self.start_time = jobs[0].submit_time if jobs else 0.0

        # slot 스케쥴링 상태 정보
        self.current_plan = None
        self.current_slot = None
        self.slot_jobs = {}
        self.plan_start_time = 0

        # Job 관리
        self.running_jobs = {}  # {job_id: job}
        self.completed_jobs = []
        self.job_states = {}  # {job_id: JobState}

        # 메트릭
        self.total_jct = 0.0
        self.total_wait_time = 0.0

        # 상세 메트릭 추적
        self.deployed_hp_count = 0  # 배포된 HP job 수
        self.deployed_spot_count = 0  # 배포된 Spot job 수
        self.original_jobs = jobs  # 원본 job 리스트 (job_type 참조용)

        # queue 상태 정보
        self.event_queue: List[Event] = []
        self.node_waiting_spot_queues = {}
        self.node_waiting_hp_queues = {}
        for node in self.nodes:
            node_key = (node.name, node.gpu_index)
            self.node_waiting_spot_queues[node_key] = []
            self.node_waiting_hp_queues[node_key] = []

        # job 배정 결과
        self.job_to_target_node = {}  # {job_id: node}
        self.job_to_node = {}  # {job_id: node} - 현재 실행 중인 job
        self.job_deployment_history = {}  # {job_id: node_name} - 완료된 job 포함 (통계용)

        # Job arrival events 생성
        for job in jobs:
            arrival_event = Event(
                time=job.submit_time,
                event_type='arrival',
                job=job
            )
            heapq.heappush(self.event_queue, arrival_event)
    
    def select_node_for_job(self, job):
        """노드 선택 전략에 따라 도착한 job을 배포할 노드 선정"""
        # RAN job은 항상 skt-6gtb-ars 노드로 배포
        if getattr(job, 'workload_type', 'AI') == "RAN":
            for node in self.nodes:
                if node.name == "skt-6gtb-ars":
                    return node
            # skt-6gtb-ars 노드가 없으면 첫 번째 노드 (fallback)
            print(f"    RAN job but skt-6gtb-ars node not found, using first node")
            return self.nodes[0]

        # 일반 job은 strategy에 따라 선택
        if self.node_selection_strategy == "least_loaded":
            if job.job_type in ("HP", "HP-scale-out"):
                # HP: (hp_running_jobs + hp_waiting_queue) 합이 가장 적은 노드
                # → 대기 중인 job도 반영하여 convoy effect 방지
                return min(self.nodes, key=lambda n: len(n.hp_running_jobs) + len(n.hp_waiting_queue))
            else:
                # Spot: spot_waiting_queue가 가장 적은 노드
                return min(self.nodes, key=lambda n: len(n.spot_waiting_queue))
        else:
            # 기본: 첫 번째 노드
            return self.nodes[0]

    # -----------------------------------------
    # enqueue & dequeue 관련 함수들
    def handle_arrival(self, job):
        # JobState 생성 (메트릭 추적용)
        self.job_states[job.job_id] = JobState(
            job_id=job.job_id,
            original_duration=job.duration,
            remaining_duration=job.duration,
            job_type=job.job_type,  # HP, Spot 등
            workload_type=getattr(job, 'workload_type', 'AI'),  # AI, RAN
            submit_time=self.current_time
        )

        # 1. 도착한 job을 배포할 노드 & GPU 선정 후 해당 큐에 넣기
        target_node = self.select_node_for_job(job)
        time_str = self.format_time(self.current_time)

        # 2. job type에 따라 다른 큐에 넣기
        if job.job_type == "HP":
            print(f"\n [{time_str}] {job.job_id} arrived ({job.req}g) | {job.job_type} | → {target_node.name} GPU {target_node.gpu_index}")
            target_node.hp_waiting_queue.append(job)
            self.try_schedule_hp(target_node)
        elif job.job_type == "HP-scale-out":
            print(f"\n [{time_str}] {job.job_id} arrived ({job.req}g) | {job.job_type} | target={job.target_job_id} | → {target_node.name} GPU {target_node.gpu_index}")
            target_node.hp_waiting_queue.append(job)
            self.try_schedule_hp_scale_out(target_node, job)
        elif job.job_type == "HP-scale-in":
            print(f"\n [{time_str}] {job.job_id} arrived | {job.job_type} | target={job.target_job_id}")
            self.handle_scale_in(job)
        elif job.job_type == "HP-scale-up":
            print(f"\n [{time_str}] {job.job_id} arrived ({job.req}g) | {job.job_type} | target={job.target_job_id}")
            self.handle_scale_up(job)
        elif job.job_type == "HP-scale-down":
            print(f"\n [{time_str}] {job.job_id} arrived ({job.req}g) | {job.job_type} | target={job.target_job_id}")
            self.handle_scale_down(job)
        else:  # Spot
            print(f"\n [{time_str}] {job.job_id} arrived ({job.req}g) | {job.job_type} | → {target_node.name} GPU {target_node.gpu_index}")
            target_node.spot_waiting_queue.append(job)
            self.try_schedule_spot(target_node)

    
    def handle_completion(self, job):
        """Job 완료 처리 (multi-node 지원)"""
        # 완료된 job의 노드 정보 저장 (삭제 전)
        completed_node = None
        if job.job_id in self.job_to_node:
            completed_node = self.job_to_node[job.job_id]

        if job.job_id in self.running_jobs:
            # Job 상태 업데이트: 완료 (optional)
            if job.job_id in self.job_states:
                self.update_job_state_on_completion(job.job_id, self.current_time)

            # Pod undeploy
            if not self.dry_run:
                self.undeploy(job)

            # Job이 배포된 노드에서 해제 (running list에서도 자동 제거됨)
            if completed_node:
                completed_node.deallocate_job(job)
                del self.job_to_node[job.job_id]

            del self.running_jobs[job.job_id]

            # Pod가 완전히 삭제될 때까지 대기 (dry-run에서는 skip)
            if not self.dry_run:
                if not deploy.wait_for_pod_deletion([job.job_id], max_wait_sec=60):
                    print(f"    Warning: Pod {job.job_id} may still be running!")

        self.completed_jobs.append(job)
        self.total_jct += job.jct()
        self.total_wait_time += job.wait_time()

        time_str = self.format_time(self.current_time)

        # 어느 노드에서 완료되었는지 표시
        if completed_node:
            print(f" [{time_str}] {job.job_id} completed on {completed_node.name} ({job.req}g freed)")
            print(f"  {completed_node.get_slice_info()}")
        else:
            print(f" [{time_str}] {job.job_id} completed ({job.req}g freed)")

        # Slot transition 체크 (Slot-based ILP)
        if completed_node and completed_node.current_plan is not None:
            self.check_slot_transition(completed_node)

        # HP job 완료 시 대기 중인 HP job 스케줄링 재시도
        if completed_node and job.job_type in ("HP", "HP-scale-out"):
            if len(completed_node.hp_waiting_queue) > 0:
                print(f"  HP completed, retrying {len(completed_node.hp_waiting_queue)} waiting HP jobs")
                self.try_schedule_hp(completed_node)

    # -----------------------------------------
    # pop 해서 실행하는 함수들 

    def try_schedule_spot(self, node: GPUNode):
        """Spot job 스케줄링 (Slot-based ILP)"""

        # ========================================
        # Fast Path: RAN HP 실행 중이면 빈 슬롯에 바로 배포
        # ========================================
        # RAN workload는 MIG 재구성이 불가능하므로 bin packing 없이
        # 현재 빈 슬롯에 바로 Spot job을 배포
        running_hp_jobs = node.get_running_hp_jobs()
        has_ran_hp = any(getattr(job, 'workload_type', 'AI') == "RAN" for job in running_hp_jobs)

        if has_ran_hp and len(node.spot_waiting_queue) > 0:
            print(f"  [Fast Path] RAN HP running, deploying Spot to available slots directly")
            print(f"    Current profile: {node.mig_profile}")
            print(f"    Node state: {node.get_slice_info()}")

            deployed_count = 0
            # spot_waiting_queue 복사본으로 순회 (순회 중 수정 방지)
            for job in list(node.spot_waiting_queue):
                if node.can_allocate(job.req):
                    slice_idx = node.allocate_job(job)
                    if slice_idx is not None:
                        # 시간 설정
                        if not self.dry_run:
                            actual_time = (time.time() - self.wall_start) * self.speed + self.start_time
                        else:
                            actual_time = self.current_time

                        job.start_time = actual_time
                        job.end_time = actual_time + (job.duration * 60)

                        # JobState 업데이트
                        if job.job_id in self.job_states:
                            self.job_states[job.job_id].actual_start_time = actual_time

                        self.running_jobs[job.job_id] = job
                        self.job_to_node[job.job_id] = node
                        self.deployed_spot_count += 1

                        # Completion event
                        completion_event = Event(
                            time=job.end_time,
                            event_type='completion',
                            job=job
                        )
                        heapq.heappush(self.event_queue, completion_event)

                        # 실제 배포
                        if not self.dry_run:
                            avoid_uuids = [j.mig_uuid for j in node.get_running_hp_jobs() if j.mig_uuid]
                            avoid_uuids += [j.mig_uuid for j in node.get_running_spot_jobs() if j.mig_uuid and j.job_id != job.job_id]

                            mig_uuid = deploy.deploy_job(
                                job_id=job.job_id,
                                node_name=node.name,
                                gpu_g=job.allocated_size if job.allocated_size else job.req,
                                gpu_index=node.gpu_index,
                                slice_index=slice_idx,
                                avoid_uuids=avoid_uuids if avoid_uuids else None
                            )
                            if mig_uuid:
                                job.mig_uuid = mig_uuid

                        deployed_count += 1
                        print(f"    Spot {job.job_id} ({job.req}g) deployed to slice {slice_idx}")
                else:
                    print(f"    Spot {job.job_id} ({job.req}g) waiting - no suitable slot")

            if deployed_count > 0:
                print(f"    Node state after: {node.get_slice_info()}")

            # Fast path 사용 후 종료 (bin packing 스킵)
            return

        # ========================================
        # Phase 1: Initial Planning (Plan이 없을 때)
        # ========================================
        if node.current_plan is None and len(node.spot_waiting_queue) >= 3:
            available_profile = node.get_available_profile()
            jobs_for_ilp = node.get_jobs_for_ilp()

            # HP job이 있는지 확인하여 reserved 정보 계산
            running_hp_jobs = node.get_running_hp_jobs()
            reserved_profile = None
            reserved_gi_ids = []
            cleanup_mode = False

            if running_hp_jobs:
                # HP가 있으면 cleanup_mode 사용 (가장 정확)
                # allocated_size 사용 (실제 할당된 크기)
                hp_sizes = sorted([job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs], reverse=True)
                reserved_profile = "".join(str(s) for s in hp_sizes)
                reserved_uuids = [job.mig_uuid for job in running_hp_jobs if job.mig_uuid]

                # UUID → GI ID 매핑
                if reserved_uuids and not self.dry_run:
                    try:
                        uuid_to_gi = deploy.get_uuid_to_gi_mapping(node.name, node.gpu_index)
                        reserved_gi_ids = [uuid_to_gi[uuid] for uuid in reserved_uuids if uuid in uuid_to_gi]
                        cleanup_mode = True
                        print(f"    [cleanup_mode] HP UUIDs: {reserved_uuids}")
                        print(f"    [cleanup_mode] Reserved GI IDs: {reserved_gi_ids}")
                    except Exception as e:
                        print(f"    [cleanup_mode] WARNING: Failed to get GI IDs: {e}")
                        cleanup_mode = False
                elif self.dry_run and running_hp_jobs:
                    # dry_run: HP가 있으면 cleanup_mode 시뮬레이션
                    # 가상의 GI ID 사용 (실제와 동일한 로직 흐름 유지)
                    reserved_gi_ids = list(range(len(running_hp_jobs)))  # [0, 1, ...] 가상 ID
                    cleanup_mode = True
                    print(f"    [dry-run cleanup_mode] Simulated reserved GI IDs: {reserved_gi_ids}")

                # Cleanup mode 전에 실행 중인 Spot job preempt
                if cleanup_mode:
                    running_spot_jobs = node.get_running_spot_jobs()
                    for spot_job in running_spot_jobs:
                        print(f"    [cleanup_mode] Preempting Spot job {spot_job.job_id} for ILP planning")
                        # Completion event 취소
                        for event in self.event_queue:
                            if event.job and event.job.job_id == spot_job.job_id and event.event_type == 'completion':
                                event.cancelled = True
                        # Job을 슬라이스에서 해제하고 대기 큐로 이동
                        node.deallocate_job(spot_job)
                        spot_job.start_time = None
                        spot_job.end_time = None
                        node.spot_waiting_queue.append(spot_job)
                    # jobs_for_ilp 재계산 (preempt된 job 포함)
                    jobs_for_ilp = node.get_jobs_for_ilp()

            print(f"  Initial planning triggered ({len(node.spot_waiting_queue)} jobs waiting)")
            print(f"    Available profile: {available_profile}")
            print(f"    Reserved profile (HP): {reserved_profile}")
            print(f"    Cleanup mode: {cleanup_mode}")
            print(f"    Jobs for ILP: {[job.job_id for job in jobs_for_ilp]}")

            # Job 객체를 dict 형식으로 변환 (bin_pack이 기대하는 형식)
            import math
            jobs_for_bp = []
            for job in jobs_for_ilp:
                # Duration: 분 단위, 올림 처리 (job이 slot을 넘어가면 여러 slot 필요)
                duration_min = max(1, math.ceil(job.duration))
                jobs_for_bp.append({
                    "name": job.job_id,
                    "size": job.req,
                    "duration": duration_min
                })

            # ILP 실행
            # slot 단위: 30분
            slot_minutes = 30
            try:
                bp_result = run_bin_pack(
                    available_profile,
                    jobs_for_bp,
                    slot_minutes=slot_minutes,
                    reserved_profile=reserved_profile,
                    node_name=node.name,
                    gpu_index=node.gpu_index,
                    cleanup_mode=cleanup_mode,
                    reserved_gi_ids=reserved_gi_ids if cleanup_mode else None,
                    dry_run=self.dry_run,
                )

                # ILP 실패 시 (HP가 전체 GPU 사용 등) 스킵
                if bp_result is None:
                    print(f"    Bin packing returned None (no space for Spot jobs)")
                    print(f"    Spot jobs will wait until HP jobs complete")
                    return

                # Plan 저장 (전체 bp_result 저장!)
                node.current_plan = bp_result
                node.current_plan['cleanup_mode'] = cleanup_mode  # cleanup_mode 저장
                node.current_plan['reserved_profile'] = reserved_profile  # reserved_profile 저장
                node.current_slot = 1
                node.plan_start_time = self.current_time

                # DEBUG: chosen_cfg 구조 확인
                tmax_slot = bp_result.get('Tmax_slot', len(bp_result['chosen_cfg']))
                print(f"    Plan created:")
                print(f"       Tmax_slot: {tmax_slot}")
                print(f"       slot_jobs: {bp_result.get('slot_jobs', {})}")
                print(f"       Reconfigs needed at slots: {bp_result.get('reconfigs', [])}")

                # Timeout 이벤트 생성 (Tmax_slot까지만)
                slot_duration_sec = slot_minutes * 60  # ILP의 slot_minutes와 동일하게 (초 단위)
                for slot_idx in range(1, tmax_slot + 1):
                    timeout_time = node.plan_start_time + slot_idx * slot_duration_sec

                    timeout_event = Event(
                        time=timeout_time,
                        event_type='slot_timeout',
                        job=None,
                        node=node,
                        slot_idx=slot_idx,
                        plan=bp_result  # Plan 객체 참조 (검증용)
                    )
                    heapq.heappush(self.event_queue, timeout_event)
                    print(f"       Timeout event added: T={timeout_time:.1f}s for slot {slot_idx}")

                # Plan 생성 직후 첫 slot 배포
                self._execute_current_slot(node)

            except Exception as e:
                print(f"    Bin packing failed: {e}")
                node.current_plan = None
                return

    def _execute_current_slot(self, node: GPUNode):
        """현재 slot의 job들을 배포 (Phase 2)"""
        if node.current_plan is None:
            return

        current_slot = node.current_slot
        chosen_cfg = node.current_plan["chosen_cfg"]
        slot_jobs = node.current_plan["slot_jobs"]
        reconfigs = node.current_plan.get("reconfigs", [])
        tmax_slot = node.current_plan.get("Tmax_slot", len(chosen_cfg))

        # Slot이 plan 범위를 벗어나면 plan 종료
        if current_slot > tmax_slot:
            print(f"  Plan completed (all slots finished)")
            node.current_plan = None
            node.current_slot = None

            # Plan 종료 후 대기 중인 job이 있으면 새 plan 생성
            if len(node.spot_waiting_queue) >= 1:
                print(f"  Re-planning triggered ({len(node.spot_waiting_queue)} jobs still waiting)")
                self.try_schedule_spot(node)
            return

        # 1. MIG Reconfiguration 필요 여부 확인
        spot_target_profile = chosen_cfg[current_slot]  # ILP 결과 (Spot용 profile)

        # HP job이 있으면 HP profile을 합쳐서 전체 target profile 계산
        running_hp_jobs = node.get_running_hp_jobs()
        if running_hp_jobs:
            # allocated_size 사용 (실제 할당된 크기)
            hp_sizes = sorted([job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs], reverse=True)
            hp_profile = "".join(str(s) for s in hp_sizes)
            full_target_profile = hp_profile + spot_target_profile
            reserved_profile = hp_profile
            reserved_uuids = [job.mig_uuid for job in running_hp_jobs if job.mig_uuid]
            print(f"  Slot {current_slot}: HP reserved={hp_profile}, Spot target={spot_target_profile}")
        else:
            full_target_profile = spot_target_profile
            reserved_profile = None
            reserved_uuids = None

        # cleanup_mode 후 slot 1에서 인스턴스 재생성 필요 여부 확인
        cleanup_mode = node.current_plan.get('cleanup_mode', False)
        needs_recreate = False

        if cleanup_mode and current_slot == 1 and current_slot not in reconfigs:
            # cleanup_mode로 managed 인스턴스가 삭제됐지만 slot 1이 reconfigs에 없는 경우
            # → 삭제된 인스턴스를 재생성해야 함
            print(f"  Slot {current_slot}: cleanup_mode used, need to recreate managed instances")
            needs_recreate = True

        if (current_slot in reconfigs and node.mig_profile != full_target_profile) or needs_recreate:
            if needs_recreate:
                print(f"  Slot {current_slot}: Recreating instances after cleanup (profile: {full_target_profile})")
            else:
                print(f"  Slot {current_slot}: Reconfiguring {node.mig_profile} → {full_target_profile}")

            # 실행 중인 모든 Spot job preempt
            running_spot_jobs = node.get_running_spot_jobs()
            if len(running_spot_jobs) > 0:
                print(f"    Preempting {len(running_spot_jobs)} running jobs for MIG reconfiguration:")

                # 상태 변경 (batch undeploy 전)
                for job in running_spot_jobs:
                    print(f"       - {job.job_id}")
                    self.preempt_job(job, node)

                # Batch undeploy (한 번에 모든 job 삭제)
                if not self.dry_run:
                    job_ids = [job.job_id for job in running_spot_jobs]
                    print(f"    Batch undeploying {len(job_ids)} jobs...")
                    success_count, failed = deploy.undeploy_jobs_batch(job_ids)
                    print(f"    Undeployed: {success_count}/{len(job_ids)} jobs")

                    if failed:
                        print(f"    Failed releases: {failed}")

                    # Pod deletion 대기 (MIG reconfig 전에 반드시 필요)
                    print(f"    Waiting for pod deletion...")
                    if not deploy.wait_for_pod_deletion(job_ids, max_wait_sec=60):
                        print(f"    Warning: Some pods may still be running!")

            # MIG reconfiguration 실행 (HP job 보호)
            if not self.dry_run:
                if reserved_profile:
                    print(f"    Preserving HP jobs: {reserved_profile} (UUIDs: {reserved_uuids})")
                    result = deploy.reconfigure_mig(
                        node_name=node.name,
                        profile=full_target_profile,
                        reserved_profile=reserved_profile,
                        reserved_uuids=reserved_uuids if reserved_uuids else None,
                        gpu_index=node.gpu_index
                    )
                else:
                    result = deploy.reconfigure_mig(
                        node_name=node.name,
                        profile=full_target_profile,
                        gpu_index=node.gpu_index
                    )

                # "In use by another client" 에러 처리
                if result == "IN_USE_ERROR":
                    print(f"    'In use' error detected, triggering hard reset...")
                    if not self._handle_in_use_error(node, full_target_profile):
                        print(f"    Hard reset failed!")
                        return
                    # Hard reset 성공 → 기존 slot plan대로 계속 배포 (ILP 재계산 없이)
                    print(f"    Hard reset completed, continuing with existing slot plan...")
                    # node.reconfigure()는 이미 _handle_in_use_error에서 했으므로 스킵
                elif not result:
                    print(f"    MIG reconfiguration failed!")
                    return
                else:
                    # 정상 성공
                    node.reconfigure(full_target_profile)
                    print(f"    Reconfigured to {full_target_profile}")
            else:
                node.reconfigure(full_target_profile)
                print(f"    Reconfigured to {full_target_profile} (dry-run)")

        # 2. 현재 slot의 job들 배포
        jobs_to_deploy_ids = slot_jobs.get(current_slot, [])

        print(f"  Slot {current_slot}: Deploying {len(jobs_to_deploy_ids)} jobs")
        for job_id in jobs_to_deploy_ids:
            # Queue에서 해당 job 찾기
            job = None
            for j in node.spot_waiting_queue:
                if j.job_id == job_id:
                    job = j
                    break

            if job is None:
                # 이미 배포된 job이거나 queue에 없음
                continue

            # 배포 시도
            if node.can_allocate(job.req):
                slice_idx = node.allocate_job(job)
                if slice_idx is not None:
                    # Job state에서 남은 duration 가져오기
                    if hasattr(self, 'job_states') and job.job_id in self.job_states:
                        remaining_duration = self.job_states[job.job_id].remaining_duration
                    else:
                        remaining_duration = job.duration

                    # 실제 배포 시 wall clock 시간 사용, dry_run 시 이벤트 시간 사용
                    if not self.dry_run:
                        actual_time = (time.time() - self.wall_start) * self.speed + self.start_time
                    else:
                        actual_time = self.current_time

                    job.start_time = actual_time
                    job.end_time = actual_time + (remaining_duration * 60)

                    # JobState 업데이트
                    if job.job_id in self.job_states:
                        self.job_states[job.job_id].actual_start_time = actual_time

                    # Running jobs에 추가
                    self.running_jobs[job.job_id] = job
                    self.job_to_node[job.job_id] = node

                    # 메트릭: Spot 배포 카운트 증가
                    self.deployed_spot_count += 1

                    # Completion event 생성
                    completion_event = Event(
                        time=job.end_time,
                        event_type='completion',
                        job=job
                    )
                    heapq.heappush(self.event_queue, completion_event)

                    print(f"    Deployed {job.job_id} to slice {slice_idx}")
                    print(f"       Completion scheduled at T={job.end_time/60:.1f}min")

                    # 실제 Pod 배포
                    if not self.dry_run:
                        # HP job + 이미 배포된 Spot job의 UUID를 피해서 배포
                        hp_uuids = [j.mig_uuid for j in node.hp_running_jobs if j.mig_uuid]
                        spot_uuids = [j.mig_uuid for j in node.spot_running_jobs if j.mig_uuid]
                        avoid_uuids = hp_uuids + spot_uuids if (hp_uuids or spot_uuids) else None

                        # DEBUG: avoid_uuids 출력
                        print(f"    [DEBUG] HP running: {[j.job_id for j in node.hp_running_jobs]}")
                        print(f"    [DEBUG] HP UUIDs: {hp_uuids}")
                        print(f"    [DEBUG] Spot running: {[j.job_id for j in node.spot_running_jobs]}")
                        print(f"    [DEBUG] Spot UUIDs: {spot_uuids}")
                        print(f"    [DEBUG] avoid_uuids: {avoid_uuids}")
                        # allocated_size 사용 (실제 슬라이스 크기)
                        actual_gpu_size = job.allocated_size if job.allocated_size else job.req
                        mig_uuid = deploy.deploy_job(
                            job_id=job.job_id,
                            node_name=node.name,
                            gpu_g=actual_gpu_size,
                            gpu_index=node.gpu_index,
                            slice_index=slice_idx,
                            avoid_uuids=avoid_uuids
                        )
                        if mig_uuid:
                            job.mig_uuid = mig_uuid  # Spot job에도 UUID 저장
                        else:
                            print(f"    Pod deployment failed for {job.job_id}")
                else:
                    print(f"    Failed to allocate {job.job_id}")
            else:
                print(f"    Cannot allocate {job.job_id} ({job.req}g)")

    def try_schedule_hp(self, node: GPUNode):
        """HP job 스케줄링 (High Priority - 즉시 배포)

        전략:
        1. 빈 슬라이스가 있으면 즉시 배포
        2. 없으면 모든 Spot job을 preempt
        3. 모든 HP job을 수용하는 profile로 MIG reconfig
        4. HP job 배포 후 남은 공간에 Spot 재계획
        """

        if len(node.hp_waiting_queue) == 0:
            return

        hp_job = node.hp_waiting_queue[0]

        print(f"  HP job {hp_job.job_id} ({hp_job.req}g) scheduling...")

        # ========================================
        # 최적화: 빈 슬라이스가 있으면 바로 배포
        # ========================================
        if node.can_allocate(hp_job.req):
            print(f"    Found available slice")
            self._deploy_hp_job(hp_job, node)
            return

        # ========================================
        # Feasibility check: preempt 전에 HP 총합 확인
        # ========================================
        running_hp_jobs = node.get_running_hp_jobs()
        hp_sizes_check = [job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs] + [hp_job.req]
        total_hp_check = sum(hp_sizes_check)

        if total_hp_check > 7:
            print(f"    Cannot fit HP jobs: total {total_hp_check}g > 7g (skipping preempt)")
            return

        # ========================================
        # 모든 Spot job 끄기
        # ========================================
        running_spots = node.get_running_spot_jobs()
        if running_spots:
            print(f"    Preempting {len(running_spots)} Spot jobs")

            # 먼저 job_ids 추출 (preempt_job에서 리스트가 변경되기 전에)
            job_ids = [j.job_id for j in running_spots]
            spots_to_preempt = list(running_spots)  # 복사본 생성

            # 실제 undeploy 먼저 실행
            if not self.dry_run:
                print(f"    Batch undeploying Spot jobs...{job_ids}")
                success_count, failed = deploy.undeploy_jobs_batch(job_ids)
                print(f"    Undeployed: {success_count}/{len(job_ids)} Spot jobs")

                if failed:
                    print(f"    Failed releases: {failed}")

                deploy.wait_for_pod_deletion(job_ids, max_wait_sec=60)

            # 내부 상태 업데이트 (deallocate)
            for job in spots_to_preempt:
                self.preempt_job(job, node)

        # ========================================
        # Profile 계산: 동적으로 MIG placement 제약 반영
        # ========================================
        if not self.dry_run:
            # 동적 profile 조회 (cleanup_mode로 실제 가능한 profile 확인)
            new_profile, new_hp_allocated_size = self._get_valid_hp_profile_dynamic(
                node, running_hp_jobs, hp_job.req
            )

            # "In use by another client" 에러 처리 → Hard reset
            if new_profile == "IN_USE_ERROR":
                print(f"    'In use' error detected in cleanup, triggering hard reset...")
                # 현재 프로파일 계산 (기존 HP + 새 HP)
                hp_sizes = [job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs] + [hp_job.req]
                hp_sizes.sort(reverse=True)
                target_profile = "".join(str(size) for size in hp_sizes)
                remaining = 7 - sum(hp_sizes)
                if remaining > 0:
                    target_profile += "1" * remaining

                if self._handle_in_use_error(node, target_profile):
                    # Hard reset 성공 후 HP 재스케줄링
                    print(f"    Hard reset completed, HP job {hp_job.job_id} re-queued")
                else:
                    print(f"    Hard reset failed!")
                return

            if new_profile is None:
                print(f"    No valid MIG profile for HP job {hp_job.job_id}")
                print(f"    HP job will wait until resources available")
                return

            # 새 HP job의 allocated_size 업데이트 (upgrade된 경우)
            if new_hp_allocated_size != hp_job.req:
                print(f"    HP job {hp_job.job_id} upgraded: {hp_job.req}g → {new_hp_allocated_size}g")
        else:
            # Dry-run: 시뮬레이션 기반 계산 (VALID_MIG_PROFILES 검증 포함)
            new_profile, new_hp_allocated_size = self._get_valid_hp_profile_simulated(
                node, running_hp_jobs, hp_job.req
            )

            if new_profile is None:
                print(f"    [dry-run] No valid MIG profile for HP job {hp_job.job_id}")
                print(f"    HP job will wait until resources available")
                return

            # 새 HP job의 allocated_size 업데이트 (upgrade된 경우)
            if new_hp_allocated_size != hp_job.req:
                print(f"    [dry-run] HP job {hp_job.job_id} upgraded: {hp_job.req}g → {new_hp_allocated_size}g")

        print(f"    HP Profile: {new_profile}")

        # ========================================
        # MIG reconfiguration
        # ========================================
        # 동적 profile 조회 시 cleanup이 수행되었으므로 항상 reconfigure 필요
        # (managed instances가 이미 삭제됨)
        needs_reconfig = not self.dry_run or (node.mig_profile != new_profile)

        if needs_reconfig:
            print(f"    MIG reconfiguration: {node.mig_profile} → {new_profile}")
            if not self.dry_run:
                # 기존 HP jobs가 있으면 reserved_profile로 보호
                if running_hp_jobs:
                    # allocated_size 사용 (실제 슬라이스 크기), 없으면 req 사용
                    reserved_profile = "".join(
                        str(job.allocated_size if job.allocated_size else job.req)
                        for job in running_hp_jobs
                    )
                    reserved_uuids = [job.mig_uuid for job in running_hp_jobs if job.mig_uuid]

                    # 실제 reserved 용량 체크
                    reserved_capacity = sum(job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs)
                    new_job_size = hp_job.req
                    total_needed = reserved_capacity + new_job_size

                    if total_needed > 7:
                        print(f"    Cannot fit: reserved={reserved_capacity}g + new={new_job_size}g = {total_needed}g > 7g")
                        print(f"    HP job {hp_job.job_id} will wait until resources available")
                        return

                    print(f"    Preserving existing HP jobs: {reserved_profile} (UUIDs: {reserved_uuids})")
                    result = deploy.reconfigure_mig(
                        node_name=node.name,
                        profile=new_profile,
                        reserved_profile=reserved_profile,
                        reserved_uuids=reserved_uuids if reserved_uuids else None,
                        gpu_index=node.gpu_index
                    )
                else:
                    result = deploy.reconfigure_mig(
                        node_name=node.name,
                        profile=new_profile,
                        gpu_index=node.gpu_index
                    )

                # "In use by another client" 에러 처리
                if result == "IN_USE_ERROR":
                    print(f"    'In use' error detected, triggering hard reset...")
                    # HP job도 waiting queue에 추가 (재배포 대상)
                    if hp_job not in node.hp_waiting_queue:
                        node.hp_waiting_queue.insert(0, hp_job)
                    if self._handle_in_use_error(node, new_profile):
                        # Hard reset 성공 후 HP 재스케줄링 (이미 _handle_in_use_error에서 처리됨)
                        pass
                    return

                if not result:
                    print(f"    MIG reconfiguration failed!")
                    return

            node.reconfigure(new_profile)
            print(f"    Reconfigured to {new_profile}")

            # MIG reconfig 후 dcgm-exporter 재시작
            print(f"    Restarting dcgm-exporter daemonset...")
            subprocess.run(
                ["kubectl", "rollout", "restart", "daemonset", "dcgm-exporter", "-n", "kube-system"],
                capture_output=True, text=True, timeout=30
            )
            print(f"    dcgm-exporter restarted")
        else:
            # dry_run이고 profile이 같은 경우만 여기에 도달
            print(f"    [dry-run] Profile already {new_profile}, skip reconfig")
            node.reconfigure(new_profile)  # 내부 상태는 업데이트

        # ========================================
        # HP job 배포
        # ========================================
        self._deploy_hp_job(hp_job, node)

        # ========================================
        # 남은 공간에 Spot 재계획
        # ========================================
        # Spot plan 제거 (HP가 들어왔으므로 기존 plan 무효)
        node.current_plan = None
        node.current_slot = None

        if len(node.spot_waiting_queue) >= 1:
            print(f"    Re-planning Spot jobs ({len(node.spot_waiting_queue)} waiting, HP reserved)")
            self.try_schedule_spot(node)

    def _deploy_hp_job(self, hp_job, node):
        """HP job 배포 헬퍼 함수

        Args:
            hp_job: 배포할 HP job
            node: 배포할 노드
        """
        slice_idx = node.allocate_job(hp_job)
        if slice_idx is None:
            print(f"    Failed to allocate HP job {hp_job.job_id}")
            return

        # 실제 배포 시 wall clock 시간 사용, dry_run 시 이벤트 시간 사용
        if not self.dry_run:
            actual_time = (time.time() - self.wall_start) * self.speed + self.start_time
        else:
            actual_time = self.current_time

        hp_job.start_time = actual_time

        # 남은 duration 사용 (재배포 시 처음부터 다시 시작하지 않도록)
        if hp_job.job_id in self.job_states:
            remaining_duration = self.job_states[hp_job.job_id].remaining_duration
            self.job_states[hp_job.job_id].actual_start_time = actual_time
        else:
            remaining_duration = hp_job.duration

        hp_job.end_time = actual_time + (remaining_duration * 60)

        self.running_jobs[hp_job.job_id] = hp_job
        self.job_to_node[hp_job.job_id] = node

        # 메트릭: HP 배포 카운트 증가
        self.deployed_hp_count += 1

        # Completion event 생성
        completion_event = Event(
            time=hp_job.end_time,
            event_type='completion',
            job=hp_job
        )
        heapq.heappush(self.event_queue, completion_event)

        print(f"    HP job {hp_job.job_id} deployed to slice {slice_idx}")
        print(f"       Completion scheduled at T={hp_job.end_time/60:.1f}min")
        print(f"       Node state: {node.get_slice_info()}")

        # 실제 Pod 배포
        if not self.dry_run:
            # allocated_size 사용 (실제 슬라이스 크기, req와 다를 수 있음)
            actual_gpu_size = hp_job.allocated_size if hp_job.allocated_size else hp_job.req

            # 현재 실행 중인 모든 job들의 UUID를 피함 (HP + Spot)
            avoid_uuids = []
            for job in node.get_running_hp_jobs():
                if job.mig_uuid and job.job_id != hp_job.job_id:
                    avoid_uuids.append(job.mig_uuid)
            for job in node.get_running_spot_jobs():
                if job.mig_uuid:
                    avoid_uuids.append(job.mig_uuid)

            if avoid_uuids:
                print(f"    Avoiding UUIDs (in use): {avoid_uuids}")

            # workload_type에 따라 배포 분기 (AI vs RAN)
            workload_type = getattr(hp_job, 'workload_type', 'AI')
            if workload_type == "RAN":
                # scale-up이면 launch_pattern 변경 (2C) 및 cell_group_num=2
                is_scale_up = getattr(hp_job, 'is_scale_up', False)
                launch_pattern = "F08 2C 59" if is_scale_up else "F08 1C 59"
                cell_group_num = 2 if is_scale_up else 1
                mig_uuid = deploy.deploy_ran_job(
                    job_id=hp_job.job_id,
                    node_name=node.name,
                    gpu_g=actual_gpu_size,
                    gpu_index=node.gpu_index,
                    slice_index=slice_idx,
                    avoid_uuids=avoid_uuids if avoid_uuids else None,
                    launch_pattern=launch_pattern,
                    cell_group_num=cell_group_num
                )
            else:
                mig_uuid = deploy.deploy_job(
                    job_id=hp_job.job_id,
                    node_name=node.name,
                    gpu_g=actual_gpu_size,
                    gpu_index=node.gpu_index,
                    slice_index=slice_idx,
                    avoid_uuids=avoid_uuids if avoid_uuids else None
                )
            if mig_uuid:
                hp_job.mig_uuid = mig_uuid  # UUID 저장 (MIG 보호에 사용)
            else:
                print(f"    Pod deployment failed for {hp_job.job_id}")

    def try_schedule_hp_scale_out(self, node: GPUNode, scale_out_job):
        """HP-scale-out job 스케줄링 (기존 HP와 동일한 로직, 배포만 다름)

        Args:
            node: 대상 노드
            scale_out_job: scale-out job (target_job_id로 원본 HP job 참조)
        """
        if scale_out_job not in node.hp_waiting_queue:
            return

        print(f"  HP-scale-out job {scale_out_job.job_id} ({scale_out_job.req}g) scheduling...")
        print(f"    Target service: {scale_out_job.target_job_id}")

        # ========================================
        # 최적화: 빈 슬라이스가 있으면 바로 배포
        # ========================================
        if node.can_allocate(scale_out_job.req):
            print(f"    Found available slice")
            self._deploy_hp_scale_out_job(scale_out_job, node)
            return

        # ========================================
        # Feasibility check: preempt 전에 HP 총합 확인
        # ========================================
        running_hp_jobs = node.get_running_hp_jobs()
        hp_sizes_check = [job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs] + [scale_out_job.req]
        total_hp_check = sum(hp_sizes_check)

        if total_hp_check > 7:
            print(f"    Cannot fit HP jobs: total {total_hp_check}g > 7g (skipping preempt)")
            return

        # ========================================
        # 모든 Spot job 끄기
        # ========================================
        running_spots = node.get_running_spot_jobs()
        if running_spots:
            print(f"    Preempting {len(running_spots)} Spot jobs")

            job_ids = [j.job_id for j in running_spots]
            spots_to_preempt = list(running_spots)

            if not self.dry_run:
                print(f"    Batch undeploying Spot jobs...{job_ids}")
                success_count, failed = deploy.undeploy_jobs_batch(job_ids)
                print(f"    Undeployed: {success_count}/{len(job_ids)} Spot jobs")

                if failed:
                    print(f"    Failed releases: {failed}")

                deploy.wait_for_pod_deletion(job_ids, max_wait_sec=60)

            for job in spots_to_preempt:
                self.preempt_job(job, node)

        # ========================================
        # Profile 계산: 모든 HP jobs + scale-out job 고려
        # ========================================
        # 기존 HP jobs는 allocated_size (실제 할당된 크기) 사용, scale-out job은 req 사용
        hp_sizes = [job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs] + [scale_out_job.req]
        hp_sizes.sort(reverse=True)

        hp_profile = "".join(str(size) for size in hp_sizes)
        total_hp_capacity = sum(hp_sizes)
        remaining = 7 - total_hp_capacity

        if remaining < 0:
            print(f"    Cannot fit HP jobs: total {total_hp_capacity}g > 7g")
            return

        new_profile = hp_profile + "1" * remaining
        print(f"    HP jobs: {hp_sizes} → Profile: {new_profile}")

        # ========================================
        # MIG reconfiguration
        # ========================================
        if node.mig_profile != new_profile:
            print(f"    MIG reconfiguration: {node.mig_profile} → {new_profile}")
            if not self.dry_run:
                if running_hp_jobs:
                    reserved_profile = "".join(str(job.allocated_size if job.allocated_size else job.req) for job in running_hp_jobs)
                    reserved_uuids = [job.mig_uuid for job in running_hp_jobs if job.mig_uuid]
                    print(f"    Preserving existing HP jobs: {reserved_profile} (UUIDs: {reserved_uuids})")
                    result = deploy.reconfigure_mig(
                        node_name=node.name,
                        profile=new_profile,
                        reserved_profile=reserved_profile,
                        reserved_uuids=reserved_uuids if reserved_uuids else None,
                        gpu_index=node.gpu_index
                    )
                else:
                    result = deploy.reconfigure_mig(
                        node_name=node.name,
                        profile=new_profile,
                        gpu_index=node.gpu_index
                    )

                # "In use by another client" 에러 처리
                if result == "IN_USE_ERROR":
                    print(f"    'In use' error detected, triggering hard reset...")
                    # scale-out job도 waiting queue에 추가
                    if scale_out_job not in node.hp_waiting_queue:
                        node.hp_waiting_queue.insert(0, scale_out_job)
                    if self._handle_in_use_error(node, new_profile):
                        pass
                    return

                if not result:
                    print(f"    MIG reconfiguration failed!")
                    return

            node.reconfigure(new_profile)
            print(f"    Reconfigured to {new_profile}")

            # MIG reconfig 후 dcgm-exporter 재시작
            print(f"    Restarting dcgm-exporter daemonset...")
            subprocess.run(
                ["kubectl", "rollout", "restart", "daemonset", "dcgm-exporter", "-n", "kube-system"],
                capture_output=True, text=True, timeout=30
            )
            print(f"    dcgm-exporter restarted")
        else:
            print(f"    Profile already {new_profile}, skip reconfig")

        # ========================================
        # HP-scale-out job 배포
        # ========================================
        self._deploy_hp_scale_out_job(scale_out_job, node)

        # ========================================
        # 남은 공간에 Spot 재계획
        # ========================================
        node.current_plan = None
        node.current_slot = None

        if len(node.spot_waiting_queue) >= 1:
            print(f"    Re-planning Spot jobs ({len(node.spot_waiting_queue)} waiting, HP reserved)")
            self.try_schedule_spot(node)

    def _deploy_hp_scale_out_job(self, job, node):
        """HP-scale-out job 배포 헬퍼 함수 (serviceId 연결)

        Args:
            job: 배포할 HP-scale-out job
            node: 배포할 노드
        """
        slice_idx = node.allocate_job(job)
        if slice_idx is None:
            print(f"    Failed to allocate HP-scale-out job {job.job_id}")
            return

        # 실제 배포 시 wall clock 시간 사용, dry_run 시 이벤트 시간 사용
        if not self.dry_run:
            actual_time = (time.time() - self.wall_start) * self.speed + self.start_time
        else:
            actual_time = self.current_time

        job.start_time = actual_time

        # 남은 duration 사용 (재배포 시 처음부터 다시 시작하지 않도록)
        if job.job_id in self.job_states:
            remaining_duration = self.job_states[job.job_id].remaining_duration
            self.job_states[job.job_id].actual_start_time = actual_time
        else:
            remaining_duration = job.duration

        job.end_time = actual_time + (remaining_duration * 60)

        self.running_jobs[job.job_id] = job
        self.job_to_node[job.job_id] = node

        # 메트릭: HP 배포 카운트 증가
        self.deployed_hp_count += 1

        # Completion event 생성
        completion_event = Event(
            time=job.end_time,
            event_type='completion',
            job=job
        )
        heapq.heappush(self.event_queue, completion_event)

        print(f"    HP-scale-out {job.job_id} deployed to slice {slice_idx}")
        print(f"       Target service: {job.target_job_id}")
        print(f"       Completion scheduled at T={job.end_time/60:.1f}min")
        print(f"       Node state: {node.get_slice_info()}")

        # 실제 Pod 배포 (deploy_job_replica 사용!)
        if not self.dry_run:
            # allocated_size 사용 (실제 슬라이스 크기, req와 다를 수 있음)
            actual_gpu_size = job.allocated_size if job.allocated_size else job.req

            # 현재 실행 중인 모든 job들의 UUID를 피함 (HP + Spot)
            avoid_uuids = []
            for j in node.get_running_hp_jobs():
                if j.mig_uuid and j.job_id != job.job_id:
                    avoid_uuids.append(j.mig_uuid)
            for j in node.get_running_spot_jobs():
                if j.mig_uuid:
                    avoid_uuids.append(j.mig_uuid)

            if avoid_uuids:
                print(f"    Avoiding UUIDs (in use): {avoid_uuids}")

            mig_uuid = deploy.deploy_job_replica(
                replica_id=job.job_id,
                target_job_id=job.target_job_id,  # 원본 job의 service에 연결
                node_name=node.name,
                gpu_g=actual_gpu_size,
                gpu_index=node.gpu_index,
                slice_index=slice_idx,
                avoid_uuids=avoid_uuids if avoid_uuids else None
            )
            if mig_uuid:
                job.mig_uuid = mig_uuid
            else:
                print(f"    Pod deployment failed for {job.job_id}")

    def handle_scale_in(self, job):
        """HP-scale-in 처리 (즉시 삭제)

        Args:
            job: scale-in job (job_id가 삭제할 replica의 ID)
        """
        time_str = self.format_time(self.current_time)
        print(f" [{time_str}] HP-scale-in: Removing {job.job_id}")

        target_job_id = job.job_id  # scale-in job의 id가 삭제 대상

        if target_job_id in self.running_jobs:
            running_job = self.running_jobs[target_job_id]
            node = self.job_to_node.get(target_job_id)

            # 노드에서 해제
            if node:
                node.deallocate_job(running_job)
                del self.job_to_node[target_job_id]
                print(f"    Deallocated from {node.name} GPU {node.gpu_index}")
                print(f"    Node state: {node.get_slice_info()}")

            # running jobs에서 제거
            del self.running_jobs[target_job_id]

            # Completion event 취소
            self._remove_completion_event(target_job_id)

            # 실제 Pod 삭제
            if not self.dry_run:
                deploy.undeploy_replica(target_job_id)
                deploy.wait_for_pod_deletion([target_job_id], max_wait_sec=60)

            print(f"    Removed replica {target_job_id}")

            # 노드에 대기 중인 Spot job이 있으면 재스케줄링
            if node and len(node.spot_waiting_queue) >= 1:
                print(f"    Re-planning Spot jobs ({len(node.spot_waiting_queue)} waiting)")
                self.try_schedule_spot(node)
        else:
            print(f"    Job {target_job_id} not found in running jobs")

    def handle_scale_up(self, job):
        """HP-scale-up 처리 (기존 pod 끄고 더 큰 사이즈로 재배포)

        Args:
            job: scale-up job
                - target_job_id: 기존 running job의 ID
                - req: 새로운 GPU 사이즈 (기존보다 커야 함)
        """
        time_str = self.format_time(self.current_time)
        target_job_id = job.target_job_id
        new_req = job.req

        print(f" [{time_str}] HP-scale-up: {target_job_id} → {new_req}g")

        if not target_job_id:
            print(f"    target_job_id is required for HP-scale-up")
            return

        if target_job_id not in self.running_jobs:
            print(f"    Target job {target_job_id} not found in running jobs")
            return

        running_job = self.running_jobs[target_job_id]
        node = self.job_to_node.get(target_job_id)
        old_req = running_job.req

        print(f"    Current: {old_req}g → New: {new_req}g")

        if new_req <= old_req:
            print(f"    new_req ({new_req}g) should be > old_req ({old_req}g) for scale-up")
            return

        # 1. 기존 job undeploy
        print(f"    Step 1: Undeploy existing job {target_job_id}")
        if node:
            node.deallocate_job(running_job)
            del self.job_to_node[target_job_id]
            print(f"    Deallocated from {node.name} GPU {node.gpu_index}")

        del self.running_jobs[target_job_id]
        self._remove_completion_event(target_job_id)

        if not self.dry_run:
            workload_type = getattr(running_job, 'workload_type', 'AI')
            if workload_type == "RAN":
                deploy.undeploy_ran_job(target_job_id)
            else:
                deploy.undeploy_job(target_job_id)
            deploy.wait_for_pod_deletion([target_job_id], max_wait_sec=60)

        print(f"    Undeployed {target_job_id}")

        # 2. 새로운 사이즈로 HP job 생성 및 스케줄링
        print(f"    Step 2: Re-schedule with new size {new_req}g")

        # 기존 job 정보로 새 job 생성 (동일한 job_id 사용)
        new_job = running_job
        new_job.req = new_req
        new_job.allocated_size = None  # 새로 할당받아야 함
        new_job.mig_uuid = None
        new_job.is_scale_up = True  # scale-up 표시 (launch_pattern 변경용)

        if node:
            node.hp_waiting_queue.append(new_job)
            self.try_schedule_hp(node)
        else:
            print(f"    No node found for re-scheduling")

    def handle_scale_down(self, job):
        """HP-scale-down 처리 (기존 pod 끄고 더 작은 사이즈로 재배포)

        Args:
            job: scale-down job
                - target_job_id: 기존 running job의 ID
                - req: 새로운 GPU 사이즈 (기존보다 작아야 함)
        """
        time_str = self.format_time(self.current_time)
        target_job_id = job.target_job_id
        new_req = job.req

        print(f" [{time_str}] HP-scale-down: {target_job_id} → {new_req}g")

        if not target_job_id:
            print(f"    target_job_id is required for HP-scale-down")
            return

        if target_job_id not in self.running_jobs:
            print(f"    Target job {target_job_id} not found in running jobs")
            return

        running_job = self.running_jobs[target_job_id]
        node = self.job_to_node.get(target_job_id)
        old_req = running_job.req

        print(f"    Current: {old_req}g → New: {new_req}g")

        if new_req >= old_req:
            print(f"    new_req ({new_req}g) should be < old_req ({old_req}g) for scale-down")
            return

        # 1. 기존 job undeploy
        print(f"    Step 1: Undeploy existing job {target_job_id}")
        if node:
            node.deallocate_job(running_job)
            del self.job_to_node[target_job_id]
            print(f"    Deallocated from {node.name} GPU {node.gpu_index}")

        del self.running_jobs[target_job_id]
        self._remove_completion_event(target_job_id)

        if not self.dry_run:
            workload_type = getattr(running_job, 'workload_type', 'AI')
            if workload_type == "RAN":
                deploy.undeploy_ran_job(target_job_id)
            else:
                deploy.undeploy_job(target_job_id)
            deploy.wait_for_pod_deletion([target_job_id], max_wait_sec=60)

        print(f"    Undeployed {target_job_id}")

        # 2. 새로운 사이즈로 HP job 생성 및 스케줄링
        print(f"    Step 2: Re-schedule with new size {new_req}g")

        # 기존 job 정보로 새 job 생성 (동일한 job_id 사용)
        new_job = running_job
        new_job.req = new_req
        new_job.allocated_size = None  # 새로 할당받아야 함
        new_job.mig_uuid = None

        if node:
            node.hp_waiting_queue.append(new_job)
            self.try_schedule_hp(node)
        else:
            print(f"    No node found for re-scheduling")

    def handle_slot_timeout(self, event):
        """Slot timeout 처리 (30분 경과)"""
        node = event.node
        slot_idx = event.slot_idx
        plan = event.plan

        time_str = self.format_time(self.current_time)

        # 방법 1: Plan 검증 (stale event 필터링)
        if node.current_plan is not plan:
            # 옛날 plan의 timeout → 무시!
            print(f" [{time_str}] Slot {slot_idx} timeout on {node.name} GPU {node.gpu_index} (stale plan, ignore)")
            return

        # 현재 slot이 이미 넘어갔는지 확인
        if node.current_slot != slot_idx:
            # Early transition 했음 → 무시
            print(f" [{time_str}] Slot {slot_idx} timeout on {node.name} GPU {node.gpu_index} (already at slot {node.current_slot}, ignore)")
            return

        # ⏰ 실제 timeout 발생!
        print(f"\n [{time_str}] ⏰ Slot {slot_idx} TIMEOUT on {node.name} GPU {node.gpu_index} (30 min elapsed)")

        # 현재 slot의 실행 중인 job들 확인
        slot_jobs_ids = node.current_plan["slot_jobs"].get(slot_idx, [])
        running_jobs = [
            j for j in node.get_running_spot_jobs()
            if j.job_id in slot_jobs_ids
        ]

        # 다음 slot의 profile 확인
        next_slot = slot_idx + 1
        current_profile = node.mig_profile
        chosen_cfg = node.current_plan["chosen_cfg"]
        next_profile = chosen_cfg.get(next_slot, current_profile)

        # Profile이 변경되는 경우만 preempt
        need_preempt = (current_profile != next_profile)

        # 남은 job들 preempt (profile 변경 시에만)
        if len(running_jobs) > 0:
            if need_preempt:
                print(f"    Preempting {len(running_jobs)} remaining jobs (profile change: {current_profile} → {next_profile}):")

                # 상태 변경 (batch undeploy 전)
                for job in running_jobs:
                    print(f"       - {job.job_id}")
                    self.preempt_job(job, node)

                # Batch undeploy (한 번에 모든 job 삭제)
                if not self.dry_run:
                    job_ids = [job.job_id for job in running_jobs]
                    print(f"    Batch undeploying {len(job_ids)} jobs...")
                    success_count, failed = deploy.undeploy_jobs_batch(job_ids)
                    print(f"    Undeployed: {success_count}/{len(job_ids)} jobs")

                    if failed:
                        print(f"    Failed releases: {failed}")

                    # Pod deletion 대기
                    print(f"    Waiting for pod deletion...")
                    if not deploy.wait_for_pod_deletion(job_ids, max_wait_sec=60):
                        print(f"    Warning: Some pods may still be running!")
            else:
                print(f"     {len(running_jobs)} jobs still running (profile unchanged: {current_profile})")
                print(f"       Jobs will continue in next slot: {[j.job_id for j in running_jobs]}")

        # Slot transition
        node.current_slot += 1

        # Plan 끝났는지 확인
        tmax_slot = node.current_plan.get("Tmax_slot", len(node.current_plan["chosen_cfg"]))
        if node.current_slot > tmax_slot:
            print(f"    All slots completed")
            node.current_plan = None
            node.current_slot = None

            # Plan 종료 후 대기 중인 job이 있으면 새 plan 생성
            if len(node.spot_waiting_queue) >= 1:
                print(f"    Re-planning triggered ({len(node.spot_waiting_queue)} jobs still waiting)")
                self.try_schedule_spot(node)

            # HP job 대기 중이면 스케줄링 재시도
            if len(node.hp_waiting_queue) > 0:
                print(f"    HP retry triggered ({len(node.hp_waiting_queue)} HP jobs waiting)")
                self.try_schedule_hp(node)
            return

        # 다음 slot 배포
        print(f"    → Slot {slot_idx} → {node.current_slot}")
        self._execute_current_slot(node)

        # Slot timeout마다 HP job 대기 큐 체크
        if len(node.hp_waiting_queue) > 0:
            print(f"    HP retry on slot timeout ({len(node.hp_waiting_queue)} HP jobs waiting)")
            self.try_schedule_hp(node)

    def check_slot_transition(self, node: GPUNode):
        """Slot 전환 조건 확인 (All jobs completed)"""
        if node.current_plan is None:
            return

        current_slot = node.current_slot
        slot_jobs_ids = node.current_plan["slot_jobs"].get(current_slot, [])

        # 조건: 현재 slot의 모든 job 완료됨?
        running_job_ids = {j.job_id for j in node.get_running_spot_jobs()}
        slot_jobs_remaining = [jid for jid in slot_jobs_ids if jid in running_job_ids]
        all_completed = len(slot_jobs_remaining) == 0

        if all_completed:
            time_str = self.format_time(self.current_time)
            print(f"\n [{time_str}] Slot {current_slot} → {current_slot + 1} on {node.name} GPU {node.gpu_index} (all jobs completed)")

            # Slot 전환
            node.current_slot += 1

            # Plan 범위 체크
            tmax_slot = node.current_plan.get("Tmax_slot", len(node.current_plan["chosen_cfg"]))
            if node.current_slot > tmax_slot:
                print(f"      All slots completed! Plan finished.")
                node.current_plan = None
                node.current_slot = None

                # Plan 종료 후 대기 중인 job이 있으면 새 plan 생성
                if len(node.spot_waiting_queue) >= 1:
                    print(f"      Re-planning triggered ({len(node.spot_waiting_queue)} jobs still waiting)")
                    self.try_schedule_spot(node)
                return

            # 다음 slot 배포
            self._execute_current_slot(node)

    def preempt_job(self, job, node):
        """Job preemption (중단 후 재배치 대기)

        Note: 이 함수는 상태만 변경하고, 실제 Pod undeploy는 호출하는 쪽에서 batch로 처리
        """

        print(f"    Preempting {job.job_id} from {node.name} GPU {node.gpu_index}")

        # 1. GPU 리소스 해제 (running list에서도 자동 제거됨)
        success = node.deallocate_job(job)
        if not success:
            print(f"      Failed to deallocate {job.job_id}")
            return

        # 2. MIG 관련 상태 초기화 (재배포 시 자기 자신과 충돌 방지)
        old_uuid = job.mig_uuid
        job.mig_uuid = None
        job.allocated_size = None
        job.start_time = None
        print(f"      → Cleared mig_uuid (was: {old_uuid[:20] if old_uuid else None}...)")

        # 3. Waiting queue에 다시 넣기 (재실행 대기)
        if job not in node.spot_waiting_queue:
            node.spot_waiting_queue.append(job)
            print(f"      → Re-queued for later execution")

        # 4. Job 상태 업데이트 (preemption 횟수 및 실행 시간 누적)
        if hasattr(self, 'job_states') and job.job_id in self.job_states:
            state = self.job_states[job.job_id]
            # 실행 시간 누적 (preempt 전까지 실행된 시간)
            if state.actual_start_time is not None:
                elapsed_min = (self.current_time - state.actual_start_time) / 60
                state.total_run_time += elapsed_min
            # actual_start_time 초기화 (다음 배포 시 새로 설정됨)
            state.actual_start_time = None
            state.times_preempted += 1

        # 5. Completion event 제거 필요!
        self._remove_completion_event(job.job_id)

    def _remove_completion_event(self, job_id):
        """Event queue에서 해당 job의 completion event 취소"""
        for event in self.event_queue:
            if (event.event_type == 'completion' and
                event.job is not None and
                event.job.job_id == job_id):
                event.cancelled = True  # 플래그로 표시
                print(f"      Completion event cancelled")
                break

    # -----------------------------------------
    # 헬퍼 메소드들
    # -----------------------------------------

    def _handle_in_use_error(self, node: GPUNode, new_profile: str):
        """
        'In use by another client' 에러 발생 시 hard reset 수행

        1. 모든 jobs undeploy (HP + Spot)
        2. hard_reconfigure_mig() 호출
        3. HP jobs 재배포
        4. node 상태 업데이트

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f" HANDLING 'In use by another client' ERROR")
        print(f"  Node: {node.name} GPU {node.gpu_index}")
        print(f"{'='*60}")

        # 1. 현재 실행 중인 jobs 저장
        running_hp_jobs = list(node.hp_running_jobs)
        running_spot_jobs = list(node.spot_running_jobs)
        all_running_job_ids = [j.job_id for j in running_hp_jobs + running_spot_jobs]

        print(f"  Running HP jobs: {[j.job_id for j in running_hp_jobs]}")
        print(f"  Running Spot jobs: {[j.job_id for j in running_spot_jobs]}")

        # 2. 모든 jobs undeploy
        if all_running_job_ids:
            print(f"\n  --- Undeploying ALL jobs ---")
            success_count, failed = deploy.undeploy_jobs_batch(all_running_job_ids)
            print(f"  Undeployed: {success_count}/{len(all_running_job_ids)} jobs")
            if failed:
                print(f"  Failed: {failed}")

            # Pod 삭제 대기
            deploy.wait_for_pod_deletion(all_running_job_ids, max_wait_sec=120)

        # 3. Hard reconfigure MIG
        print(f"\n  --- Hard MIG Reconfiguration ---")
        success = deploy.hard_reconfigure_mig(node.name, node.gpu_index, new_profile)
        if not success:
            print(f"  Hard reconfiguration failed!")
            return False

        # 4. Node 상태 업데이트
        node.reconfigure(new_profile)
        node.hp_running_jobs.clear()
        node.spot_running_jobs.clear()

        # 5. HP jobs 재배포
        if running_hp_jobs:
            print(f"\n  --- Re-deploying HP jobs ---")

            # 현재 시간 계산
            if not self.dry_run:
                current_time = (time.time() - self.wall_start) * self.speed + self.start_time
            else:
                current_time = self.current_time

            for hp_job in running_hp_jobs:
                # remaining_duration 업데이트 (이미 실행된 시간 차감)
                if hp_job.job_id in self.job_states and hp_job.start_time is not None:
                    state = self.job_states[hp_job.job_id]
                    elapsed_min = (current_time - hp_job.start_time) / 60
                    state.remaining_duration = max(0, state.remaining_duration - elapsed_min)
                    print(f"    {hp_job.job_id}: elapsed={elapsed_min:.1f}min, remaining={state.remaining_duration:.1f}min")

                # HP job을 waiting queue에 추가
                if hp_job not in node.hp_waiting_queue:
                    node.hp_waiting_queue.insert(0, hp_job)  # 앞에 추가 (우선순위)

                # 상태 초기화
                hp_job.mig_uuid = None
                hp_job.allocated_size = None
                hp_job.start_time = None

            # HP jobs 순서대로 배포
            for hp_job in running_hp_jobs:
                if node.can_allocate(hp_job.req):
                    self._deploy_hp_job(hp_job, node)
                    print(f"    Re-deployed HP job {hp_job.job_id}")
                else:
                    print(f"    Cannot re-deploy HP job {hp_job.job_id} (no space)")

        # 6. Spot jobs는 waiting queue에 추가 (preempt 처리)
        if running_spot_jobs:
            print(f"\n  --- Queueing Spot jobs for re-scheduling ---")

            # 현재 시간 계산 (HP 섹션에서 이미 했지만, HP가 없는 경우를 위해)
            if not self.dry_run:
                current_time = (time.time() - self.wall_start) * self.speed + self.start_time
            else:
                current_time = self.current_time

            for spot_job in running_spot_jobs:
                # remaining_duration 업데이트 (이미 실행된 시간 차감)
                if spot_job.job_id in self.job_states and spot_job.start_time is not None:
                    state = self.job_states[spot_job.job_id]
                    elapsed_min = (current_time - spot_job.start_time) / 60
                    state.remaining_duration = max(0, state.remaining_duration - elapsed_min)
                    print(f"    {spot_job.job_id}: elapsed={elapsed_min:.1f}min, remaining={state.remaining_duration:.1f}min")

                # 상태 초기화
                spot_job.mig_uuid = None
                spot_job.allocated_size = None
                spot_job.start_time = None

                # Completion event 취소
                for event in self.event_queue:
                    if event.job and event.job.job_id == spot_job.job_id and event.event_type == 'completion':
                        event.cancelled = True

                # Waiting queue에 추가
                if spot_job not in node.spot_waiting_queue:
                    node.spot_waiting_queue.append(spot_job)
                    print(f"    → {spot_job.job_id} queued for re-scheduling")

        print(f"\n{'='*60}")
        print(f"'In use' ERROR HANDLED SUCCESSFULLY")
        print(f"{'='*60}")
        return True

    def _get_valid_hp_profile_dynamic(self, node, running_hp_jobs, new_hp_req):
        """
        동적으로 가능한 HP profile 찾기 (MIG placement 제약 반영)

        1. 기존 HP jobs의 GI ID를 reserved로 설정
        2. Managed 인스턴스 삭제 (Spot jobs는 이미 preempt됨)
        3. nvidia-smi mig -lgip로 가능한 profile 조회
        4. 새 HP job을 수용할 수 있는 profile 선택

        Args:
            node: GPUNode
            running_hp_jobs: 현재 실행 중인 HP jobs
            new_hp_req: 새 HP job의 요청 크기 (g)

        Returns:
            (new_profile, new_hp_allocated_size) or (None, None)
        """
        from mig_dynamic_config import get_dynamic_configs_with_cleanup

        print(f"    [dynamic HP] Finding valid profile for new HP {new_hp_req}g...")

        # 1. 기존 HP jobs의 UUID → GI ID 매핑
        reserved_gi_ids = []
        reserved_sizes = []
        if running_hp_jobs:
            uuid_to_gi = deploy.get_uuid_to_gi_mapping(node.name, node.gpu_index)
            for hp_job in running_hp_jobs:
                if hp_job.mig_uuid and hp_job.mig_uuid in uuid_to_gi:
                    gi_id = uuid_to_gi[hp_job.mig_uuid]
                    reserved_gi_ids.append(gi_id)
                    size = hp_job.allocated_size if hp_job.allocated_size else hp_job.req
                    reserved_sizes.append(size)
                    print(f"    [dynamic HP] Reserved: HP {hp_job.job_id} → GI {gi_id} ({size}g)")

        # 2. Cleanup mode로 가능한 profile 조회 (managed 삭제됨)
        configs = get_dynamic_configs_with_cleanup(
            node_name=node.name,
            gpu_index=node.gpu_index,
            reserved_gi_ids=reserved_gi_ids,
        )

        # "In use by another client" 에러 감지
        if isinstance(configs, dict) and configs.get("IN_USE_ERROR"):
            print(f"    [dynamic HP] 'In use by another client' error detected!")
            return "IN_USE_ERROR", None

        if not configs:
            print(f"    [dynamic HP] No valid configs available")
            return None, None

        print(f"    [dynamic HP] Available configs: {list(configs.keys())}")

        # MIG placement 제약으로 인해 유효한 전체 profile 목록
        # H100/A100에서 실제로 생성 가능한 MIG 조합만 허용
        VALID_MIG_PROFILES = {
            "7",        # 1x 7g
            "43",       # 1x 4g + 1x 3g
            "421",      # 1x 4g + 1x 2g + 1x 1g
            "4111",     # 1x 4g + 3x 1g
            "3211",     # 1x 3g + 1x 2g + 2x 1g
            "31111",    # 1x 3g + 4x 1g
            "22111",    # 2x 2g + 3x 1g
            "211111",   # 1x 2g + 5x 1g
            "1111111",  # 7x 1g
            # 부분 사용 (총합 < 7)
            "4", "3", "2", "1",
            "41", "42",
            "31", "32", "311", "3111",
            "21", "22", "211", "221", "2111", "21111",
            "11", "111", "1111", "11111", "111111",
        }

        reserved_profile_str = "".join(str(s) for s in sorted(reserved_sizes, reverse=True))

        # 3. 새 HP job을 수용할 수 있는 profile 선택 (MIG placement 검증 포함)
        valid_profiles = []
        for profile_name, slot_counts in configs.items():
            # 이 profile에서 new_hp_req 이상의 슬라이스가 있는지 체크
            for size_str in ["7g", "4g", "3g", "2g", "1g"]:
                size = int(size_str[0])
                count = slot_counts.get(size_str, 0)
                if size >= new_hp_req and count > 0:
                    # reserved + profile_name 조합이 유효한지 검증
                    combined = list(reserved_profile_str) + list(profile_name)
                    combined_int = [int(c) for c in combined]
                    combined_int.sort(reverse=True)
                    combined_profile = "".join(str(s) for s in combined_int)

                    # 총 용량 및 MIG placement 검증
                    total = sum(combined_int)
                    if total > 7:
                        print(f"    [dynamic HP] Skip {profile_name}: total {total}g > 7g")
                        break

                    if not is_valid_mig_profile(combined_profile):
                        print(f"    [dynamic HP] Skip {profile_name}: '{combined_profile}' is invalid MIG profile")
                        break

                    # 유효한 조합 발견
                    waste = size - new_hp_req
                    valid_profiles.append((profile_name, size, waste, combined_profile))
                    break

        if not valid_profiles:
            print(f"    [dynamic HP] No profile can accommodate {new_hp_req}g with valid MIG placement")
            return None, None

        # 4. 낭비 최소화하는 profile 선택 (Spot job 고려)
        # Helper: remaining profile에서 Spot jobs를 몇 개 배치할 수 있는지 추정
        def estimate_spot_fit(hp_profile, full_profile):
            """HP가 차지한 후 남은 슬롯으로 Spot jobs 배치 가능 수 추정 (Greedy)"""
            # full_profile에서 hp_profile 슬롯 제거 → 남은 슬롯
            remaining = list(full_profile)
            for c in hp_profile:
                if c in remaining:
                    remaining.remove(c)

            slots = sorted([int(c) for c in remaining], reverse=True)
            spot_jobs = node.spot_waiting_queue
            if not spot_jobs:
                return 0, 0  # Spot 없으면 차이 없음

            job_sizes = sorted([j.req for j in spot_jobs], reverse=True)

            placed = 0
            total_spot_waste = 0
            for job_size in job_sizes:
                for i, slot in enumerate(slots):
                    if slot >= job_size:
                        total_spot_waste += slot - job_size
                        slots.pop(i)
                        placed += 1
                        break

            # 배치 못한 job은 패널티
            unplaced = len(job_sizes) - placed
            return placed, total_spot_waste + (unplaced * 7)  # 배치 못하면 큰 패널티

        # HP waste 1순위, Spot 배치 수 2순위 (많을수록 좋음), Spot waste 3순위
        def profile_score(item):
            profile_name, allocated_size, hp_waste, final_profile = item
            # reserved + new HP가 실제 사용하는 슬롯 (profile_name이 아닌 allocated_size)
            hp_profile = reserved_profile_str + str(allocated_size)
            spot_placed, spot_waste = estimate_spot_fit(hp_profile, final_profile)
            return (hp_waste, -spot_placed, spot_waste)

        valid_profiles.sort(key=profile_score)
        best_profile, allocated_size, waste, final_profile = valid_profiles[0]

        # 선택된 profile의 Spot 배치 정보 출력
        hp_profile = reserved_profile_str + str(allocated_size)
        spot_placed, spot_waste = estimate_spot_fit(hp_profile, final_profile)

        print(f"    [dynamic HP] Selected: profile={best_profile}, allocated={allocated_size}g, waste={waste}g")
        if node.spot_waiting_queue:
            print(f"    [dynamic HP]    Spot consideration: {spot_placed}/{len(node.spot_waiting_queue)} jobs placeable, spot_waste={spot_waste}g")
        print(f"    [dynamic HP] Final profile: {final_profile} (reserved={reserved_profile_str}, new={best_profile})")

        return final_profile, allocated_size

    def _get_valid_hp_profile_simulated(self, node, running_hp_jobs, new_hp_req):
        """
        dry_run용: 내부 상태 기반으로 유효한 HP profile 계산

        실제 GPU 쿼리 없이 VALID_MIG_PROFILES 기반으로 계산합니다.
        _get_valid_hp_profile_dynamic과 동일한 인터페이스.

        Args:
            node: GPUNode
            running_hp_jobs: 현재 실행 중인 HP jobs
            new_hp_req: 새 HP job의 요청 크기 (g)

        Returns:
            (new_profile, new_hp_allocated_size) or (None, None)
        """
        print(f"    [simulated HP] Finding valid profile for new HP {new_hp_req}g...")

        # 1. HP 크기 수집 (기존 HP + 새 HP)
        hp_sizes = [job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs]
        reserved_sizes = hp_sizes.copy()  # 기존 HP만
        hp_sizes.append(new_hp_req)
        hp_sizes.sort(reverse=True)

        total_hp = sum(hp_sizes)
        if total_hp > 7:
            print(f"    [simulated HP] Cannot fit: total {total_hp}g > 7g")
            return None, None

        remaining = 7 - total_hp
        reserved_profile_str = "".join(str(s) for s in sorted(reserved_sizes, reverse=True))

        print(f"    [simulated HP] HP sizes: {hp_sizes}, reserved: {reserved_profile_str}, remaining: {remaining}g")

        # 2. 기본 profile 생성 (HP + 남은 공간 1로 채움)
        hp_profile = "".join(str(s) for s in hp_sizes)
        full_profile = hp_profile + "1" * remaining

        # 3. 유효성 검증
        if is_valid_mig_profile(full_profile):
            print(f"    [simulated HP] Valid profile found: {full_profile} (allocated={new_hp_req}g)")
            return full_profile, new_hp_req

        print(f"    [simulated HP] Profile '{full_profile}' is invalid, trying upgrade...")

        # 4. 유효하지 않으면 → 새 HP 크기 업그레이드 시도
        #    예: "331" (invalid) → 새 HP 1g를 2g로 업그레이드 → "322" 시도
        for upgrade in range(1, 7 - new_hp_req + 1):  # 최대 7g까지
            upgraded_req = new_hp_req + upgrade
            if upgraded_req > 7:
                break

            test_sizes = [job.allocated_size if job.allocated_size else job.req for job in running_hp_jobs]
            test_sizes.append(upgraded_req)
            test_sizes.sort(reverse=True)

            test_total = sum(test_sizes)
            if test_total > 7:
                continue

            test_remaining = 7 - test_total
            test_profile = "".join(str(s) for s in test_sizes) + "1" * test_remaining

            if is_valid_mig_profile(test_profile):
                print(f"    [simulated HP] Upgraded: {new_hp_req}g → {upgraded_req}g, profile={test_profile}")
                return test_profile, upgraded_req

        print(f"    [simulated HP] No valid profile found for {new_hp_req}g")
        return None, None

    def _wait_until(self, event_time):
        """실제 시간까지 대기 (real-time 실행용)"""
        if self.dry_run:
            return  # Dry-run: 즉시 진행

        # Real deployment: 실제 시간까지 대기
        wall_elapsed = time.time() - self.wall_start
        sim_elapsed = event_time - self.start_time
        target_wall_time = sim_elapsed / self.speed

        if target_wall_time > wall_elapsed:
            time.sleep(target_wall_time - wall_elapsed)

    def format_time(self, time_seconds: float) -> str:
        """시간을 MM:SS 형식으로 변환"""
        elapsed = time_seconds - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_job_state_on_completion(self, job_id: str, current_time: float):
        """Job 완료 시 상태 업데이트"""
        if job_id in self.job_states:
            state = self.job_states[job_id]
            # 완료 시간 기록
            state.completion_time = current_time
            # 실행 시간 계산
            if state.actual_start_time is not None:
                run_time_min = (current_time - state.actual_start_time) / 60
                state.total_run_time += run_time_min
            # 완료 처리
            state.remaining_duration = 0

    def undeploy(self, job):
        """Job undeploy (Pod 삭제)"""
        if self.dry_run:
            print(f"  [dry-run] undeploy {job.job_id}")
            return True

        # workload_type에 따라 undeploy 분기 (AI vs RAN)
        workload_type = getattr(job, 'workload_type', 'AI')
        if workload_type == "RAN":
            return deploy.undeploy_ran_job(job.job_id)
        return deploy.undeploy_job(job.job_id)

    def get_metrics(self) -> dict:
        """메트릭 반환 (모든 job 기준으로 wait time 계산)

        - total_jobs: 전체 입력 job 수
        - completed_jobs: 완료된 job 수
        - started_jobs: 시작된 job 수 (actual_start_time이 있는 job)
        - avg_jct_min: 완료된 job의 평균 JCT (분)
        - avg_wait_min: 모든 job의 평균 대기 시간 (분)
          - 시작된 job: actual_start_time - submit_time
          - 대기 중인 job: current_time - submit_time
        - avg_progress: 모든 job의 평균 진행률 (%)
        """
        total_jobs = len(self.job_states)  # 전체 입력 job 수
        completed_count = len(self.completed_jobs)
        completed_job_ids = {j.job_id for j in self.completed_jobs}
        running_job_ids = set(self.running_jobs.keys())

        # 실시간 모드에서는 wall clock 기반 current_time 사용
        if not self.dry_run and hasattr(self, 'wall_start'):
            metrics_current_time = (time.time() - self.wall_start) * self.speed + self.start_time
        else:
            metrics_current_time = self.current_time

        # 모든 job의 wait time 및 progress 계산
        started_count = 0
        total_wait_time = 0.0
        total_progress = 0.0

        for job_id, state in self.job_states.items():
            original_min = state.original_duration

            if state.actual_start_time is not None:
                # 시작된 job
                started_count += 1
                wait_time = state.actual_start_time - state.submit_time

                # Progress rate 계산
                if job_id in completed_job_ids:
                    progress = 100.0
                elif job_id in running_job_ids:
                    elapsed_min = (metrics_current_time - state.actual_start_time) / 60
                    run_time = state.total_run_time + elapsed_min
                    progress = min(100.0, (run_time / original_min) * 100) if original_min > 0 else 0
                else:
                    # preempted되어 waiting 중
                    run_time = state.total_run_time
                    progress = (run_time / original_min) * 100 if original_min > 0 else 0
            else:
                # 아직 시작되지 않은 job: current_time - submit_time
                wait_time = metrics_current_time - state.submit_time
                progress = 0.0

            total_wait_time += wait_time
            total_progress += progress

        # 완료된 job의 JCT 계산
        avg_jct = self.total_jct / completed_count if completed_count > 0 else 0.0
        avg_wait = total_wait_time / total_jobs if total_jobs > 0 else 0.0
        avg_progress = total_progress / total_jobs if total_jobs > 0 else 0.0

        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_count,
            "started_jobs": started_count,
            "avg_jct_min": avg_jct / 60.0,  # 초 → 분
            "avg_wait_min": avg_wait / 60.0,
            "avg_progress": avg_progress,
        }

    def get_progress_metrics(self) -> dict:
        """테스트 진행 메트릭 계산

        Returns:
            dict: {
                'completed_count': 완료된 job 수,
                'running_count': 실행 중인 job 수,
                'waiting_count': 대기 중인 job 수,
                'total_jobs': 전체 job 수,
                'job_progress': [각 job별 진행 정보]
            }
        """
        completed_job_ids = {j.job_id for j in self.completed_jobs}
        running_job_ids = set(self.running_jobs.keys())

        # 실시간 모드에서는 wall clock 기반 current_time 사용
        if not self.dry_run and hasattr(self, 'wall_start'):
            metrics_current_time = (time.time() - self.wall_start) * self.speed + self.start_time
        else:
            metrics_current_time = self.current_time

        metrics = {
            'completed_count': len(self.completed_jobs),
            'running_count': len(self.running_jobs),
            'waiting_count': 0,
            'total_jobs': len(self.job_states),
            'job_progress': []
        }

        waiting_count = 0

        for job_id, state in self.job_states.items():
            original_min = state.original_duration  # 원래 duration (분)

            if job_id in completed_job_ids:
                # 완료된 job
                progress = 100.0
                status = "completed"
                run_time = original_min
            elif job_id in running_job_ids:
                # 실행 중인 job
                if state.actual_start_time is not None:
                    elapsed_min = (metrics_current_time - state.actual_start_time) / 60
                    run_time = state.total_run_time + elapsed_min
                else:
                    run_time = state.total_run_time
                progress = min(100.0, (run_time / original_min) * 100) if original_min > 0 else 0
                status = "running"
            else:
                # 대기 중 (preempt되어 re-queued 포함)
                run_time = state.total_run_time
                progress = (run_time / original_min) * 100 if original_min > 0 else 0
                status = "waiting"
                waiting_count += 1

            metrics['job_progress'].append({
                'job_id': job_id,
                'status': status,
                'progress': progress,
                'run_time_min': run_time,
                'duration_min': original_min,
                'preempted': state.times_preempted
            })

        metrics['waiting_count'] = waiting_count
        return metrics

    def get_detailed_metrics(self) -> dict:
        """상세 메트릭 계산 (HP/Spot 분리) - Legacy, get_paper_metrics() 사용 권장"""
        return self.get_paper_metrics()

    def get_paper_metrics(self) -> dict:
        """
        논문용 5개 메트릭 계산 (HP/Spot/Total 분리)

        메트릭 정의:
        1. Job Completion Time (JCT): 완료된 job의 submit → completion 시간 (분)
           - 완료된 job만 계산
           - JCT = completion_time - submit_time

        2. Job Queueing Time: submit → 실제 시작 시간 (분)
           - 시작된 job: actual_start_time - submit_time
           - 시작 안 된 job: current_time - submit_time (현재 대기 시간)

        3. # of Completed Jobs: 완료된 job 수

        4. Job Progression Rate: (실제 돌아간 시간 / duration) × 100 (%)
           - 완료: 100%
           - 실행 중: (total_run_time + elapsed) / duration × 100
           - 대기 중: total_run_time / duration × 100

        5. Job Processed Time: 실제 클러스터에서 돌아간 시간 (분)
           - total_run_time + (실행 중이면 current elapsed)

        Returns:
            dict: HP/Spot/Total 별 5개 메트릭 (avg, sum, min, max, std, count, raw_values)
        """
        import numpy as np

        # 현재 시간 계산 (실제 모드: wall clock, dry_run: 이벤트 큐 시간)
        if not self.dry_run and hasattr(self, 'wall_start'):
            metrics_current_time = (time.time() - self.wall_start) * self.speed + self.start_time
        else:
            metrics_current_time = self.current_time

        completed_job_ids = {j.job_id for j in self.completed_jobs}
        running_job_ids = set(self.running_jobs.keys())

        # Raw data 수집용 리스트
        hp_data = {
            'jct': [],              # 완료된 job만
            'queueing_time': [],    # 시작된 job만 (정확한 queueing time)
            'queueing_time_all': [],  # 모든 job (대기 중인 job은 current_time 기준)
            'progression_rate': [], # 모든 job
            'processed_time': [],   # 모든 job
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
            is_hp = state.job_type in ("HP", "HP-scale-out")
            data = hp_data if is_hp else spot_data
            original_min = state.original_duration

            data['total_count'] += 1

            # ============================================
            # 1. Job Completion Time (JCT) - 완료된 job만
            # ============================================
            if job_id in completed_job_ids:
                data['completed_count'] += 1
                data['started_count'] += 1

                # JCT = completion_time - submit_time
                if state.completion_time is not None:
                    jct_min = (state.completion_time - state.submit_time) / 60
                else:
                    # fallback: job.end_time 사용
                    completed_job = next((j for j in self.completed_jobs if j.job_id == job_id), None)
                    if completed_job and completed_job.end_time is not None:
                        jct_min = (completed_job.end_time - state.submit_time) / 60
                    else:
                        jct_min = 0.0
                data['jct'].append(jct_min)

            # ============================================
            # 2. Job Queueing Time
            # ============================================
            if state.actual_start_time is not None:
                # 시작된 job: 정확한 queueing time
                queue_time_min = (state.actual_start_time - state.submit_time) / 60
                data['queueing_time'].append(queue_time_min)
                data['queueing_time_all'].append(queue_time_min)

                if job_id not in completed_job_ids:
                    data['started_count'] += 1
                    if job_id in running_job_ids:
                        data['running_count'] += 1
            else:
                # 시작 안 된 job: 현재까지 대기 시간
                queue_time_min = (metrics_current_time - state.submit_time) / 60
                data['queueing_time_all'].append(queue_time_min)
                data['pending_count'] += 1

            # ============================================
            # 3. Job Progression Rate (%)
            # ============================================
            if job_id in completed_job_ids:
                progress = 100.0
                run_time = original_min
            elif job_id in running_job_ids and state.actual_start_time is not None:
                # 실행 중: total_run_time + 현재 session elapsed
                elapsed_min = (metrics_current_time - state.actual_start_time) / 60
                run_time = state.total_run_time + elapsed_min
                progress = min(100.0, (run_time / original_min) * 100) if original_min > 0 else 0.0
            else:
                # 대기 중 (preempt되어 re-queued 포함)
                run_time = state.total_run_time
                progress = (run_time / original_min) * 100 if original_min > 0 else 0.0

            data['progression_rate'].append(progress)

            # ============================================
            # 4. Job Processed Time (분)
            # ============================================
            data['processed_time'].append(run_time)

        def calc_stats(values: list) -> dict:
            """통계값 계산 (avg, sum, min, max, std, count)"""
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
                # 1. Job Completion Time
                'jct': calc_stats(data['jct']),

                # 2. Job Queueing Time
                'queueing_time': calc_stats(data['queueing_time']),  # 시작된 job만
                'queueing_time_all': calc_stats(data['queueing_time_all']),  # 모든 job

                # 3. # of Completed Jobs
                'completed_count': data['completed_count'],

                # 4. Job Progression Rate
                'progression_rate': calc_stats(data['progression_rate']),

                # 5. Job Processed Time
                'processed_time': calc_stats(data['processed_time']),

                # 추가 카운트 정보
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
                'current_time': metrics_current_time,
                'simulation_elapsed_sec': metrics_current_time - self.start_time,
                'deployed_hp_count': self.deployed_hp_count,
                'deployed_spot_count': self.deployed_spot_count,
            }
        }

    def get_paper_metrics_summary(self) -> dict:
        """
        논문 테이블용 요약 메트릭 (raw 데이터 제외, 핵심 통계만)

        Returns:
            dict: {
                'hp': {
                    'jct_avg', 'jct_std',
                    'queueing_time_avg', 'queueing_time_std',
                    'completed_count',
                    'progression_rate_avg',
                    'processed_time_avg', 'processed_time_sum'
                },
                'spot': {...},
                'total': {...}
            }
        """
        full_metrics = self.get_paper_metrics()

        def summarize(m: dict) -> dict:
            return {
                # 1. JCT
                'jct_avg': m['jct']['avg'],
                'jct_std': m['jct']['std'],
                'jct_min': m['jct']['min'],
                'jct_max': m['jct']['max'],
                'jct_median': m['jct']['median'],

                # 2. Queueing Time (시작된 job 기준)
                'queueing_time_avg': m['queueing_time']['avg'],
                'queueing_time_std': m['queueing_time']['std'],
                'queueing_time_min': m['queueing_time']['min'],
                'queueing_time_max': m['queueing_time']['max'],
                'queueing_time_median': m['queueing_time']['median'],

                # 3. Completed Count
                'completed_count': m['completed_count'],

                # 4. Progression Rate
                'progression_rate_avg': m['progression_rate']['avg'],
                'progression_rate_std': m['progression_rate']['std'],
                'progression_rate_min': m['progression_rate']['min'],
                'progression_rate_max': m['progression_rate']['max'],

                # 5. Processed Time
                'processed_time_avg': m['processed_time']['avg'],
                'processed_time_sum': m['processed_time']['sum'],
                'processed_time_std': m['processed_time']['std'],

                # 추가 정보
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

        # 헤더
        print(f"{'Metric':<35} {'HP':>12} {'Spot':>12} {'Total':>12}")
        print("-" * 75)

        # 1. Job Completion Time
        print(f"{'1. JCT (min) - Completed Jobs Only':<35}")
        print(f"{'   avg':<35} {m['hp']['jct_avg']:>12.2f} {m['spot']['jct_avg']:>12.2f} {m['total']['jct_avg']:>12.2f}")
        print(f"{'   std':<35} {m['hp']['jct_std']:>12.2f} {m['spot']['jct_std']:>12.2f} {m['total']['jct_std']:>12.2f}")
        print(f"{'   min':<35} {m['hp']['jct_min']:>12.2f} {m['spot']['jct_min']:>12.2f} {m['total']['jct_min']:>12.2f}")
        print(f"{'   max':<35} {m['hp']['jct_max']:>12.2f} {m['spot']['jct_max']:>12.2f} {m['total']['jct_max']:>12.2f}")
        print(f"{'   median':<35} {m['hp']['jct_median']:>12.2f} {m['spot']['jct_median']:>12.2f} {m['total']['jct_median']:>12.2f}")
        print()

        # 2. Job Queueing Time
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

        # Metadata
        print("-" * 75)
        sim_elapsed_min = m['metadata']['simulation_elapsed_sec'] / 60
        print(f"Simulation Time: {sim_elapsed_min:.2f} min")
        print(f"Deployed: HP={m['metadata']['deployed_hp_count']}, Spot={m['metadata']['deployed_spot_count']}")
        print()

    def print_progress_metrics(self):
        """진행 메트릭 출력"""
        metrics = self.get_progress_metrics()
        detailed = self.get_detailed_metrics()

        print()
        print("=" * 60)
        print("Progress Metrics")
        print("=" * 60)

        total = metrics['total_jobs']
        completed = metrics['completed_count']
        running = metrics['running_count']
        waiting = metrics['waiting_count']

        completion_rate = (completed / total * 100) if total > 0 else 0

        print(f"  Completed: {completed} / {total} jobs ({completion_rate:.1f}%)")
        print(f"  Running:   {running}")
        print(f"  Waiting:   {waiting}")
        print()

        # ========================================
        # 상세 메트릭 (HP/Spot 분리)
        # ========================================
        print("-" * 60)
        print("Detailed Metrics (HP / Spot)")
        print("-" * 60)

        hp = detailed['hp']
        spot = detailed['spot']
        total_m = detailed['total']

        print(f"  {'Metric':<25} {'HP':>12} {'Spot':>12} {'Total':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Deployed Jobs':<25} {hp['deployed_count']:>12} {spot['deployed_count']:>12} {total_m['deployed_count']:>12}")
        print(f"  {'Completed Jobs':<25} {hp['completed_count']:>12} {spot['completed_count']:>12} {total_m['completed_count']:>12}")
        print(f"  {'Avg JCT (min)':<25} {hp['avg_jct_min']:>12.2f} {spot['avg_jct_min']:>12.2f} {'-':>12}")
        print(f"  {'Avg Queue Time (min)':<25} {hp['avg_queue_time_min']:>12.2f} {spot['avg_queue_time_min']:>12.2f} {'-':>12}")
        print(f"  {'Avg Progress Rate (%)':<25} {hp['avg_progress_rate']:>12.1f} {spot['avg_progress_rate']:>12.1f} {total_m['avg_progress_rate']:>12.1f}")
        print()

        # Job별 진행률 (상태별로 정렬)
        job_progress = sorted(metrics['job_progress'],
                              key=lambda x: (x['status'] != 'running',
                                           x['status'] != 'waiting',
                                           -x['progress']))

        print("Job Progress:")
        for job in job_progress[:20]:  # 상위 20개만 출력
            preempt_note = f" [preempted {job['preempted']}x]" if job['preempted'] > 0 else ""
            print(f"  {job['job_id']:25s}: {job['progress']:5.1f}% ({job['status']:10s}) "
                  f"- {job['run_time_min']:.1f}/{job['duration_min']:.1f} min{preempt_note}")

        if len(job_progress) > 20:
            print(f"  ... and {len(job_progress) - 20} more jobs")
        print()

    def run(self, max_time: Optional[float] = None):

        # run() 하는 과정에서 시간을 측정
        self.wall_start = time.time()

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            # Cancelled event는 skip (preempt된 job의 completion event)
            if event.cancelled:
                continue

            # 테스트 시간 넘어가면 테스트 중단
            if max_time is not None and self.current_time > max_time:
                print("max time reached")
                break

            # 실제 trace 시간까지 대기
            self._wait_until(event.time)
            self.current_time = max(self.current_time, event.time)

            if event.event_type == 'arrival':
                self.handle_arrival(event.job)

            elif event.event_type == 'completion':
                self.handle_completion(event.job)

            elif event.event_type == 'slot_timeout':
                self.handle_slot_timeout(event)

        return self.get_metrics()