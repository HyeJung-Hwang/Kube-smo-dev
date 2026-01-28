"""
Scheduler Helper Functions - 공통 유틸리티

Backend와 experiments 모두에서 사용하는 스케줄러 헬퍼 함수들을 정의합니다.
수정이 필요하면 이 파일을 수정하세요.
"""
from typing import Dict, Optional


def merge_profiles(profile1: str, profile2: str) -> str:
    """
    두 MIG profile을 합쳐서 큰 것부터 정렬

    Args:
        profile1: 첫 번째 profile (예: "3")
        profile2: 두 번째 profile (예: "211")

    Returns:
        합쳐진 profile, 큰 순서로 정렬 (예: "3211")

    Examples:
        >>> merge_profiles("3", "211")
        "3211"
        >>> merge_profiles("", "211")
        "211"
        >>> merge_profiles("32", "")
        "32"
    """
    if not profile1:
        return profile2
    if not profile2:
        return profile1

    sizes = [int(c) for c in profile1 + profile2]
    sizes.sort(reverse=True)
    return ''.join(str(s) for s in sizes)


def calculate_available_capacity(total_capacity: int, reserved_capacity: int) -> str:
    """
    전체 용량에서 예약된 용량을 뺀 나머지를 init_config 문자열로 변환

    Args:
        total_capacity: 전체 GPU 용량 (예: 7)
        reserved_capacity: HP jobs가 사용 중인 용량 (예: 3)

    Returns:
        Available capacity를 나타내는 init_config (예: "1111" for 4g)

    Examples:
        >>> calculate_available_capacity(7, 3)
        "1111"
        >>> calculate_available_capacity(7, 7)
        ""
        >>> calculate_available_capacity(7, 0)
        "1111111"
    """
    available = total_capacity - reserved_capacity
    if available <= 0:
        return ""
    return "1" * available


def extract_initial_profile_from_bin_packing(bin_packing_result: Dict) -> Optional[str]:
    """
    Bin packing 결과에서 초기 MIG profile 추출

    Args:
        bin_packing_result: Bin packing 결과 dict
            Expected format: {
                "chosen_cfg": {1: "3211", 2: ..., ...},
                "avg_jct": float,
                ...
            }

    Returns:
        초기 MIG profile 문자열 (예: "3211"), 없으면 None

    Examples:
        >>> extract_initial_profile_from_bin_packing({"chosen_cfg": {1: "3211"}})
        "3211"
        >>> extract_initial_profile_from_bin_packing({})
        None
        >>> extract_initial_profile_from_bin_packing({"chosen_cfg": {}})
        None
    """
    if not bin_packing_result or "chosen_cfg" not in bin_packing_result:
        return None

    chosen_cfg = bin_packing_result["chosen_cfg"]
    if not chosen_cfg:
        return None

    # 첫 번째 slot의 configuration 반환
    return chosen_cfg.get(1, None)


def calculate_reserved_profile(hp_jobs: list, req_key: str = "req") -> tuple[int, str]:
    """
    HP jobs로부터 reserved capacity와 profile 계산

    Args:
        hp_jobs: HP job 목록 (각 job은 req 또는 size 필드를 가짐)
        req_key: requirement를 나타내는 키 ("req" 또는 "size")

    Returns:
        (reserved_capacity, reserved_profile) 튜플
        - reserved_capacity: HP jobs가 사용하는 총 용량
        - reserved_profile: HP jobs의 MIG profile 문자열 (큰 순서로 정렬)

    Examples:
        >>> calculate_reserved_profile([{"req": 2}, {"req": 3}])
        (5, "32")
        >>> calculate_reserved_profile([])
        (0, "")
        >>> calculate_reserved_profile([{"size": 1}, {"size": 1}], req_key="size")
        (2, "11")
    """
    if not hp_jobs:
        return 0, ""

    # req 또는 size 추출
    if isinstance(hp_jobs[0], dict):
        capacities = [job.get(req_key, 0) for job in hp_jobs]
    else:
        # Job 객체인 경우
        capacities = [getattr(job, req_key, 0) for job in hp_jobs]

    reserved_capacity = sum(capacities)
    reserved_profile_list = sorted(capacities, reverse=True)
    reserved_profile = ''.join(str(s) for s in reserved_profile_list)

    return reserved_capacity, reserved_profile
