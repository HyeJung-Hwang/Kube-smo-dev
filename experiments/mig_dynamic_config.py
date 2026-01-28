#!/usr/bin/env python3
"""
MIG Dynamic Config Module

nvidia-smi CLI를 통해 현재 GPU 상태에서 사용 가능한 MIG 프로파일 조합을 동적으로 계산합니다.

사용 시나리오:
- Reserved 없음 (HP job 없음): 정적 딕셔너리 사용 (bin_pack.py의 available_MIG_profile_per_binpacking_profile)
- Reserved 있음 (HP job 있음): 이 모듈로 동적 조회

핵심 함수:
- get_dynamic_configs(): nvidia-smi로 현재 상태에서 가능한 MIG 조합 조회
- build_configs_for_ilp(): ILP 솔버가 기대하는 형식으로 변환
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict


# ============================================================
# Data Classes
# ============================================================

@dataclass(frozen=True)
class MigProfile:
    """nvidia-smi에서 파싱한 MIG 프로파일 정보"""
    name: str           # "MIG 2g.24gb"
    profile_id: int     # nvidia-smi에서 사용하는 ID
    free: int           # 현재 생성 가능한 인스턴스 수
    total: int          # 최대 인스턴스 수
    g: int              # GPU 슬라이스 수 (1, 2, 3, 4, 7)


# ============================================================
# nvidia-smi Command Execution
# ============================================================

def run_cmd_local(cmd: List[str]) -> str:
    """
    로컬에서 nvidia-smi 명령어 실행

    Args:
        cmd: 실행할 명령어 리스트

    Returns:
        stdout 문자열
    """
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=30,
        )
        return p.stdout
    except subprocess.CalledProcessError as e:
        msg = (
            f"Command failed: {' '.join(cmd)}\n"
            f"exit_code={e.returncode}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}\n"
        )
        raise RuntimeError(msg)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out: {' '.join(cmd)}")


def run_cmd_on_node(node_name: str, cmd: str, gpu_index: int = 0) -> str:
    """
    원격 노드에서 nvidia-smi 명령어 실행 (DaemonSet 활용)

    realtime_deploy.py의 run_cmd_on_node를 재사용합니다.

    Args:
        node_name: 노드 이름
        cmd: 실행할 명령어 문자열
        gpu_index: GPU 인덱스

    Returns:
        stdout 문자열
    """
    try:
        # realtime_deploy.py의 함수 재사용
        import realtime_deploy as deploy
        result = deploy.run_cmd_on_node(node_name, cmd)

        if result.get("ok", False):
            return result.get("stdout", "")
        else:
            error_msg = result.get("error", result.get("stderr", "Unknown error"))
            raise RuntimeError(f"Remote command failed: {error_msg}")

    except ImportError as e:
        raise RuntimeError(f"Failed to import realtime_deploy: {e}")


# ============================================================
# nvidia-smi Output Parsing
# ============================================================

# nvidia-smi mig -lgip 출력 파싱용 정규식
# 예: |   0  MIG 2g.24gb          14     2 /     2 |
_LGIP_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s+MIG\s+([0-9]+g\.[0-9]+gb(?:\+me)?)\s+(\d+)\s+(\d+)\/(\d+)\s+",
    re.IGNORECASE,
)

# nvidia-smi mig -lgi 출력 파싱용 정규식
# 예: |   0        3g.24gb          9      1    0:4 |
_LGI_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s+MIG\s+([0-9]+g\.[0-9]+gb(?:\+me)?)\s+(\d+)\s+(\d+)\s+(\S+)\s+\|\s*$",
    re.IGNORECASE,
)


def parse_g(compact: str) -> int:
    """
    MIG 프로파일 이름에서 슬라이스 크기 추출

    Args:
        compact: "2g.24gb" 형식의 문자열

    Returns:
        슬라이스 크기 (1, 2, 3, 4, 7)

    Examples:
        >>> parse_g("2g.24gb")
        2
        >>> parse_g("1g.12gb+me")
        1
    """
    m = re.match(r"^(\d+)g\.", compact.strip())
    if not m:
        raise ValueError(f"Cannot parse g from: {compact}")
    return int(m.group(1))


def parse_lgip(text: str) -> Dict[str, MigProfile]:
    """
    nvidia-smi mig -lgip 출력 파싱

    Args:
        text: nvidia-smi mig -lgip 출력 문자열

    Returns:
        {profile_name: MigProfile} 딕셔너리

    Example output:
        +-------+-------------+--------+------+------+
        |  GPU  |    Name     | ProfID | Free | Total|
        +-------+-------------+--------+------+------+
        |   0   | MIG 1g.12gb |   19   |  7   |  7   |
        |   0   | MIG 2g.24gb |   14   |  3   |  3   |
        |   0   | MIG 3g.24gb |    9   |  2   |  2   |
        +-------+-------------+--------+------+------+
    """
    profiles: Dict[str, MigProfile] = {}
    for line in text.splitlines():
        m = _LGIP_ROW_RE.match(line)
        if not m:
            continue
        compact = m.group(2)  # e.g. "2g.24gb"
        pid = int(m.group(3))
        free = int(m.group(4))
        total = int(m.group(5))
        name = f"MIG {compact}"
        profiles[name] = MigProfile(
            name=name,
            profile_id=pid,
            free=free,
            total=total,
            g=parse_g(compact),
        )
    return profiles


def parse_lgi(text: str) -> List[Tuple[str, int, str]]:
    """
    nvidia-smi mig -lgi 출력 파싱 (현재 생성된 인스턴스 목록)

    Args:
        text: nvidia-smi mig -lgi 출력 문자열

    Returns:
        [(profile_name, profile_id, placement), ...] 리스트
    """
    used: List[Tuple[str, int, str]] = []
    for line in text.splitlines():
        m = _LGI_ROW_RE.match(line)
        if not m:
            continue
        compact = m.group(2)      # e.g. "1g.12gb"
        pid = int(m.group(3))
        placement = m.group(5)    # e.g. "0:1"
        used.append((f"MIG {compact}", pid, placement))
    return used


# ============================================================
# Combination Generation (Backtracking)
# ============================================================

def build_combos(
    profiles: Dict[str, MigProfile],
    remaining_slices: int,
    exclude_names: Optional[Iterable[str]] = None,
) -> List[List[str]]:
    """
    남은 슬라이스를 채우는 모든 MIG 프로파일 조합 생성 (백트래킹)

    Args:
        profiles: nvidia-smi에서 파싱한 프로파일 딕셔너리
        remaining_slices: 남은 슬라이스 수 (7 - HP가 사용 중인 슬라이스)
        exclude_names: 제외할 프로파일 이름 (예: "MIG 1g.12gb+me")

    Returns:
        가능한 조합 리스트. 각 조합은 프로파일 이름 리스트
        예: [["MIG 3g.24gb", "MIG 2g.24gb", "MIG 1g.12gb", "MIG 1g.12gb"], ...]

    Note:
        - free > 0인 프로파일만 사용
        - 큰 슬라이스부터 배치 (greedy ordering for efficiency)
    """
    excluded = set(exclude_names) if exclude_names else set()

    # 사용 가능한 프로파일 필터링 (free > 0, 제외 목록 아님)
    candidates = [
        p for p in profiles.values()
        if p.free > 0 and p.name not in excluded
    ]
    # 큰 슬라이스부터 정렬 (효율적인 백트래킹을 위해)
    candidates.sort(key=lambda p: (-p.g, p.name))

    results: List[List[str]] = []

    def backtrack(i: int, slices_left: int, cur: List[str]) -> None:
        """백트래킹으로 모든 조합 탐색"""
        if slices_left == 0:
            results.append(cur.copy())
            return
        if i >= len(candidates):
            return

        p = candidates[i]
        # 이 프로파일로 채울 수 있는 최대 개수
        max_take = min(p.free, slices_left // p.g)

        # k개 사용하는 경우 (k = max_take, ..., 0)
        for k in range(max_take, -1, -1):
            if k:
                cur.extend([p.name] * k)
            backtrack(i + 1, slices_left - k * p.g, cur)
            if k:
                del cur[-k:]

    backtrack(0, remaining_slices, [])
    return results


# ============================================================
# ILP Format Conversion
# ============================================================

def combo_to_profile_string(combo: List[str]) -> str:
    """
    MIG 프로파일 조합을 프로파일 문자열로 변환

    Args:
        combo: ["MIG 3g.24gb", "MIG 2g.24gb", "MIG 1g.12gb", "MIG 1g.12gb"]

    Returns:
        "3211" (큰 순서로 정렬)

    Examples:
        >>> combo_to_profile_string(["MIG 1g.12gb", "MIG 3g.24gb", "MIG 1g.12gb"])
        "311"
    """
    sizes = []
    for name in combo:
        # "MIG 2g.24gb" → 2
        compact = name.replace("MIG ", "")
        sizes.append(parse_g(compact))

    # 큰 순서로 정렬
    sizes.sort(reverse=True)
    return "".join(str(s) for s in sizes)


def combo_to_slot_counts(combo: List[str]) -> Dict[str, int]:
    """
    MIG 프로파일 조합을 ILP 슬롯 카운트 형식으로 변환

    Args:
        combo: ["MIG 3g.24gb", "MIG 2g.24gb", "MIG 1g.12gb", "MIG 1g.12gb"]

    Returns:
        {"1g": 2, "2g": 1, "3g": 1, "4g": 0, "7g": 0}

    Note:
        ILP 솔버 (bin.py)가 기대하는 형식
    """
    counts = {"1g": 0, "2g": 0, "3g": 0, "4g": 0, "7g": 0}

    for name in combo:
        compact = name.replace("MIG ", "")
        g = parse_g(compact)
        key = f"{g}g"
        if key in counts:
            counts[key] += 1

    return counts


def build_configs_for_ilp(
    profiles: Dict[str, MigProfile],
    remaining_slices: int,
    exclude_names: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    ILP 솔버용 MIG 구성 딕셔너리 생성

    Args:
        profiles: nvidia-smi에서 파싱한 프로파일 딕셔너리
        remaining_slices: 남은 슬라이스 수
        exclude_names: 제외할 프로파일 이름

    Returns:
        {
            "3211": {"1g": 2, "2g": 1, "3g": 1, "4g": 0, "7g": 0},
            "322":  {"1g": 0, "2g": 2, "3g": 1, "4g": 0, "7g": 0},
            ...
        }

    Note:
        bin_pack.py의 available_MIG_profile_per_binpacking_profile[init_config]와
        동일한 형식
    """
    combos = build_combos(profiles, remaining_slices, exclude_names)

    configs: Dict[str, Dict[str, int]] = {}

    for combo in combos:
        if not combo:
            continue

        profile_str = combo_to_profile_string(combo)

        # MIG placement 제약 검증: 유효하지 않은 profile 제외
        if not is_valid_mig_profile(profile_str):
            continue

        slot_counts = combo_to_slot_counts(combo)

        # 중복 체크 (같은 profile_str이 다른 combo에서 나올 수 있음)
        if profile_str not in configs:
            configs[profile_str] = slot_counts

    return configs


# MIG placement 제약으로 유효한 profile 목록
# H100/A100에서 실제로 생성 가능한 조합만 (3+3+1 같은 조합은 불가)
VALID_MIG_PROFILES = {
    # === 전체 사용 (총합 = 7) ===
    "7",        # 1x 7g
    "43",       # 1x 4g + 1x 3g
    "421",      # 1x 4g + 1x 2g + 1x 1g
    "4111",     # 1x 4g + 3x 1g
    "3211",     # 1x 3g + 1x 2g + 2x 1g
    "31111",    # 1x 3g + 4x 1g
    "22111",    # 2x 2g + 3x 1g
    "211111",   # 1x 2g + 5x 1g
    "1111111",  # 7x 1g
    # === 부분 사용 (remaining space) ===
    # 4g remaining
    "4", "31", "22", "211", "1111",
    # 3g remaining
    "3", "21", "111",
    # 2g remaining
    "2", "11",
    # 1g remaining
    "1",
    # 5g remaining
    "41", "32", "311", "221", "2111", "11111",
    # 6g remaining
    "42", "33", "321", "3111", "222", "2211", "21111", "111111",
    # 추가 조합 (HP + Spot 혼합)
    "2221",     # 2+2+2+1 = 7
}

# 알려진 무효 profile (placement 충돌)
INVALID_MIG_PROFILES = {
    "331",      # 3+3+1: 3g 두 개 배치 후 1g 불가
    "322",      # 3+2+2: placement 충돌
    "3221",     # 3+2+2+1: placement 충돌
    "3311",     # 3+3+1+1: placement 충돌 (총합 8이라 어차피 불가)
}


def is_valid_mig_profile(profile_str: str) -> bool:
    """
    MIG profile이 placement 제약을 만족하는지 검사

    Args:
        profile_str: "3211", "331" 등의 profile 문자열

    Returns:
        유효하면 True, 무효면 False
    """
    # 명시적으로 무효인 profile
    if profile_str in INVALID_MIG_PROFILES:
        return False

    # 명시적으로 유효한 profile
    if profile_str in VALID_MIG_PROFILES:
        return True

    # 알 수 없는 profile: 보수적으로 무효 처리
    # (필요시 VALID_MIG_PROFILES에 추가)
    total = sum(int(c) for c in profile_str)
    if total > 7:
        return False

    # 3g가 2개 이상이면 placement 충돌 가능성 높음
    count_3 = profile_str.count('3')
    if count_3 >= 2:
        return False

    # 그 외는 일단 유효하다고 가정 (경고 출력)
    print(f"    [MIG] Unknown profile '{profile_str}' - assuming valid (add to VALID_MIG_PROFILES if needed)")
    return True


# ============================================================
# Main API Functions
# ============================================================

# 제외할 프로파일 (특수 프로파일)
DEFAULT_EXCLUDE_PROFILES = [
    "MIG 1g.12gb+me",  # Media Engine 포함
    "MIG 1g.24gb",     # 특수 메모리 구성
]


def get_dynamic_configs(
    node_name: Optional[str] = None,
    gpu_index: int = 0,
    reserved_slices: int = 0,
    exclude_profiles: Optional[List[str]] = None,
    use_total: bool = True,  # True: total 사용 (삭제 후 시나리오), False: free 사용
) -> Dict[str, Dict[str, int]]:
    """
    nvidia-smi를 통해 현재 GPU 상태에서 사용 가능한 MIG 구성 조회

    Args:
        node_name: 노드 이름 (None이면 로컬)
        gpu_index: GPU 인덱스
        reserved_slices: HP job이 사용 중인 슬라이스 수
        exclude_profiles: 제외할 프로파일 이름 리스트
        use_total: True면 total 값 사용 (managed 삭제 후 시나리오)
                   False면 free 값 사용 (현재 상태에서 추가 시나리오)

    Returns:
        ILP 솔버용 MIG 구성 딕셔너리

    Example:
        >>> # HP가 3g 사용 중 → 4g 남음
        >>> configs = get_dynamic_configs(reserved_slices=3)
        >>> print(configs)
        {
            "4":    {"1g": 0, "2g": 0, "3g": 0, "4g": 1, "7g": 0},
            "31":   {"1g": 1, "2g": 0, "3g": 1, "4g": 0, "7g": 0},
            "22":   {"1g": 0, "2g": 2, "3g": 0, "4g": 0, "7g": 0},
            "211":  {"1g": 2, "2g": 1, "3g": 0, "4g": 0, "7g": 0},
            "1111": {"1g": 4, "2g": 0, "3g": 0, "4g": 0, "7g": 0},
        }
    """
    if exclude_profiles is None:
        exclude_profiles = DEFAULT_EXCLUDE_PROFILES

    # nvidia-smi mig -lgip 명령어 구성
    cmd_str = f"nvidia-smi mig -lgip -i {gpu_index}"

    try:
        if node_name:
            # 원격 노드에서 실행 (DaemonSet 활용)
            lgip_output = run_cmd_on_node(node_name, cmd_str, gpu_index)
        else:
            # 로컬에서 실행
            cmd = ["nvidia-smi", "mig", "-lgip", "-i", str(gpu_index)]
            lgip_output = run_cmd_local(cmd)
    except RuntimeError as e:
        print(f"Warning: Failed to run nvidia-smi: {e}")
        return {}

    profiles = parse_lgip(lgip_output)

    if not profiles:
        print("Warning: No MIG profiles found from nvidia-smi")
        return {}

    # use_total=True면 free를 total로 대체 (삭제 후 시나리오 시뮬레이션)
    if use_total:
        profiles = {
            name: MigProfile(
                name=p.name,
                profile_id=p.profile_id,
                free=p.total,  # free를 total로 대체
                total=p.total,
                g=p.g,
            )
            for name, p in profiles.items()
        }

    # 총 슬라이스 계산 (보통 7)
    total_slices = infer_total_slices(profiles)
    remaining_slices = max(0, total_slices - reserved_slices)

    if remaining_slices == 0:
        return {}

    # ILP용 구성 생성
    configs = build_configs_for_ilp(profiles, remaining_slices, exclude_profiles)

    return configs


def infer_total_slices(profiles: Dict[str, MigProfile]) -> int:
    """
    GPU의 총 슬라이스 수 추론

    Args:
        profiles: nvidia-smi에서 파싱한 프로파일 딕셔너리

    Returns:
        총 슬라이스 수 (보통 7)
    """
    # 1g 프로파일의 total 값이 가장 정확
    one_g_totals = [p.total for p in profiles.values() if p.g == 1]
    if one_g_totals:
        return max(one_g_totals)

    # fallback: g * total 최댓값
    return max(p.g * p.total for p in profiles.values())


def get_current_mig_instances(
    node_name: Optional[str] = None,
    gpu_index: int = 0,
) -> List[Tuple[str, int, str]]:
    """
    현재 생성된 MIG 인스턴스 목록 조회

    Args:
        node_name: 노드 이름 (None이면 로컬)
        gpu_index: GPU 인덱스

    Returns:
        [(profile_name, profile_id, placement), ...] 리스트
    """
    cmd_str = f"nvidia-smi mig -lgi -i {gpu_index}"

    try:
        if node_name:
            # 원격 노드에서 실행 (DaemonSet 활용)
            lgi_output = run_cmd_on_node(node_name, cmd_str, gpu_index)
        else:
            # 로컬에서 실행
            cmd = ["nvidia-smi", "mig", "-lgi", "-i", str(gpu_index)]
            lgi_output = run_cmd_local(cmd)
    except RuntimeError as e:
        print(f"Warning: Failed to run nvidia-smi: {e}")
        return []

    return parse_lgi(lgi_output)


def get_dynamic_configs_with_cleanup(
    node_name: str,
    gpu_index: int = 0,
    reserved_gi_ids: Optional[List[int]] = None,
    exclude_profiles: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Managed MIG 인스턴스를 실제로 삭제한 후 정확한 프로파일 조회

    주의: 이 함수는 실제로 MIG 인스턴스를 삭제합니다!
    Spot job은 미리 preempt되어야 합니다.

    Args:
        node_name: 노드 이름 (필수 - 원격 실행)
        gpu_index: GPU 인덱스
        reserved_gi_ids: 삭제하지 않을 GI ID 리스트 (HP job이 사용 중인 인스턴스)
        exclude_profiles: 제외할 프로파일 이름 리스트

    Returns:
        ILP 솔버용 MIG 구성 딕셔너리 (placement 제약 반영)

    Flow:
        1. 현재 MIG 인스턴스 조회 (-lgi)
        2. Reserved 제외한 managed 인스턴스 식별
        3. Managed 인스턴스 삭제 (dci, dgi)
        4. -lgip 조회 (정확한 free 값!)
        5. 조합 계산 및 반환
    """
    import realtime_deploy as deploy

    if exclude_profiles is None:
        exclude_profiles = DEFAULT_EXCLUDE_PROFILES

    if reserved_gi_ids is None:
        reserved_gi_ids = []

    print(f"    [dynamic] Starting cleanup-based config query on {node_name} GPU {gpu_index}")
    print(f"    [dynamic] Reserved GI IDs (HP): {reserved_gi_ids}")

    # Step 1: 현재 MIG 인스턴스 조회
    current_instances = deploy.query_mig_instances(node_name, gpu_index)
    if not current_instances:
        # 빈 리스트 또는 None: MIG 인스턴스가 없는 것은 정상일 수 있음
        print(f"    [dynamic] No current MIG instances found (may be normal)")
        current_instances = []

    print(f"    [dynamic] Current instances: {len(current_instances)}")
    for inst in current_instances:
        marker = " [RESERVED]" if inst.get("gi_id") in reserved_gi_ids else ""
        print(f"      - GI {inst.get('gi_id')}: {inst.get('size')}{marker}")

    # Step 2: Managed 인스턴스 식별 (reserved 제외)
    managed_gi_ids = [
        inst["gi_id"] for inst in current_instances
        if inst.get("gi_id") not in reserved_gi_ids
    ]

    # Step 3: Managed 인스턴스 삭제
    if managed_gi_ids:
        print(f"    [dynamic] Deleting managed instances: {managed_gi_ids}")
        gi_ids_str = ",".join(map(str, managed_gi_ids))

        # dci (Compute Instance 삭제) - 실패해도 계속 진행
        dci_cmd = f"nvidia-smi mig -i {gpu_index} -dci -gi {gi_ids_str}"
        try:
            deploy.run_cmd_on_node(node_name, dci_cmd)
        except Exception as e:
            print(f"    [dynamic] dci warning (may be OK): {e}")

        # dgi (GPU Instance 삭제)
        dgi_cmd = f"nvidia-smi mig -i {gpu_index} -dgi -gi {gi_ids_str}"
        try:
            result = deploy.run_cmd_on_node(node_name, dgi_cmd)
            if not result.get("ok", False):
                stderr = result.get("stderr", "") or ""
                stdout = result.get("stdout", "") or ""
                print(f"    [dynamic] WARNING: dgi failed: {stderr}")

                # "In use by another client" 에러 감지
                if "In use by another client" in stderr or "In use by another client" in stdout:
                    print(f"    [dynamic] DETECTED: 'In use by another client' error")
                    print(f"    [dynamic] → Returning 'IN_USE_ERROR' for hard reset")
                    return {"IN_USE_ERROR": True}
        except Exception as e:
            print(f"    [dynamic] ERROR: dgi failed: {e}")
            return {}

        print(f"    [dynamic] Managed instances deleted")
    else:
        print(f"    [dynamic] No managed instances to delete")

    # Step 4: -lgip 조회 (이제 정확한 free 값!)
    cmd_str = f"nvidia-smi mig -lgip -i {gpu_index}"
    try:
        lgip_output = run_cmd_on_node(node_name, cmd_str, gpu_index)
    except RuntimeError as e:
        print(f"    [dynamic] ERROR: Failed to run -lgip after cleanup: {e}")
        return {}

    profiles = parse_lgip(lgip_output)
    if not profiles:
        print(f"    [dynamic] WARNING: No profiles found after cleanup")
        return {}

    print(f"    [dynamic] Profiles after cleanup:")
    for name, p in profiles.items():
        if p.name not in exclude_profiles:
            print(f"      - {name}: free={p.free}, total={p.total}")

    # Step 5: Reserved 슬라이스 계산
    reserved_slices = 0
    for inst in current_instances:
        if inst.get("gi_id") in reserved_gi_ids:
            size_str = inst.get("size", "0g").replace("g", "")
            try:
                reserved_slices += int(size_str)
            except ValueError:
                pass

    total_slices = infer_total_slices(profiles)
    remaining_slices = max(0, total_slices - reserved_slices)

    print(f"    [dynamic] Slices: total={total_slices}, reserved={reserved_slices}, remaining={remaining_slices}")

    if remaining_slices == 0:
        print(f"    [dynamic] WARNING: No remaining slices")
        return {}

    # Step 6: ILP용 구성 생성
    configs = build_configs_for_ilp(profiles, remaining_slices, exclude_profiles)

    print(f"    [dynamic] Generated {len(configs)} configs (placement-aware)")
    for cfg_name in sorted(configs.keys())[:5]:
        print(f"      - {cfg_name}: {configs[cfg_name]}")
    if len(configs) > 5:
        print(f"      ... and {len(configs) - 5} more")

    return configs


# ============================================================
# Validation & Debugging
# ============================================================

def validate_configs(configs: Dict[str, Dict[str, int]]) -> bool:
    """
    ILP 구성 딕셔너리 유효성 검사

    Args:
        configs: ILP 솔버용 MIG 구성 딕셔너리

    Returns:
        유효하면 True
    """
    if not configs:
        return False

    for profile_str, slot_counts in configs.items():
        # 프로파일 문자열 검증
        if not profile_str or not all(c.isdigit() for c in profile_str):
            print(f"Invalid profile string: {profile_str}")
            return False

        # 슬롯 카운트 검증
        required_keys = {"1g", "2g", "3g", "4g", "7g"}
        if not required_keys.issubset(slot_counts.keys()):
            print(f"Missing slot keys in {profile_str}: {slot_counts}")
            return False

        # 슬라이스 합계 검증
        total = sum(int(c) for c in profile_str)
        calc_total = (
            slot_counts["1g"] * 1 +
            slot_counts["2g"] * 2 +
            slot_counts["3g"] * 3 +
            slot_counts["4g"] * 4 +
            slot_counts["7g"] * 7
        )
        if total != calc_total:
            print(f"Slice mismatch in {profile_str}: profile={total}, counts={calc_total}")
            return False

    return True


def print_configs(configs: Dict[str, Dict[str, int]], title: str = "MIG Configs") -> None:
    """디버깅용 구성 출력"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    for profile_str in sorted(configs.keys(), key=lambda x: (-len(x), x)):
        counts = configs[profile_str]
        parts = []
        for size in ["7g", "4g", "3g", "2g", "1g"]:
            if counts[size] > 0:
                parts.append(f"{size}x{counts[size]}")
        print(f"  {profile_str:>10}: {', '.join(parts)}")

    print()


# ============================================================
# Test / Demo
# ============================================================

if __name__ == "__main__":
    print("Testing MIG Dynamic Config Module")
    print("="*60)

    # 테스트 1: Reserved 없는 경우 (7 슬라이스 전체 사용 가능)
    print("\n[Test 1] No reserved slices (full 7g available)")
    try:
        configs = get_dynamic_configs(reserved_slices=0)
        if configs:
            print_configs(configs, "Available MIG Configs (7g)")
            print(f"Validation: {'PASS' if validate_configs(configs) else 'FAIL'}")
        else:
            print("No configs returned (nvidia-smi may not be available)")
    except Exception as e:
        print(f"Error: {e}")

    # 테스트 2: HP가 3g 사용 중 (4g 남음)
    print("\n[Test 2] HP using 3g (4g remaining)")
    try:
        configs = get_dynamic_configs(reserved_slices=3)
        if configs:
            print_configs(configs, "Available MIG Configs (4g)")
            print(f"Validation: {'PASS' if validate_configs(configs) else 'FAIL'}")
        else:
            print("No configs returned")
    except Exception as e:
        print(f"Error: {e}")

    # 테스트 3: 현재 MIG 인스턴스 조회
    print("\n[Test 3] Current MIG Instances")
    try:
        instances = get_current_mig_instances()
        if instances:
            for name, pid, placement in instances:
                print(f"  - {name} (id={pid}) at {placement}")
        else:
            print("  No MIG instances found")
    except Exception as e:
        print(f"Error: {e}")
