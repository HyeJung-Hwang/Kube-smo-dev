"""
Real-time Deployment Functions

Uses kubectl exec to GPU Admin DaemonSet pods for nvidia-smi commands.
"""
import subprocess
import time
import os

from job import Job, RANJob, GPUNode

# MIG Profile ID mapping (nvidia-smi)
MIG_PROFILE_IDS = {
    "1g": 19,  # MIG 1g.12gb
    "2g": 14,  # MIG 2g.24gb
    "3g": 9,   # MIG 3g.48gb (실제는 3g.40gb일 수도 있음)
    "4g": 5,   # MIG 4g.48gb
    "7g": 0,   # MIG 7g.94gb (Full GPU)
}

# GPU Admin Pod 설정 (kubectl exec 사용)
GPU_ADMIN_NAMESPACE = "nvidia-device-plugin"
GPU_ADMIN_PODS = {
    "skt-6gtb-ars": "gpu-admin-arm",      # ARM 노드용 DaemonSet
    "sys-221he-tnr": "gpu-admin-x86",     # x86 노드용 DaemonSet
}

# device-plugin 설정
DEVICE_PLUGIN_DAEMONSET = "nvdp-nvidia-device-plugin"
DEVICE_PLUGIN_NAMESPACE = "nvidia-device-plugin"


def get_gpu_admin_pod_name(node_name: str) -> str:
    """노드에 해당하는 gpu-admin pod 이름 가져오기"""
    if node_name not in GPU_ADMIN_PODS:
        print(f"  Warning: Node {node_name} not in GPU_ADMIN_PODS")
        return None

    daemonset_name = GPU_ADMIN_PODS[node_name]
    cmd = f"kubectl get pods -n {GPU_ADMIN_NAMESPACE} -l app=gpu-admin -o jsonpath='{{.items[?(@.spec.nodeName==\"{node_name}\")].metadata.name}}'"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            print(f"  Failed to get gpu-admin pod for {node_name}: {result.stderr}")
            return None
    except Exception as e:
        print(f"  Failed to get gpu-admin pod for {node_name}: {e}")
        return None


def run_cmd_on_node(node_name: str, cmd: str, timeout: int = 60) -> dict:
    """kubectl exec으로 노드에 명령 실행 (nsenter로 호스트 namespace 사용)"""
    pod_name = get_gpu_admin_pod_name(node_name)
    if not pod_name:
        return {"ok": False, "error": f"No gpu-admin pod found for {node_name}", "returncode": -1}

    # nsenter로 호스트 namespace에서 명령 실행 (MIG device 접근 위해 필요)
    # -t 1: PID 1 (init)의 namespace 사용
    # -m: mount namespace, -u: UTS namespace, -n: network namespace, -i: IPC namespace
    # bash -c로 감싸서 ; 같은 shell 연산자가 nsenter 안에서 실행되도록 함
    escaped_cmd = cmd.replace("'", "'\"'\"'")  # single quote escape
    nsenter_cmd = f"nsenter -t 1 -m -u -n -i bash -c '{escaped_cmd}'"
    kubectl_cmd = f"kubectl exec -n {GPU_ADMIN_NAMESPACE} {pod_name} -- {nsenter_cmd}"

    try:
        result = subprocess.run(kubectl_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Command timed out", "returncode": -1}
    except Exception as e:
        return {"ok": False, "error": str(e), "returncode": -1}


def get_gpu_resource(gpu_g: int, node_name: str = "") -> str:
    """GPU 크기 → MIG 리소스 키 (노드별 메모리 크기 분기)"""
    if node_name == "sys-221he-tnr":
        memory_map = {1: "12gb", 2: "24gb", 3: "47gb", 4: "47gb", 7: "80gb"}
    else:
        memory_map = {1: "12gb", 2: "24gb", 3: "48gb", 4: "48gb", 7: "80gb"}
    return f"nvidia.com/mig-{gpu_g}g.{memory_map.get(gpu_g, '10gb')}"


def get_mig_instance_uuids(node_name: str, gpu_index: int, max_retries: int = 3, include_size: bool = False) -> list:
    """
    nvidia-smi -L로 GPU의 모든 MIG 인스턴스 UUID 가져오기

    Args:
        node_name: 노드 이름
        gpu_index: GPU index (0, 1, ...)
        max_retries: 최대 재시도 횟수 (MIG reconfiguration 직후 대응)
        include_size: True면 (size_g, uuid) 튜플 리스트 반환, False면 uuid만 반환

    Returns:
        include_size=False: MIG 인스턴스 UUID 리스트 (Device index 순서대로)
            예: ["MIG-xxx-0", "MIG-xxx-1", "MIG-xxx-2", ...]
        include_size=True: (size_g, uuid) 튜플 리스트 (Device index 순서대로)
            예: [(2, "MIG-xxx-0"), (1, "MIG-xxx-1"), (1, "MIG-xxx-2")]
        실패 시 빈 리스트

    예시 출력:
        GPU 0: NVIDIA H100 NVL (UUID: GPU-661b464d-c7f8-a7ca-136c-64a234c567a6)
          MIG 2g.24gb     Device  0: (UUID: MIG-32bb2701-8e1d-5a83-b8a3-3fea272f197b)
          MIG 1g.12gb     Device  1: (UUID: MIG-0bb09895-71b4-5f75-8494-e1028a16cb9b)
          MIG 1g.12gb     Device  2: (UUID: MIG-3625749d-fa21-5d54-9625-06f5f4b37d06)
    """
    for attempt in range(max_retries):
        cmd = "nvidia-smi -L"
        resp = run_cmd_on_node(node_name, cmd)

        if not resp.get("ok", False):
            if attempt < max_retries - 1:
                print(f"  Failed to get MIG UUIDs from {node_name}, retry {attempt + 1}/{max_retries} after 2s...")
                print(f"  Error: {resp.get('error', resp.get('stderr', 'Unknown error'))}")
                time.sleep(2)
                continue
            print(f"  Failed to get MIG UUIDs from {node_name}")
            return []

        output = resp.get("stdout", "")

        # Parse output
        # GPU 라인과 MIG 라인을 구분
        lines = output.split('\n')
        mig_entries = []  # (device_num, size_g, uuid)
        in_target_gpu = False

        for line in lines:
            # GPU 라인 확인
            if line.startswith(f"GPU {gpu_index}:"):
                in_target_gpu = True
                print(f"  Found GPU {gpu_index} on {node_name}")
                continue

            # 다른 GPU 라인을 만났을 때 처리
            if line.startswith("GPU ") and not line.startswith(f"GPU {gpu_index}:"):
                if not in_target_gpu:
                    # 아직 target GPU 섹션 전이면 그냥 다음 줄로 (GPU 0~gpu_index-1 스킵)
                    continue
                # 이미 target GPU 섹션 안이었으면 여기서 섹션 종료
                in_target_gpu = False
                print(f"  End of GPU {gpu_index} section")
                break

            # Target GPU의 MIG 인스턴스 라인
            # 예: "  MIG 4g.48gb     Device  0: (UUID: MIG-45e9804a-...)"
            if in_target_gpu and "MIG" in line and "Device" in line and "UUID:" in line:
                try:
                    device_str = line.split("Device")[1].split(":")[0].strip()
                    device_num = int(device_str)
                    uuid = line.split("UUID:")[1].strip().rstrip(")")
                    # MIG 크기 파싱: "MIG 4g.48gb" → 4
                    mig_part = line.strip().split("Device")[0].strip()  # "MIG 4g.48gb"
                    size_str = mig_part.split()[1]  # "4g.48gb"
                    size_g = int(size_str.split("g")[0])  # 4
                    mig_entries.append((device_num, size_g, uuid))
                except (ValueError, IndexError):
                    continue

        # UUID를 찾았으면 반환
        if mig_entries:
            # Device 번호 순으로 정렬
            mig_entries.sort(key=lambda x: x[0])
            if include_size:
                return [(size_g, uuid) for _, size_g, uuid in mig_entries]
            else:
                return [uuid for _, _, uuid in mig_entries]

        # UUID를 못 찾았으면 재시도
        if attempt < max_retries - 1:
            print(f"  No MIG UUIDs found, retry {attempt + 1}/{max_retries} after 2s...")
            time.sleep(2)

    # 모든 재시도 실패
    return []

def deploy_job(job: Job, node_name: str, gpu_g: int, gpu_index: int = 0,
               slice_index: int = None, size_index: int = 0, avoid_uuids: list = None,
               launch_pattern: str = "F08 1C 59", cell_group_num: int = 1):

    job_id = job.job_id
    is_ran = isinstance(job, RANJob)


    gpu_resource = get_gpu_resource(gpu_g, node_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # MIG 인스턴스 UUID 가져오기 (크기 정보 포함)
    mig_info = get_mig_instance_uuids(node_name, gpu_index, include_size=True)
    if not mig_info:
        print(f" Failed to get MIG instance UUIDs for {node_name} GPU {gpu_index}")
        return None

    # gpu_g 크기로 매칭 (nvidia-smi Device 순서 ≠ profile 문자열 순서)
    matching_uuids = [uuid for sz, uuid in mig_info if sz == gpu_g]
    all_uuids = [uuid for _, uuid in mig_info]
    print(f"   MIG info: {[(sz, uuid[:20]+'...') for sz, uuid in mig_info]}")
    print(f"   Looking for {gpu_g}g slice, found {len(matching_uuids)} match(es)")

    if matching_uuids:
        # avoid_uuids 필터 적용
        if avoid_uuids:
            available = [u for u in matching_uuids if u not in avoid_uuids]
            if available:
                idx = min(size_index, len(available) - 1)
                mig_uuid = available[idx]
                print(f"   Selected UUID by size match ({gpu_g}g, size_index={size_index}): {mig_uuid}")
            else:
                # 같은 크기 UUID가 모두 avoid → 다른 크기 중 available
                fallback = [u for u in all_uuids if u not in avoid_uuids]
                if fallback:
                    mig_uuid = fallback[0]
                    print(f"    Size-matched UUIDs all in avoid list, fallback: {mig_uuid}")
                else:
                    print(f"    No available MIG UUID (all in avoid list)")
                    return None
        else:
            # size_index로 같은 크기 중 N번째 선택 (프로파일 내 상대 위치)
            idx = min(size_index, len(matching_uuids) - 1)
            mig_uuid = matching_uuids[idx]
            print(f"   Selected UUID by size match ({gpu_g}g, size_index={size_index}): {mig_uuid}")
    else:
        # 크기 매칭 실패 → 기존 slice_index 방식 fallback
        print(f"    No size match for {gpu_g}g, falling back to positional index")
        if slice_index is not None and 0 <= slice_index < len(all_uuids):
            mig_uuid = all_uuids[slice_index]
        elif avoid_uuids:
            available = [u for u in all_uuids if u not in avoid_uuids]
            if available:
                mig_uuid = available[0]
            else:
                print(f"    No available MIG UUID")
                return None
        else:
            mig_uuid = all_uuids[0]
        print(f"   Fallback UUID: {mig_uuid}")

    # RAN / AI 분기 사용해야할 helm install 명령어 다름.
    if is_ran:
        node_name = "skt-6gtb-ars"
        release = f"aerial-l1-{job_id}".lower().replace("_", "-")
        helm_chart = os.path.join(repo_root, "workload", "heng","aerial-l1")
        cmd = [
            "helm", "install", release, helm_chart,
            "--set", f"nodeName={node_name}",
            "--set", f"gpuResource={gpu_resource}",
            "--set", f"launchPattern={launch_pattern}",
            "--set", f"cell_group_num={cell_group_num}"
        ]

    else:
        release = f"test-{job_id}".lower().replace("_", "-")

        helm_chart = os.path.join(repo_root, "workload", "vllm-server")
        cmd = [
            "helm", "install", release, helm_chart,
            "--set", f"nodeName={node_name}",
            "--set", f"gpuResource={gpu_resource}",
            "--set", "server.modelPath=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct",
            "--set-string", "secret.hfApiToken=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",
            "--set", f"gpuId={mig_uuid}"
        ]
    tag = "[RAN]" if is_ran else "[AI]"
    try:
        print(f"   {tag} Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f" {tag} Deployed {release} on {node_name} GPU {gpu_index} slice {slice_index} (MIG UUID: {mig_uuid})")
            return mig_uuid
        else:
            print(f" {tag} Deploy failed: {result.stderr}")
            return None
    except Exception as e:
        print(f" {tag} Deploy error: {e}")
        return None



def undeploy_ran_job(job_id: str):
    """RAN Job 삭제"""
    release = f"aerial-l1-{job_id}".lower().replace("_", "-")
    cmd = ["helm", "uninstall", release]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  [RAN] Undeployed {release}")
            return True
        else:
            print(f"  [RAN] Undeploy failed: {result.stderr}")
            return False
    except Exception as e:
        print(f" [RAN] Undeploy error: {e}")
        return False


def undeploy_job(job_id: str):
    """Job 삭제"""
    release = f"test-{job_id}".lower().replace("_", "-")
    cmd = ["helm", "uninstall", release]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  Undeployed {release}")
            return True
        else:
            print(f"  Undeploy failed: {result.stderr}")
            return False
    except Exception as e:
        print(f" Undeploy error: {e}")
        return False


def undeploy_jobs_batch(job_ids: list):
    """여러 job을 한 번에 삭제 (batch undeploy)

    Args:
        job_ids: Job ID 리스트

    Returns:
        tuple: (success_count, failed_releases)
    """
    if not job_ids:
        return 0, []

    # List comprehension으로 모든 release 이름 생성
    releases = [f"test-{job_id}".lower().replace("_", "-") for job_id in job_ids]

    # helm uninstall release1 release2 release3 형태로 한 번에 실행
    cmd = ["helm", "uninstall"] + releases
    print(f"  Undeploying {len(releases)} jobs: {releases} using {cmd}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # 성공한 release 확인
        success_count = 0
        failed_releases = []

        # stdout에서 "uninstalled" 메시지 확인
        output_lines = result.stdout.split('\n')
        for release in releases:
            # helm uninstall의 출력에서 해당 release가 성공했는지 확인
            if any(release in line and "uninstalled" in line.lower() for line in output_lines):
                success_count += 1
                print(f"  Undeployed {release}")
            else:
                failed_releases.append(release)

        # stderr 출력
        if result.stderr:
            print(f"  Undeploy warnings/errors: {result.stderr}")

        # 실패한 release 출력
        if failed_releases:
            print(f"  Failed to undeploy: {failed_releases}")

        return success_count, failed_releases

    except subprocess.TimeoutExpired:
        print(f"  Undeploy timeout (60s) for {len(releases)} jobs")
        return 0, releases
    except Exception as e:
        print(f"  Undeploy batch error: {e}")
        return 0, releases


def deploy_job_replica(replica_id: str, target_job_id: str, node_name: str, gpu_g: int,
                       gpu_index: int = 0, slice_index: int = None, size_index: int = 0, avoid_uuids: list = None):
    """
    HP Job의 replica pod 배포 (scale-out)

    Args:
        replica_id: Replica ID (e.g., "hp-1g-1-replica-1")
        target_job_id: 원본 HP job ID (e.g., "hp-1g-1")
        node_name: 노드 이름
        gpu_g: GPU 크기
        gpu_index: GPU index
        slice_index: MIG slice index
        avoid_uuids: 피해야 할 UUID 리스트

    Returns:
        mig_uuid: 배포된 MIG UUID (성공 시), None (실패 시)
    """
    release = f"test-{replica_id}".lower().replace("_", "-")
    gpu_resource = get_gpu_resource(gpu_g, node_name)

    # 원본 job의 service ID 생성
    service_id = f"test-{target_job_id}".lower().replace("_", "-")

    # Get absolute path to repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    helm_chart = os.path.join(repo_root, "workload", "vllm-server-replica")

    # MIG 인스턴스 UUID 가져오기 (크기 정보 포함)
    mig_info = get_mig_instance_uuids(node_name, gpu_index, include_size=True)
    if not mig_info:
        print(f" Failed to get MIG instance UUIDs for {node_name} GPU {gpu_index}")
        return None

    # gpu_g 크기로 매칭 (nvidia-smi Device 순서 ≠ profile 문자열 순서)
    matching_uuids = [uuid for sz, uuid in mig_info if sz == gpu_g]
    all_uuids = [uuid for _, uuid in mig_info]

    if matching_uuids:
        if avoid_uuids:
            available = [u for u in matching_uuids if u not in avoid_uuids]
            if available:
                idx = min(size_index, len(available) - 1)
                mig_uuid = available[idx]
                print(f"   Replica UUID by size match ({gpu_g}g, size_index={size_index}): {mig_uuid}")
            else:
                fallback = [u for u in all_uuids if u not in avoid_uuids]
                if fallback:
                    mig_uuid = fallback[0]
                    print(f"    Size-matched UUIDs all in avoid list, fallback: {mig_uuid}")
                else:
                    print(f"    No available MIG UUID (all in avoid list)")
                    return None
        else:
            idx = min(size_index, len(matching_uuids) - 1)
            mig_uuid = matching_uuids[idx]
            print(f"   Replica UUID by size match ({gpu_g}g, size_index={size_index}): {mig_uuid}")
    else:
        print(f"    No size match for {gpu_g}g, falling back to positional index")
        if slice_index is not None and 0 <= slice_index < len(all_uuids):
            mig_uuid = all_uuids[slice_index]
        elif avoid_uuids:
            available = [u for u in all_uuids if u not in avoid_uuids]
            if available:
                mig_uuid = available[0]
            else:
                print(f"    No available MIG UUID (all in avoid list)")
                return None
        else:
            mig_uuid = all_uuids[0]
        print(f"   Fallback UUID: {mig_uuid}")

    cmd = [
        "helm", "install", release, helm_chart,
        "--set", f"nodeName={node_name}",
        "--set", f"gpuResource={gpu_resource}",
        "--set", "server.modelPath=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct",
        "--set-string", "secret.hfApiToken=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",
        "--set", f"gpuId={mig_uuid}",
        "--set", f"serviceId={service_id}"  # 원본 job의 service에 연결
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f" Deployed replica {release} for service {service_id} on {node_name} GPU {gpu_index} slice {slice_index} (MIG UUID: {mig_uuid})")
            return mig_uuid  # Return UUID on success
        else:
            print(f" Replica deploy failed: {result.stderr}")
            return None  # Return None on failure
    except Exception as e:
        print(f" Replica deploy error: {e}")
        return None


def undeploy_replica(replica_id: str):
    """
    Replica pod 삭제 (scale-in)

    Args:
        replica_id: Replica ID (e.g., "hp-1g-1-replica-1")

    Returns:
        bool: True if successful
    """
    release = f"test-{replica_id}".lower().replace("_", "-")
    cmd = ["helm", "uninstall", release]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  Undeployed replica {release}")
            return True
        else:
            print(f"  Replica undeploy failed: {result.stderr}")
            return False
    except Exception as e:
        print(f" Replica undeploy error: {e}")
        return False


def wait_for_pod_deletion(job_ids: list, max_wait_sec: int = 60) -> bool:
    """
    모든 Pod가 완전히 삭제될 때까지 대기

    Args:
        job_ids: 삭제 확인할 job ID 리스트
        max_wait_sec: 최대 대기 시간 (초)

    Returns:
        True if all pods deleted, False if timeout
    """
    if not job_ids:
        return True

    print(f"   Waiting for {len(job_ids)} pods to be deleted...")

    # Pod 이름 생성 (AI/Spot: test-{job_id}, RAN: aerial-l1-{job_id})
    pod_patterns = []
    for job_id in job_ids:
        # AI/Spot job pattern
        ai_release = f"test-{job_id}".lower().replace("_", "-")
        pod_patterns.append(ai_release)
        # RAN job pattern
        ran_release = f"aerial-l1-{job_id}".lower().replace("_", "-")
        pod_patterns.append(ran_release)

    start_time = time.time()

    for attempt in range(max_wait_sec):
        # 모든 Pod가 삭제되었는지 확인 (실제 pod 상태 체크)
        all_deleted = True

        for pattern in pod_patterns:
            # kubectl로 실제 pod 존재 여부 확인
            # Pod 이름 형식: test-{job_id}-vllm-server-xxxxx-xxxxx
            cmd = f"kubectl get pods 2>/dev/null | grep '{pattern}'"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            # Pod가 존재하면 (Running or Terminating)
            if result.returncode == 0 and result.stdout.strip():
                # 아직 Pod가 존재함
                all_deleted = False
                break

        if all_deleted:
            elapsed = time.time() - start_time
            print(f"   All pods deleted ({elapsed:.1f}s)")
            return True

        # 1초 대기 후 재확인
        time.sleep(1)

        # 진행 상황 출력 (5초마다)
        if (attempt + 1) % 5 == 0:
            print(f"   Still waiting... ({attempt + 1}/{max_wait_sec}s)")

    # 타임아웃
    elapsed = time.time() - start_time
    print(f"    Timeout waiting for pod deletion ({elapsed:.1f}s)")
    return False


def profile_to_mig_commands(profile: str, gpu_index: int = 0) -> tuple:
    """
    프로필 문자열을 nvidia-smi MIG 명령으로 변환

    Args:
        profile: "43", "31111", "1111111" 등
        gpu_index: GPU index (default: 0)

    Returns:
        (erase_cmd, create_cmd)
    """
    # 1. Erase all MIG instances
    # -dci가 CI 없으면 실패하므로 ; 사용 (&&면 -dgi가 실행 안됨)
    erase_cmd = f"sudo nvidia-smi mig -dci -i {gpu_index}; sudo nvidia-smi mig -dgi -i {gpu_index}"

    # 2. Parse profile string to get slice counts
    # "43" -> [4, 3], "31111" -> [3, 1, 1, 1, 1], "1111111" -> [1, 1, 1, 1, 1, 1, 1]
    slices = [int(c) for c in profile]

    # 3. Convert to profile IDs
    profile_ids = []
    for slice_size in slices:
        profile_id = MIG_PROFILE_IDS.get(f"{slice_size}g")
        if profile_id is None:
            raise ValueError(f"Unknown slice size: {slice_size}g")
        profile_ids.append(str(profile_id))

    # 4. Create command
    # sudo nvidia-smi mig -cgi 5,9 -C -i 0  (for "43")
    profile_ids_str = ",".join(profile_ids)
    create_cmd = f"sudo nvidia-smi mig -cgi {profile_ids_str} -C -i {gpu_index}"

    return erase_cmd, create_cmd


def query_mig_instances(node_name: str, gpu_index: int = 0) -> list:
    """
    현재 MIG instance 조회 (UUID 포함)

    Args:
        node_name: 노드 이름
        gpu_index: GPU index (default: 0)

    Returns:
        List of dicts: [{"gi_id": 7, "profile_id": 19, "size": "1g", "uuid": "MIG-xxx"}, ...]
    """
    # Get GI info (gi_id, profile_id, size)
    cmd = f"nvidia-smi mig -lgi -i {gpu_index}"
    resp = run_cmd_on_node(node_name, cmd)

    if not resp.get("ok", False):
        return []

    # Parse output
    output = resp.get("stdout", "")
    instances = []

    for line in output.split('\n'):
        # Look for lines like: |   0  MIG 1g.10gb      19        7          0:1       |
        if '|' in line and 'MIG' in line and not line.strip().startswith('+'):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    gpu_id = int(parts[1])
                    if gpu_id != gpu_index:
                        continue

                    # parts[2] = "MIG", parts[3] = "1g.10gb"
                    size_str = parts[3]  # "1g.10gb"
                    size = size_str.split('.')[0]  # "1g"

                    profile_id = int(parts[4])
                    gi_id = int(parts[5])

                    instances.append({
                        "gi_id": gi_id,
                        "profile_id": profile_id,
                        "size": size,
                        "uuid": None  # Will be filled below
                    })
                except (ValueError, IndexError):
                    continue

    # Get UUIDs with nvidia-smi -L (Device order matches query order)
    cmd_uuid = "nvidia-smi -L"
    resp_uuid = run_cmd_on_node(node_name, cmd_uuid)

    if resp_uuid.get("ok", False):
        output_uuid = resp_uuid.get("stdout", "")
        uuids = []

        # Parse MIG UUIDs from nvidia-smi -L output
        # Example: MIG 1g.12gb Device  0: (UUID: MIG-32bb2701-8e1d-5a83-b8a3-3fea272f197b)
        for line in output_uuid.split('\n'):
            if 'MIG' in line and 'Device' in line and 'UUID:' in line:
                try:
                    uuid_part = line.split('UUID:')[1].strip().rstrip(')')
                    uuids.append(uuid_part)
                except IndexError:
                    continue

        # Match UUIDs to instances (assuming same order)
        for i, inst in enumerate(instances):
            if i < len(uuids):
                inst["uuid"] = uuids[i]

    return instances


def get_uuid_to_gi_mapping(node_name: str, gpu_index: int = 0) -> dict:
    """
    MIG UUID → GI ID 매핑 생성

    nvidia-smi -L의 Device number와 nvidia-smi MIG devices table의 MIG Dev를 사용하여 매핑

    Args:
        node_name: 노드 이름
        gpu_index: GPU index (default: 0)

    Returns:
        dict: {UUID: GI_ID} 매핑
        예: {"MIG-ceb743f1-...": 2, "MIG-32bb2701-...": 7}
    """
    # Step 1: nvidia-smi -L에서 Device number → UUID 매핑 생성
    cmd_l = "nvidia-smi -L"
    resp_l = run_cmd_on_node(node_name, cmd_l)

    if not resp_l.get("ok", False):
        print(f"   Failed to get nvidia-smi -L output")
        return {}

    output_l = resp_l.get("stdout", "")
    device_to_uuid = {}  # Device number → UUID

    in_target_gpu = False
    for line in output_l.split('\n'):
        # GPU 라인 확인
        if line.startswith(f"GPU {gpu_index}:"):
            in_target_gpu = True
            continue

        # 다른 GPU 라인을 만났을 때
        if line.startswith("GPU ") and not line.startswith(f"GPU {gpu_index}:"):
            if in_target_gpu:
                break
            continue

        # Target GPU의 MIG 인스턴스 라인
        if in_target_gpu and "MIG" in line and "Device" in line and "UUID:" in line:
            try:
                device_str = line.split("Device")[1].split(":")[0].strip()
                device_num = int(device_str)
                uuid = line.split("UUID:")[1].strip().rstrip(")")
                device_to_uuid[device_num] = uuid
            except (ValueError, IndexError):
                continue

    if not device_to_uuid:
        print(f"   No MIG devices found in nvidia-smi -L")
        return {}

    print(f"   Device → UUID mapping: {device_to_uuid}")

    # Step 2: nvidia-smi의 MIG devices table에서 MIG Dev → GI ID 매핑 생성
    cmd_smi = f"nvidia-smi -i {gpu_index}"
    resp_smi = run_cmd_on_node(node_name, cmd_smi)

    if not resp_smi.get("ok", False):
        print(f"   Failed to get nvidia-smi output")
        return {}

    output_smi = resp_smi.get("stdout", "")
    mig_dev_to_gi = {}  # MIG Dev → GI ID

    # MIG devices table 파싱
    # 형식: | GPU  GI  CI  MIG |
    #       |      ID  ID  Dev |
    #       |  0    2   0   0  |
    in_mig_devices = False

    for line in output_smi.split('\n'):
        # MIG devices 섹션 시작
        if "MIG devices:" in line:
            in_mig_devices = True
            continue

        # MIG devices 섹션 종료 (Processes 섹션 시작)
        if in_mig_devices and ("Processes:" in line or line.strip().startswith("+===") and "Processes" in output_smi[output_smi.find(line):]):
            break

        # MIG device 라인 파싱
        if in_mig_devices and '|' in line and not line.strip().startswith('+'):
            parts = line.split('|')
            if len(parts) >= 2:
                # 첫 번째 데이터 셀에서 GPU, GI, CI, MIG Dev 추출
                data_cell = parts[1].strip()
                numbers = data_cell.split()

                # Header 라인 스킵 (GPU, GI, CI, MIG 등 텍스트 포함)
                if len(numbers) >= 4 and all(n.isdigit() or n == 'N/A' for n in numbers[:4]):
                    try:
                        gpu_id = int(numbers[0])
                        if gpu_id != gpu_index:
                            continue
                        gi_id = int(numbers[1])
                        # numbers[2]는 CI ID
                        mig_dev = int(numbers[3])
                        mig_dev_to_gi[mig_dev] = gi_id
                    except (ValueError, IndexError):
                        continue

    if not mig_dev_to_gi:
        print(f"   No MIG Dev → GI mapping found in nvidia-smi")
        return {}

    print(f"   MIG Dev → GI mapping: {mig_dev_to_gi}")

    # Step 3: UUID → GI ID 매핑 생성
    # Device number == MIG Dev 이므로 조합
    uuid_to_gi = {}
    for device_num, uuid in device_to_uuid.items():
        if device_num in mig_dev_to_gi:
            gi_id = mig_dev_to_gi[device_num]
            uuid_to_gi[uuid] = gi_id

    print(f"   UUID → GI mapping: {uuid_to_gi}")
    return uuid_to_gi


def query_active_gi_ids(node_name: str, gpu_index: int = 0) -> list:
    """
    현재 프로세스가 사용 중인 GI ID 조회

    Args:
        node_name: 노드 이름
        gpu_index: GPU index (default: 0)

    Returns:
        사용 중인 GI ID 리스트 (예: [7, 12])
    """
    cmd = f"nvidia-smi -i {gpu_index}"
    resp = run_cmd_on_node(node_name, cmd)

    if not resp.get("ok", False):
        return []

    output = resp.get("stdout", "")
    active_gi_ids = set()

    # Parse "Processes:" section only
    # Look for lines like: |    0    7    0    3256609    C   VLLM::EngineCore    10754MiB |
    in_processes_section = False

    for line in output.split('\n'):
        # Start parsing after "Processes:" line
        if "Processes:" in line:
            in_processes_section = True
            continue

        # Stop if we hit the end of the table
        if in_processes_section and line.strip().startswith('+---'):
            break

        # Parse process lines in the Processes section
        if in_processes_section and '|' in line:
            parts = line.split()
            # Process line format: | GPU | GI | CI | PID | Type | Process | Memory |
            # parts[0]='|', parts[1]=GPU, parts[2]=GI, parts[3]=CI, parts[4]=PID, ...
            if len(parts) >= 5 and parts[0] == '|':
                try:
                    # Skip header lines (ID, Usage, etc.)
                    if not parts[1].isdigit():
                        continue

                    gpu_id = int(parts[1])
                    if gpu_id != gpu_index:
                        continue

                    gi_id = int(parts[2])
                    # Verify this is a valid process line (PID should be large number)
                    pid = int(parts[4])
                    if pid > 1000:  # Valid PID
                        active_gi_ids.add(gi_id)
                except (ValueError, IndexError):
                    continue

    return sorted(list(active_gi_ids))


def restart_device_plugin(node_name:str) -> bool:
    """특정 노드의 device-plugin pod 재시작"""
    import time

    print(f"  Restarting device-plugin on {node_name}...")

    # 1. 특정 노드의 device-plugin pod 삭제 (DaemonSet이 자동으로 재생성)
    delete_cmd = f"kubectl delete pod -n {DEVICE_PLUGIN_NAMESPACE} -l app.kubernetes.io/name=nvidia-device-plugin --field-selector spec.nodeName={node_name}"
    result = subprocess.run(delete_cmd, shell=True, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"   Delete failed: {result.stderr}")
        return False

    print(f"   Pod deleted, waiting for DaemonSet to recreate...")

    # DaemonSet이 새 pod를 생성할 시간 대기
    time.sleep(5)

    # 2. 새 pod가 Ready 상태가 될 때까지 대기
    print(f"   Waiting for device-plugin pod to become ready on {node_name}...")
    wait_cmd = f"kubectl wait --for=condition=ready pod -n {DEVICE_PLUGIN_NAMESPACE} -l app.kubernetes.io/name=nvidia-device-plugin --field-selector spec.nodeName={node_name} --timeout=60s"
    result = subprocess.run(wait_cmd, shell=True, capture_output=True, text=True, timeout=70)

    if result.returncode != 0:
        print(f"   Wait failed: {result.stderr}")
        return False

    print(f"   Device-plugin restarted on {node_name}")
    return True


def _log_cmd_response(cmd: str, resp: dict, prefix: str = ""):
    """Helper to log command response details"""
    print(f"{prefix}[CMD] {cmd}")
    print(f"{prefix}  returncode: {resp.get('returncode', 'N/A')}")
    print(f"{prefix}  ok: {resp.get('ok', 'N/A')}")
    if resp.get('stdout'):
        stdout_lines = resp['stdout'].strip().split('\n')
        for line in stdout_lines[:10]:  # Limit to first 10 lines
            print(f"{prefix}  [stdout] {line}")
        if len(stdout_lines) > 10:
            print(f"{prefix}  [stdout] ... ({len(stdout_lines) - 10} more lines)")
    if resp.get('stderr'):
        stderr_lines = resp['stderr'].strip().split('\n')
        for line in stderr_lines:
            print(f"{prefix}  [stderr] {line}")
    if resp.get('error'):
        print(f"{prefix}  [error] {resp['error']}")


def hard_reconfigure_mig(node_name: str, gpu_index: int, new_profile: str):
    """
    Hard MIG reconfiguration - 모든 CI/GI 삭제 후 전체 재구성

    "In use by another client" 에러 발생 시 사용.
    모든 jobs를 undeploy한 후 호출해야 함.

    Args:
        node_name: 노드 이름 (e.g., "skt-6gtb-ars")
        gpu_index: GPU index
        new_profile: 새로 생성할 MIG 프로파일 (e.g., "3211")

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f" HARD MIG RECONFIGURATION (In use error recovery)")
    print(f"  node: {node_name}, gpu_index: {gpu_index}")
    print(f"  new_profile: {new_profile}")
    print(f"{'='*60}")

    # 1. 모든 CI 삭제
    print(f"\n  --- Step 1: Delete ALL Compute Instances ---")
    dci_cmd = f"nvidia-smi mig -dci -i {gpu_index}"
    dci_resp = run_cmd_on_node(node_name, dci_cmd)
    _log_cmd_response(dci_cmd, dci_resp, prefix="    ")
    # CI가 없으면 실패해도 OK

    # 2. 모든 GI 삭제
    print(f"\n  --- Step 2: Delete ALL GPU Instances ---")
    dgi_cmd = f"nvidia-smi mig -dgi -i {gpu_index}"
    dgi_resp = run_cmd_on_node(node_name, dgi_cmd)
    _log_cmd_response(dgi_cmd, dgi_resp, prefix="    ")
    if not dgi_resp.get("ok", False):
        stderr = dgi_resp.get("stderr", "") or ""
        stdout = dgi_resp.get("stdout", "") or ""
        # "No GPU instances" 에러는 OK (이미 삭제됨)
        if "No GPU instances" not in stderr and "No GPU instances" not in stdout:
            print(f"    ERROR: Hard reset failed to delete GI")
            return False
    print(f"    All GPU instances deleted")

    # 3. 새 프로파일 생성
    print(f"\n  --- Step 3: Create new MIG profile {new_profile} ---")
    profile_ids = []
    for slice_char in new_profile:
        slice_size = int(slice_char)
        profile_id = MIG_PROFILE_IDS.get(f"{slice_size}g")
        if profile_id is None:
            print(f"    Unknown slice size: {slice_size}g")
            return False
        profile_ids.append(str(profile_id))

    profile_ids_str = ",".join(profile_ids)
    create_cmd = f"nvidia-smi mig -cgi {profile_ids_str} -C -i {gpu_index}"
    create_resp = run_cmd_on_node(node_name, create_cmd)
    _log_cmd_response(create_cmd, create_resp, prefix="    ")
    if not create_resp.get("ok", False):
        print(f"    ERROR: Failed to create MIG instances")
        return False
    print(f"    MIG instances created: {new_profile}")

    # 4. Device plugin 재시작
    print(f"\n  --- Step 4: Restart Device Plugin ---")
    if not restart_device_plugin(node_name):
        print(f"    ERROR: Device plugin restart failed!")
        return False

    # 5. Wait for MIG resources
    print(f"    Waiting for MIG resources to be available...")
    time.sleep(10)

    print(f"\n{'='*60}")
    print(f"HARD MIG RECONFIGURATION COMPLETED")
    print(f"  profile: {new_profile}")
    print(f"{'='*60}")
    return True


def reconfigure_mig(node_name: str, profile: str, reserved_profile: str = None, gpu_index: int = 0, reserved_uuids: list = None):
    """
    MIG 재구성 (선택적 삭제 지원, Multi-GPU 지원)

    Args:
        node_name: 노드 이름 (e.g., "skt-6gtb-ars")
        profile: MIG profile 문자열 (e.g., "43", "31111", "1111111")
        reserved_profile: Reserved MIG profile (e.g., "3" - 유지할 slice)
                          None이면 전체 재구성 (기존 동작)
        gpu_index: GPU index for nvidia-smi -i <index> (default: 0)
        reserved_uuids: Reserved MIG UUIDs (e.g., ["MIG-xxx", "MIG-yyy"])
                        지정하면 해당 UUID의 GI를 reserve

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"MIG RECONFIGURATION START")
    print(f"  node: {node_name}, gpu_index: {gpu_index}")
    print(f"  target profile: {profile}")
    print(f"  reserved_profile: {reserved_profile}")
    print(f"  reserved_uuids: {reserved_uuids}")
    print(f"{'='*60}")

    if reserved_profile:
        print(f" Reconfiguring {node_name} GPU {gpu_index}: {profile} (reserving {reserved_profile})")
    else:
        print(f" Reconfiguring {node_name} GPU {gpu_index} to MIG profile {profile}")

    # 1. Check node is supported
    if node_name not in GPU_ADMIN_PODS:
        print(f"   Unknown node: {node_name}")
        print(f"   Available nodes: {list(GPU_ADMIN_PODS.keys())}")
        return False

    # 2. Selective reconfiguration (reserved instance 유지)
    if reserved_profile:
        print(f"   Reserved profile: {reserved_profile}")

        # Query current MIG instances
        print(f"  Querying current MIG configuration...")
        current_instances = query_mig_instances(node_name, gpu_index)

        if not current_instances:
            print(f"    Could not query MIG instances, falling back to full reconfiguration")
            reserved_profile = None  # Fall back to full reconfig
        else:
            print(f"   Found {len(current_instances)} MIG instances:")
            for inst in current_instances:
                print(f"     GI {inst['gi_id']}: {inst['size']}")

            # Identify reserved instances
            reserved_gi_ids = []
            managed_gi_ids = []

            if reserved_uuids:
                # UUID 기반 reserve: UUID → GI ID 매핑 사용
                print(f"   Reserved UUIDs: {reserved_uuids}")

                # UUID → GI ID 매핑 생성 (nvidia-smi -L의 Device와 nvidia-smi의 MIG Dev 매핑)
                print(f"  Building UUID → GI ID mapping...")
                uuid_to_gi = get_uuid_to_gi_mapping(node_name, gpu_index)

                # Reserved UUID에 해당하는 GI ID 찾기
                reserved_gi_from_uuid = set()
                for uuid in reserved_uuids:
                    if uuid in uuid_to_gi:
                        gi_id = uuid_to_gi[uuid]
                        reserved_gi_from_uuid.add(gi_id)
                        print(f"     UUID {uuid} → GI {gi_id} (reserved)")
                    else:
                        print(f"      UUID {uuid} not found in current MIG instances")

                # 모든 인스턴스 분류
                for inst in current_instances:
                    if inst["gi_id"] in reserved_gi_from_uuid:
                        reserved_gi_ids.append(inst["gi_id"])
                        print(f"     Reserving GI {inst['gi_id']} ({inst['size']}, HP job)")
                    else:
                        managed_gi_ids.append(inst["gi_id"])
                        print(f"     Managed GI {inst['gi_id']} ({inst['size']}, will be deleted)")
            else:
                # Size 기반 reserve: 실제 사용 중인 GI만 보호
                print(f"  Querying active processes...")
                active_gi_ids = query_active_gi_ids(node_name, gpu_index)
                print(f"   Active GI IDs (in use): {active_gi_ids}")

                reserved_sizes = [f"{c}g" for c in reserved_profile]
                for inst in current_instances:
                    # 크기가 맞고 + 실제 사용 중인 GI만 reserve
                    if inst["size"] in reserved_sizes and inst["gi_id"] in active_gi_ids and len(reserved_gi_ids) < len(reserved_profile):
                        reserved_gi_ids.append(inst["gi_id"])
                        print(f"     Reserving GI {inst['gi_id']} ({inst['size']}, in use)")
                    else:
                        managed_gi_ids.append(inst["gi_id"])

            print(f"  Reserved GI IDs: {reserved_gi_ids}")
            print(f"   Managed GI IDs to delete: {managed_gi_ids}")

            # Delete managed instances (if any exist)
            if managed_gi_ids:
                gi_ids_str = ",".join(map(str, managed_gi_ids))
                # -dci가 CI 없으면 실패하므로 별도 실행 (실패해도 무시)
                dci_cmd = f"nvidia-smi mig -i {gpu_index} -dci -gi {gi_ids_str}"
                dgi_cmd = f"nvidia-smi mig -i {gpu_index} -dgi -gi {gi_ids_str}"

                print(f"    Erasing managed MIG instances (GI {gi_ids_str})...")

                # CI 삭제 시도 (없으면 실패해도 OK)
                print(f"\n    --- Delete CI (Compute Instances) ---")
                dci_resp = run_cmd_on_node(node_name, dci_cmd)
                _log_cmd_response(dci_cmd, dci_resp, prefix="    ")
                if not dci_resp.get("ok", False):
                    print(f"    (No CI to delete or already deleted - continuing)")

                # GI 삭제 (이건 성공해야 함)
                print(f"\n    --- Delete GI (GPU Instances) ---")
                dgi_resp = run_cmd_on_node(node_name, dgi_cmd)
                _log_cmd_response(dgi_cmd, dgi_resp, prefix="    ")
                if not dgi_resp.get("ok", False):
                    # "In use by another client" 에러 감지
                    stderr = dgi_resp.get("stderr", "") or ""
                    stdout = dgi_resp.get("stdout", "") or ""
                    if "In use by another client" in stderr or "In use by another client" in stdout:
                        print(f"    DETECTED: 'In use by another client' error")
                        print(f"    → Returning 'IN_USE_ERROR' for hard reset")
                        return "IN_USE_ERROR"

                    print(f"    ERROR: Erase GI failed!")
                    print(f"    returncode: {dgi_resp.get('returncode')}")
                    print(f"    This usually means:")
                    print(f"      - GI IDs {managed_gi_ids} don't exist")
                    print(f"      - CI was not properly deleted first")
                    print(f"      - GPU instance is still in use by a process")
                    return False
                print(f"   Managed instances erased")
            else:
                print(f"   No managed instances to delete (already cleaned up)")
                print(f"  Proceeding to create new instances...")

            # Create new managed instances (reserved already exists)
            # reserved_profile을 제외한 나머지만 생성
            full_slices = list(profile)  # "3211" → ['3', '2', '1', '1']
            reserved_slices = list(reserved_profile) if reserved_profile else []  # "3" → ['3']

            # reserved 슬라이스 제거 (한 번씩만)
            managed_slices = full_slices.copy()
            for rs in reserved_slices:
                if rs in managed_slices:
                    managed_slices.remove(rs)

            managed_profile = "".join(managed_slices)  # "211"
            print(f"   Full profile: {profile}, Reserved: {reserved_profile}, Creating: {managed_profile}")

            if not managed_slices:
                print(f"   No new instances to create (all reserved)")
                # Restart device-plugin anyway
                print(f"\n    --- Restart Device Plugin ---")
                if not restart_device_plugin(node_name):
                    print(f"    ERROR: Device plugin restart failed!")
                    return False
                print(f"   Waiting for MIG resources to be available...")
                time.sleep(10)
                print(f"\n{'='*60}")
                print(f"MIG RECONFIGURATION COMPLETED (ALL RESERVED)")
                print(f"  profile: {profile}")
                print(f"  reserved: {reserved_profile} (all preserved)")
                print(f"  reserved_gi_ids: {reserved_gi_ids}")
                print(f"{'='*60}")
                return True

            profile_ids = []
            for slice_char in managed_slices:
                slice_size = int(slice_char)
                profile_id = MIG_PROFILE_IDS.get(f"{slice_size}g")
                if profile_id is None:
                    print(f"   Unknown slice size: {slice_size}g")
                    return False
                profile_ids.append(str(profile_id))

            profile_ids_str = ",".join(profile_ids)
            create_cmd = f"nvidia-smi mig -cgi {profile_ids_str} -C -i {gpu_index}"

            print(f"\n    --- Create New MIG Instances ---")
            print(f"  Creating new managed MIG profile {managed_profile}...")
            create_resp = run_cmd_on_node(node_name, create_cmd)
            _log_cmd_response(create_cmd, create_resp, prefix="    ")
            if not create_resp.get("ok", False):
                print(f"    ERROR: Create MIG instances failed!")
                print(f"    returncode: {create_resp.get('returncode')}")
                print(f"    This usually means:")
                print(f"      - Not enough contiguous GPU slots available")
                print(f"      - Invalid profile ID combination")
                print(f"      - Reserved instances blocking required slots")
                return False
            print(f"   Managed instances created")

            # Restart device-plugin
            print(f"\n    --- Restart Device Plugin ---")
            if not restart_device_plugin(node_name):
                print(f"    ERROR: Device plugin restart failed!")
                return False

            # Wait for MIG resources to be available
            print(f"   Waiting for MIG resources to be available...")
            time.sleep(10)

            print(f"\n{'='*60}")
            print(f"MIG RECONFIGURATION COMPLETED (SELECTIVE MODE)")
            print(f"  profile: {profile}")
            print(f"  reserved: {reserved_profile} (preserved)")
            print(f"  created: {managed_profile}")
            print(f"  reserved_gi_ids: {reserved_gi_ids}")
            print(f"{'='*60}")
            return True

    # 3. Full reconfiguration (no reserved profile)
    print(f"\n    === FULL RECONFIGURATION MODE ===")
    print(f"    No reserved HP jobs - erasing all instances")
    full_profile = profile

    # Convert profile to MIG commands
    try:
        erase_cmd, create_cmd = profile_to_mig_commands(full_profile, gpu_index)
    except ValueError as e:
        print(f"   ERROR: Invalid profile: {e}")
        return False

    print(f"   Sending MIG commands to {node_name}...")

    # Erase all MIG instances
    print(f"\n    --- Erase All MIG Instances ---")
    erase_resp = run_cmd_on_node(node_name, erase_cmd)
    _log_cmd_response(erase_cmd, erase_resp, prefix="    ")
    if not erase_resp.get("ok", False):
        stdout = erase_resp.get("stdout", "")
        # "No GPU instances found"는 이미 비어있다는 뜻 → 성공으로 처리
        if "No GPU instances found" in stdout:
            print(f"    Already empty (no instances to delete) - continuing")
        else:
            print(f"    ERROR: Erase all MIG instances failed!")
            print(f"    returncode: {erase_resp.get('returncode')}")
            print(f"    This usually means:")
            print(f"      - MIG instances are still in use by running processes")
            print(f"      - Permission denied (need sudo)")
            return False
    else:
        print(f"   Erase completed")

    # Create new MIG instances
    print(f"\n    --- Create New MIG Instances ---")
    print(f"    Creating MIG profile {full_profile}...")
    create_resp = run_cmd_on_node(node_name, create_cmd)
    _log_cmd_response(create_cmd, create_resp, prefix="    ")
    if not create_resp.get("ok", False):
        print(f"    ERROR: Create MIG instances failed!")
        print(f"    returncode: {create_resp.get('returncode')}")
        print(f"    This usually means:")
        print(f"      - Invalid profile ID")
        print(f"      - GPU doesn't support this MIG profile")
        return False
    print(f"   Create completed")

    # Restart device-plugin
    print(f"\n    --- Restart Device Plugin ---")
    if not restart_device_plugin(node_name):
        print(f"    ERROR: Device plugin restart failed!")
        return False

    # Wait for MIG resources to be available
    print(f"   Waiting for MIG resources to be available...")
    time.sleep(10)

    print(f"\n{'='*60}")
    print(f"MIG RECONFIGURATION COMPLETED SUCCESSFULLY")
    print(f"  profile: {profile}")
    print(f"{'='*60}")
    return True
