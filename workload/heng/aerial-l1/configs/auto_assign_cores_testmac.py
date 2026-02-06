# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import argparse
import yaml
import sys
import os, sys, yaml, tempfile, json

def _print_debug(title, obj):
    try:
        print(f"[DEBUG] {title}: {json.dumps(obj, ensure_ascii=False, indent=2)}")
    except Exception:
        print(f"[DEBUG] {title}: {obj}")

def _assert_writable(path):
    d = os.path.dirname(path) or "."
    ok_dir = os.path.isdir(d) and os.access(d, os.W_OK)
    print(f"[DEBUG] out_dir={d} exists={os.path.isdir(d)} writable={ok_dir}")
    if not ok_dir:
        raise RuntimeError(f"Output directory not writable: {d} (ConfigMap/읽기전용 마운트일 가능성)")

def _safe_write_yaml(out_path, data):
    # temp 파일로 먼저 쓰고 원자적 교체 — 읽기전용일 경우 여기서 예외 발생
    _assert_writable(out_path)
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        raise


def parse_cpu_list(s):
    core_list = []
    for s_comma in s.split(','):
        s_hyphon = s_comma.split('-')
        if len(s_hyphon) == 1:
            core_list.append(int(s_comma))
        else:
            core_list.extend(range(int(s_hyphon[0]),int(s_hyphon[1])+1))
    core_list.sort()
    return core_list


def get_cores_from_file(filename):
    core_list = []
    with open(filename) as f:
        cores_string = f.read()
        #print(f"Core string: {cores_string}")
        core_list = parse_cpu_list(cores_string)
        #print(f"Core list: {core_list}")
        #print(f"-----------------------------------")

    return core_list


def print_assignments(core_usage,ht_enabled):
    if ht_enabled:
        print('+-------------+----------------------------------------+-------------+----------------------------------------+')
        print('|   Primary   |              Primary Core              |   Sibling   |              Sibling Core              |')
        print('| Core Number |                  Uses                  | Core Number |                  Uses                  |')
        print('+-------------+----------------------------------------+-------------+----------------------------------------+')
        for core_pair in zip(core_usage['primary_core_uses'],core_usage['sibling_core_uses']):
            if core_pair[0][1] == '':
                continue
            print(f'| {core_pair[0][0]:6}      | {core_pair[0][1]:38} | {core_pair[1][0]:6}      | {core_pair[1][1]:38} |')
            print('+-------------+----------------------------------------+-------------+----------------------------------------+')
    else:
        print('+-------------+----------------------------------------+')
        print('| Core Number |                  Uses                  |')
        print('+-------------+----------------------------------------+')
        for core_pair in zip(core_usage['primary_core_uses'],core_usage['sibling_core_uses']):
            if core_pair[0][1] == '':
                continue
            print(f'| {core_pair[0][0]:6}      | {core_pair[0][1]:38} |')
            print('+-------------+----------------------------------------+')


def compute_core_assignments(args):
    # Get list of all CPUs
    all_cpus = [int(k.path.split('/sys/devices/system/cpu/cpu')[1]) for k in os.scandir('/sys/devices/system/cpu/')
                if k.is_dir() and
                   len(k.path.split('/sys/devices/system/cpu/cpu')) == 2 and
                   k.path.split('/sys/devices/system/cpu/cpu')[1].isnumeric()]
    all_cpus.sort()

    # Determine if hyperthreading enabled
    ht_enabled = False
    for core in all_cpus:
        sibling_cpus = get_cores_from_file(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list')
        if len(sibling_cpus) > 1:
            ht_enabled = True
            break # out of for loop
    print("using this")
    if ht_enabled:
        print(f'Detected HT Enabled')
    else:
        print(f'Detected HT Disabled')

    # Determine if system has multiple NUMA nodes
    all_numa_nodes = [int(k.path.split('/sys/devices/system/node/node')[1]) for k in os.scandir('/sys/devices/system/node')
                      if k.is_dir() and
                      len(k.path.split('/sys/devices/system/node/node')) == 2
                      and k.path.split('/sys/devices/system/node/node')[1].isnumeric()]
    all_numa_nodes.sort()
    multiple_numa_enabled = len(all_numa_nodes) > 1
    if multiple_numa_enabled:
        print(f'Detected Multiple NUMA Nodes: {all_numa_nodes}.  Will use node {args.numa_node} for scheduling.')
    else:
        print(f'Detected Single NUMA Node.  Forcing use of numa_node=0.')
        args.numa_node = 0

    affinity_cores = [k for k in os.sched_getaffinity(0)]
    affinity_cores.sort()
    if args.test_affinitycpus is not None:
        affinity_cores = parse_cpu_list(args.test_affinitycpus)
    print(f"OS core affinity: {affinity_cores}")

    affinity_numa_cores = [k for k in affinity_cores if os.path.isdir(f'/sys/devices/system/node/node{args.numa_node}/cpu{k}')]
    affinity_numa_cores.sort()
    print(f"OS core affinity for numa node {args.numa_node}: {affinity_numa_cores}")

    isolated_cpus = get_cores_from_file('/sys/devices/system/cpu/isolated')
    if args.test_isolcpus is not None:
        isolated_cpus = parse_cpu_list(args.test_isolcpus)
    print(f"OS isolated cores: {isolated_cpus}")

    primary_core_list = []
    primary_to_sibling_list = [-1 for k in range(max(affinity_cores)+1)]

    # Determine the primary cores and their siblings
    for core in affinity_numa_cores:
        sibling_cpus = get_cores_from_file(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list')
        if core == sibling_cpus[0]:
            primary_core_list.append(core)
        else:
            primary_to_sibling_list[sibling_cpus[0]] = sibling_cpus[1]

    # Prune physical cores that aren't isolated
    isolated_primary_core_list = []
    for core in primary_core_list:
        primary_isolated = core in isolated_cpus
        if ht_enabled:
            sibling_isolated = primary_to_sibling_list[core] in isolated_cpus
        else:
            sibling_isolated = True
        if primary_isolated and sibling_isolated:
            isolated_primary_core_list.append(core)

    print(f"Tentative primary cores: {isolated_primary_core_list}")

    required_core_count = 0
    if ht_enabled:
        if args.cuphycontroller_filename is not None:
            print("Cuphycontroller core assignment strategy for HT enabled:")
            print("  * 1 low priority primary core (shared with dpdk EAL), HT sibling for h2d_copy thread")
            print("  * {args.workers_ul_count} UL worker primary cores, HT siblings idle")
            print("  * {args.workers_dl_count} DL worker primary cores, HT siblings idle")
            print("  * 1 L2A timer thread primary core, HT sibling for L2A msg processing thread")
            required_core_count = 1 + args.workers_ul_count + args.workers_dl_count + 1

            if args.enable_debug_worker:
                print("  * 1 debug thread primary core, HT sibling idle")
                required_core_count+=1

        if args.testmac_filename is not None:
            print("testMAC core assignment strategy:")
            print("  * 1 low priority primary core, HT sibling idle")
            print("  * 1 mac_recv thread primary core, HT sibling idle")
            print("  * 1 builder thread primary core, HT sibling idle")
            required_core_count+=3
    else:
        if args.cuphycontroller_filename is not None:
            print("Cuphycontroller core assignment strategy for HT disabled:")
            print("  * 1 low priority core (shared with dpdk EAL)")
            print("  * 1 core for h2d_copy thread")
            print("  * {args.workers_ul_count} UL worker cores")
            print("  * {args.workers_dl_count} DL worker cores")
            print("  * 1 core shared by L2A timer thread and L2A msg processing thread")
            required_core_count = 1 + 1 + args.workers_ul_count + args.workers_dl_count + 1

            if args.enable_debug_worker:
                print("  * 1 debug thread core")
                required_core_count+=1

        if args.testmac_filename is not None:
            print("testMAC core assignment strategy:")
            print("  * 1 low priority core")
            print("  * 1 mac_recv thread core")
            print("  * 1 builder thread core")
            required_core_count+=3


    print(f"Need {required_core_count} physical cores (plus {args.reserved_cores} reserved), potential affinity for {len(isolated_primary_core_list)} isolated physical cores")
    if (required_core_count + args.reserved_cores) > len(isolated_primary_core_list):
        raise RuntimeError(f"Need {required_core_count} physical cores (plus {args.reserved_cores} reserved) but only assigned {len(isolated_primary_core_list)}")


    assignments = {}
    core_usage = {}
    core_usage['primary_core_uses'] = [[k,''] for k in range(max(isolated_primary_core_list)+1)]
    core_usage['sibling_core_uses'] = [[primary_to_sibling_list[k],''] for k in range(max(isolated_primary_core_list)+1)]

    core_count = args.reserved_cores

    if args.cuphycontroller_filename is not None:
        assignments['low_priority_core'] = isolated_primary_core_list[core_count]
        assignments['h2d_copy_core'] = primary_to_sibling_list[isolated_primary_core_list[core_count]]
        core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = 'low priority threads (inc. DPDK EAL)'
        if ht_enabled:
            core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = 'H2D copy'
        else:
            core_count+=1
            assignments['h2d_copy_core'] = isolated_primary_core_list[core_count]
            core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = 'H2D copy'
        core_count+=1

        assignments['workers_ul'] = []
        for k in range(args.workers_ul_count):
            assignments['workers_ul'].append(isolated_primary_core_list[core_count])
            core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = 'UL Worker'
            core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = '[idle]'
            core_count+=1

        assignments['workers_dl'] = []
        for k in range(args.workers_dl_count):
            assignments['workers_dl'].append(isolated_primary_core_list[core_count])
            core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = 'DL Worker'
            core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = '[idle]'
            core_count+=1

        assignments['timer'] = isolated_primary_core_list[core_count]
        if ht_enabled:
            assignments['msg_processing'] = primary_to_sibling_list[isolated_primary_core_list[core_count]]
            core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = 'L2A timer'
            core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = 'L2A msg processing'
        else:
            assignments['msg_processing'] = isolated_primary_core_list[core_count]
            core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = 'L2A timer & L2A msg processing'
        core_count+=1

        if args.enable_debug_worker:
            assignments['debug_core'] = isolated_primary_core_list[core_count]
            core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = '[debug thread]'
            core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = '[idle]'
            core_count+=1


    if args.testmac_filename is not None:
        assignments['testmac_low_priority_core'] = isolated_primary_core_list[core_count]
        core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = '[testmac] low priority threads'
        core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = '[idle]'
        core_count+=1

        assignments['testmac_recv_thread_core'] = isolated_primary_core_list[core_count]
        core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = '[testmac] recv'
        core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = '[idle]'
        core_count+=1

        assignments['testmac_builder_core'] = isolated_primary_core_list[core_count]
        core_usage['primary_core_uses'][isolated_primary_core_list[core_count]][1] = '[testmac] builder'
        core_usage['sibling_core_uses'][isolated_primary_core_list[core_count]][1] = '[idle]'
        core_count+=1

    print_assignments(core_usage,ht_enabled)

    return assignments


def update_config_files(args,assignments):

    flow_style = None
    sort_keys = False

    if args.cuphycontroller_filename is not None:
        print(f'Parsing cuphycontroller configuration template: {args.cuphycontroller_filename}')
        with open(args.cuphycontroller_filename) as cuphycontroller_file:
            cuphycontroller_yaml = yaml.safe_load(cuphycontroller_file)
            cuphycontroller_yaml['low_priority_core'] = assignments['low_priority_core']
            cuphycontroller_yaml['cuphydriver_config']['dpdk_thread'] = assignments['low_priority_core']
            cuphycontroller_yaml['cuphydriver_config']['fh_stats_dump_cpu_core'] = assignments['low_priority_core']
            cuphycontroller_yaml['cuphydriver_config']['h2d_copy_thread_cpu_affinity'] = assignments['h2d_copy_core']
            cuphycontroller_yaml['cuphydriver_config']['workers_ul'] = assignments['workers_ul']
            cuphycontroller_yaml['cuphydriver_config']['workers_dl'] = assignments['workers_dl']
            if 'debug_core' in assignments:
                cuphycontroller_yaml['cuphydriver_config']['debug_worker'] = assignments['debug_core']
            else:
                cuphycontroller_yaml['cuphydriver_config']['debug_worker'] = -1
            l2adapter_filename = os.path.dirname(args.cuphycontroller_filename)+'/'+cuphycontroller_yaml['l2adapter_filename']
            cuphycontroller_yaml['l2adapter_filename'] = 'l2_adapter_config_dyncore.yaml'

        cuphycontroller_out_filename = os.path.dirname(args.cuphycontroller_filename)+'/cuphycontroller_dyncore.yaml'
        print(f'Writing cuphycontroller configuration: {cuphycontroller_out_filename}')
        with open(cuphycontroller_out_filename,'w') as cuphycontroller_file:
            yaml.dump(cuphycontroller_yaml,cuphycontroller_file,default_flow_style=flow_style, sort_keys=sort_keys)


        print(f'Parsing l2adapter configuration template: {l2adapter_filename}')
        with open(l2adapter_filename) as l2adapter_file:
            l2adapter_yaml = yaml.safe_load(l2adapter_file)
            l2adapter_yaml['timer_thread_config']['cpu_affinity'] = assignments['timer']
            l2adapter_yaml['message_thread_config']['cpu_affinity'] = assignments['msg_processing']

        l2adapter_out_filename = os.path.dirname(l2adapter_filename)+'/l2_adapter_config_dyncore.yaml'
        print(f'Writing l2adapter configuration: {l2adapter_out_filename}')
        with open(l2adapter_out_filename,'w') as l2adapter_file:
            yaml.dump(l2adapter_yaml,l2adapter_file,default_flow_style=flow_style, sort_keys=sort_keys)


    if args.testmac_filename is None:
        print("[DEBUG] testmac_filename is None — skip")
        return

    print(f"[DEBUG] Parsing testmac configuration template: {args.testmac_filename}")
    if not os.path.exists(args.testmac_filename):
        raise FileNotFoundError(f"testmac file not found: {args.testmac_filename}")

    with open(args.testmac_filename) as testmac_file:
        testmac_yaml = yaml.safe_load(testmac_file)

    # 방어적 초기화 (키가 없거나 타입이 dict가 아닌 경우)
    if testmac_yaml is None:
        testmac_yaml = {}
    if not isinstance(testmac_yaml, dict):
        raise TypeError(f"YAML root is not a dict: {type(testmac_yaml)}")

    for k in ["recv_thread_config", "builder_thread_config"]:
        if k not in testmac_yaml or not isinstance(testmac_yaml[k], dict):
            print(f"[DEBUG] initializing missing section: {k}")
            testmac_yaml[k] = {}

    # 할당값 존재 확인
    needed = {
        "testmac_low_priority_core": assignments.get("testmac_low_priority_core"),
        "testmac_recv_thread_core": assignments.get("testmac_recv_thread_core"),
        "testmac_builder_core": assignments.get("testmac_builder_core"),
    }
    _print_debug("assignments-needed", needed)
    if any(v is None for v in needed.values()):
        raise RuntimeError("assignments for testMAC are incomplete. "
                           "Check compute_core_assignments() and args flags.")

    # 변경 전 값 로그
    _print_debug("before.low_priority_core", testmac_yaml.get("low_priority_core"))
    _print_debug("before.recv_thread_config.cpu_affinity", testmac_yaml["recv_thread_config"].get("cpu_affinity"))
    _print_debug("before.builder_thread_config.cpu_affinity", testmac_yaml["builder_thread_config"].get("cpu_affinity"))

    # 실제 업데이트
    testmac_yaml["low_priority_core"] = needed["testmac_low_priority_core"]
    testmac_yaml["recv_thread_config"]["cpu_affinity"] = needed["testmac_recv_thread_core"]
    testmac_yaml["builder_thread_config"]["cpu_affinity"] = needed["testmac_builder_core"]

    # 변경 후 값 로그
    _print_debug("after.low_priority_core", testmac_yaml.get("low_priority_core"))
    _print_debug("after.recv_thread_config.cpu_affinity", testmac_yaml["recv_thread_config"].get("cpu_affinity"))
    _print_debug("after.builder_thread_config.cpu_affinity", testmac_yaml["builder_thread_config"].get("cpu_affinity"))

    # 출력 파일 경로
    testmac_out_filename = os.path.join(os.path.dirname(args.testmac_filename), "test_mac_config_dyncore.yaml")
    print(f"[DEBUG] Writing testmac configuration: {testmac_out_filename}")

    # 쓰기 가능 여부/원자적 쓰기
    try:
        _safe_write_yaml(testmac_out_filename, testmac_yaml)
    except Exception as e:
        print(f"[ERROR] failed to write YAML to {testmac_out_filename}: {e}")
        raise

    # 최종 검증: 다시 읽어 값 확인
    with open(testmac_out_filename) as f:
        verify = yaml.safe_load(f)
    _print_debug("verify.low_priority_core", verify.get("low_priority_core"))
    _print_debug("verify.recv_thread_config.cpu_affinity", verify["recv_thread_config"].get("cpu_affinity"))
    _print_debug("verify.builder_thread_config.cpu_affinity", verify["builder_thread_config"].get("cpu_affinity"))

    print("[DEBUG] testmac YAML update completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically compute CPU core asignment for Aerial cuphycontroller (and optionally testMAC)")
    parser.add_argument("-p", type=str, dest="cuphycontroller_filename", help="cuphycontroller (L1 PHY) configuration yaml filename")
    parser.add_argument("-t", type=str, dest="testmac_filename", help="testMAC (L2 MAC) configuration yaml filename")
    parser.add_argument("-d", dest="enable_debug_worker", action="store_const", const=True, default=False, help="Enable debug_worker in cuphycontroller")
    parser.add_argument("-n", dest="numa_node", default=1, type=int, help="Select NUMA node (default numa_node=1 on multi-node system)")
    parser.add_argument("-D", dest="workers_dl_count", default=3, type=int, help="Number of cuphydriver DL workers (default 3)")
    parser.add_argument("-U", dest="workers_ul_count", default=2, type=int, help="Number of cuphydriver UL workers (default 2)")
    parser.add_argument("-R", dest="reserved_cores", default=0, type=int, help="Number of cores to reserve/skip before running core allocation algorithm (default 0)")
    parser.add_argument("--test-isolcpus", type=str, dest="test_isolcpus", help="[for debug/test only] core list")
    parser.add_argument("--test-affinitycpus", type=str, dest="test_affinitycpus", help="[for debug/test only] core list")
    args = parser.parse_args()
    print("using this")
    assignments = compute_core_assignments(args)
    update_config_files(args,assignments)
 