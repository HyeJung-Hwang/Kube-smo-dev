# pip install pulp
import json
import math
import pulp as pl
import pandas as pd
from collections import defaultdict
from enums import available_MIG_profile_per_binpacking_profile

# ===============================
# 0) 입력: 잡/구성/슬롯/오버헤드
# ===============================
slot_minutes = 10.0   # 슬롯 길이(분)
delta = 0.1           # 리파티션 1회 오버헤드(분)
init_config = "1111111"

# 허용 MIG 구성(총 7g 가정). 필요 시 자유롭게 추가 가능.
configs = {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 3g×2
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 3g×1 + 1g×4
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 3g×1 + 1g×4   
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 1g×4
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},  
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    }# available_MIG_profile_per_binpacking_profile[init_config]

# configs = {
#     "111": {"3g": 0, "1g": 3, "2g":0 },  # 1g×7
#     "3":      {"3g": 1, "1g": 0,  "2g":0},  # 3g×2
#     "21":   {"3g": 0, "1g": 1, "2g":1  },  # 3g×1 + 1g×4
# }

# ===============================
# 시나리오 생성기
#  - 총 잡 수 n, 70모델 k=0..n (size=3), 나머지는 3모델(size=1)
#  - duration은 모두 duration(기본 10)
# ===============================
def generate_scenarios(n: int, duration: int = 10, include_2g: bool = False, include_all_sizes: bool = False,
                      random_duration: bool = False, duration_range: tuple = (10, 120), seed: int = None):
    """
    Generate job scenarios with varying workload compositions.

    Args:
        n: Total number of jobs
        duration: Duration of each job in minutes (ignored if random_duration=True)
        include_2g: If True, generates scenarios with 3g, 2g, and 1g jobs.
        include_all_sizes: If True, generates scenarios with 7g, 4g, 3g, 2g, and 1g jobs.
                          This overrides include_2g.
        random_duration: If True, generates random durations for each job
        duration_range: Tuple (min, max) for random duration in minutes
        seed: Random seed for reproducibility (optional)

    Returns:
        List of job scenarios. Each scenario is a list of job dictionaries.

    For include_all_sizes=False, include_2g=False (original):
        - Only 3g and 1g jobs
        - Example n=7: [{1g×7}, {3g×1, 1g×6}, ..., {3g×7}]

    For include_all_sizes=False, include_2g=True:
        - 3g, 2g, and 1g jobs
        - Example n=6: all combinations where sum = n

    For include_all_sizes=True (논문용 full):
        - 7g, 4g, 3g, 2g, and 1g jobs
        - Generates representative scenarios (not all combinations to avoid explosion)

    For random_duration=True:
        - Each job gets a random duration between duration_range[0] and duration_range[1]
        - Example: duration_range=(10, 120) generates jobs with 10-120 min durations
    """
    import random as rand

    # Set random seed if provided
    if seed is not None:
        rand.seed(seed)

    scenarios = []

    def get_duration():
        """Helper to get duration for a job"""
        if random_duration:
            return rand.randint(duration_range[0], duration_range[1])
        else:
            return duration

    if include_all_sizes:
        # Full size support: generate representative scenarios
        # Include some scenarios with large jobs (7g, 4g)

        # Scenario 1: One 7g job + smaller jobs
        if n >= 2:
            jobs = [{"name": "7g_1", "size": 7, "duration": get_duration()}]
            remaining = n - 1
            for i in range(1, remaining + 1):
                jobs.append({"name": f"1g_{i}", "size": 1, "duration": get_duration()})
            scenarios.append(jobs)

        # Scenario 2: One 4g job + mix of smaller
        if n >= 3:
            jobs = [
                {"name": "4g_1", "size": 4, "duration": get_duration()},
                {"name": "3g_1", "size": 3, "duration": get_duration()}
            ]
            remaining = n - 2
            for i in range(1, remaining + 1):
                jobs.append({"name": f"1g_{i}", "size": 1, "duration": get_duration()})
            scenarios.append(jobs)

        # Scenario 3: Multiple 4g jobs
        if n >= 2:
            num_4g = min(2, n // 2)
            jobs = []
            for i in range(1, num_4g + 1):
                jobs.append({"name": f"4g_{i}", "size": 4, "duration": get_duration()})
            remaining = n - num_4g
            for i in range(1, remaining + 1):
                jobs.append({"name": f"2g_{i}", "size": 2, "duration": get_duration()})
            scenarios.append(jobs)

        # Then add regular 3g/2g/1g combinations
        for num_3g in range(min(n + 1, 4)):  # Limit 3g to avoid too many scenarios
            for num_2g in range(min(n - num_3g + 1, 4)):
                num_1g = n - num_3g - num_2g
                if num_1g < 0:
                    continue
                jobs = []
                for i in range(1, num_3g + 1):
                    jobs.append({"name": f"3g_{i}", "size": 3, "duration": get_duration()})
                for i in range(1, num_2g + 1):
                    jobs.append({"name": f"2g_{i}", "size": 2, "duration": get_duration()})
                for i in range(1, num_1g + 1):
                    jobs.append({"name": f"1g_{i}", "size": 1, "duration": get_duration()})
                if jobs:  # Only add non-empty scenarios
                    scenarios.append(jobs)

    elif not include_2g:
        # Original behavior: only 3g and 1g jobs
        for k in range(n + 1):  # 3g job count
            jobs = []
            for i in range(1, k + 1):
                jobs.append({"name": f"3g_{i}", "size": 3, "duration": get_duration()})
            for j in range(1, n - k + 1):
                jobs.append({"name": f"1g_{j}", "size": 1, "duration": get_duration()})
            scenarios.append(jobs)
    else:
        # Extended behavior: 3g, 2g, and 1g jobs
        for num_3g in range(n + 1):
            for num_2g in range(n - num_3g + 1):
                num_1g = n - num_3g - num_2g
                jobs = []
                # Add 3g jobs
                for i in range(1, num_3g + 1):
                    jobs.append({"name": f"3g_{i}", "size": 3, "duration": get_duration()})
                # Add 2g jobs
                for i in range(1, num_2g + 1):
                    jobs.append({"name": f"2g_{i}", "size": 2, "duration": get_duration()})
                # Add 1g jobs
                for i in range(1, num_1g + 1):
                    jobs.append({"name": f"1g_{i}", "size": 1, "duration": get_duration()})
                scenarios.append(jobs)

    return scenarios

# ===============================
# 유틸: 입력 표준화(name, req_g, p)
# ===============================
def normalize_jobs(jobs_raw):
    out = []
    for j in jobs_raw:
        out.append({
            "name": j.get("name"),
            "req_g": j.get("req_g", j.get("size")),
            "p": j.get("p", j.get("duration")),
        })
    return out

# ===============================
# MILP 구성/해석
# ===============================
def build_model(jobs, configs, slot_minutes=10.0, delta=0.0, init_config="1111111"):
    J = list(range(len(jobs)))
    name = {j: jobs[j]["name"] for j in J}
    req  = {j: jobs[j]["req_g"] for j in J}
    pmin = {j: jobs[j]["p"]     for j in J}
    d    = {j: int(math.ceil(pmin[j] / slot_minutes)) for j in J}

    # 시간 수 T: 보수적 상한 (모든 d 합), 성능을 위해 상한 제한
    T = min(sum(d.values()), max(d.values()) + len(J) + 3)
    C = list(configs.keys())

    m = pl.LpProblem("MIG_dynamic_partitioning", pl.LpMinimize)

    # 변수
    y = pl.LpVariable.dicts("y", ((c,t) for c in C for t in range(1, T+1)), 0, 1, pl.LpBinary)  # 구성
    s = pl.LpVariable.dicts("s", ((j,t) for j in J for t in range(1, T+1)), 0, 1, pl.LpBinary)  # 시작
    a = pl.LpVariable.dicts("a", ((j,t) for j in J for t in range(1, T+1)), 0, 1, pl.LpBinary)  # 실행
    z = pl.LpVariable.dicts("z", (t for t in range(1, T+1)), 0, 1, pl.LpBinary)                 # 리파티션
    Tmax = pl.LpVariable("Tmax", lowBound=0, upBound=T, cat=pl.LpInteger)                       # 마지막 슬롯

    # (1) 슬롯마다 구성 1개
    for t in range(1, T+1):
        m += pl.lpSum(y[c, t] for c in C) == 1, f"one_cfg_t{t}"

    # (2) 각 잡은 정확히 1회 시작(비선점)
    for j in J:
        m += pl.lpSum(s[j, t] for t in range(1, T+1)) == 1, f"start_once_j{j}"

    # (3) 활성(연속 실행) 선형화
    for j in J:
        dj = d[j]
        for t in range(1, T+1):
            lo = max(1, t - dj + 1)
            m += a[j, t] <= pl.lpSum(s[j, tau] for tau in range(lo, t+1)), f"a_ub_j{j}_t{t}"
            for tau in range(lo, t+1):
                m += a[j, t] >= s[j, tau], f"a_lb_j{j}_t{t}_tau{tau}"

    # (4) 슬롯별 용량 - 각 인스턴스는 하나의 job만 실행
    # 전체 실행 중인 job 수가 전체 인스턴스 수를 초과하지 않아야 함
    # 단, 큰 인스턴스가 작은 job을 실행할 수 있음 (예: 4g가 3g job 실행)

    for t in range(1, T+1):
        # 크기별 job 개수 (논문용 명확한 표기)
        num_7g_jobs = pl.lpSum(a[j, t] for j in J if req[j] == 7)
        num_4g_jobs = pl.lpSum(a[j, t] for j in J if req[j] == 4)
        num_3g_jobs = pl.lpSum(a[j, t] for j in J if req[j] == 3)
        num_2g_jobs = pl.lpSum(a[j, t] for j in J if req[j] == 2)
        num_1g_jobs = pl.lpSum(a[j, t] for j in J if req[j] == 1)

        # 각 구성별 인스턴스 개수
        num_7g_inst = pl.lpSum(y[c, t] * configs[c].get("7g", 0) for c in C)
        num_4g_inst = pl.lpSum(y[c, t] * configs[c].get("4g", 0) for c in C)
        num_3g_inst = pl.lpSum(y[c, t] * configs[c].get("3g", 0) for c in C)
        num_2g_inst = pl.lpSum(y[c, t] * configs[c].get("2g", 0) for c in C)
        num_1g_inst = pl.lpSum(y[c, t] * configs[c].get("1g", 0) for c in C)

        # First Fit Decreasing 할당 제약
        # 큰 job부터 순서대로 할당: 7g → 4g → 3g → 2g → 1g

        # 7g jobs: 반드시 7g 인스턴스만 사용
        m += num_7g_jobs <= num_7g_inst, f"cap_7g_exact_t{t}"

        # 4g jobs: 남은 7g 인스턴스 + 4g 인스턴스 사용
        m += num_4g_jobs <= (num_7g_inst - num_7g_jobs) + num_4g_inst, f"cap_4g_t{t}"

        # 3g jobs: 남은 (7g, 4g) 인스턴스 + 3g 인스턴스 사용
        m += num_3g_jobs <= (num_7g_inst - num_7g_jobs) + (num_4g_inst - num_4g_jobs) + num_3g_inst, \
             f"cap_3g_t{t}"

        # 2g jobs: 남은 (7g, 4g, 3g) 인스턴스 + 2g 인스턴스 사용
        m += num_2g_jobs <= (num_7g_inst - num_7g_jobs) + (num_4g_inst - num_4g_jobs) + \
                            (num_3g_inst - num_3g_jobs) + num_2g_inst, f"cap_2g_t{t}"

        # 1g jobs: 모든 남은 인스턴스 사용
        m += num_1g_jobs <= (num_7g_inst - num_7g_jobs) + (num_4g_inst - num_4g_jobs) + \
                            (num_3g_inst - num_3g_jobs) + (num_2g_inst - num_2g_jobs) + num_1g_inst, \
             f"cap_1g_t{t}"

    # (5) 리파티션 검출
    for c in C:
        if c != init_config:
            m += z[1] >= y[c, 1], f"reconfig_from_init_to_{c}_t1"
    for t in range(2, T+1):
        for c_now in C:
            for c_prev in C:
                if c_now != c_prev:
                    m += z[t] >= y[c_now, t] + y[c_prev, t-1] - 1, f"reconfig_t{t}_{c_prev}_to_{c_now}"

    # (6) makespan 하한
    for j in J:
        dj = d[j]
        for t in range(1, T+1):
            m += Tmax >= (t + dj - 1) * s[j, t], f"Tmax_lb_j{j}_t{t}"

    # 목적함수
    m += slot_minutes * Tmax + delta * pl.lpSum(z[t] for t in range(1, T+1)), "Min_makespan_plus_reconfig"

    data = dict(J=J, name=name, req=req, pmin=pmin, d=d, T=T, C=C, configs=configs,
                slot_minutes=slot_minutes, delta=delta, init_config=init_config)
    vars = dict(y=y, s=s, a=a, z=z, Tmax=Tmax)
    return m, data, vars

def extract_solution(m, data, vars):
    y, s, a, z, Tmax = vars["y"], vars["s"], vars["a"], vars["z"], vars["Tmax"]
    J, name, req, pmin, d, T, C, configs = (
        data["J"], data["name"], data["req"], data["pmin"], data["d"],
        data["T"], data["C"], data["configs"]
    )
    slot_minutes, delta, init_config = data["slot_minutes"], data["delta"], data["init_config"]

    # 선택된 구성
    chosen_cfg = {}
    for t in range(1, T+1):
        for c in C:
            if pl.value(y[c, t]) > 0.5:
                chosen_cfg[t] = c
                break

    # 시작/실행/완료
    starts = {}
    slot_jobs = defaultdict(list)
    for j in J:
        for t in range(1, T+1):
            if pl.value(s[j, t]) > 0.5:
                starts[j] = t
                break
        dj = d[j]
        for tt in range(starts[j], starts[j] + dj):
            slot_jobs[tt].append(name[j])

    reconfigs = [t for t in range(1, T+1) if pl.value(z[t]) > 0.5]
    T_max = int(pl.value(Tmax))
    obj = slot_minutes * T_max + delta * len(reconfigs)

    # 오버헤드 누계
    cum_rc, acc = {}, 0
    for t in range(1, T_max + 1):
        if t in reconfigs:
            acc += 1
        cum_rc[t] = acc

    # 완료 시간 및 job별 상세
    finish = {}
    job_rows = []
    for j in J:
        t0 = starts[j]
        dj = d[j]
        tend = t0 + dj - 1
        finish_min = slot_minutes * tend + delta * cum_rc[tend]
        finish[name[j]] = finish_min
        job_rows.append({
            "job_name": name[j],
            "req_g": req[j],
            "duration_min": pmin[j],
            "start_slot": t0,
            "end_slot": tend,
            "finish_min": round(finish_min, 4),
        })

    avg_jct = sum(finish.values()) / len(finish)

    return dict(
        status=pl.LpStatus[m.status],
        objective=obj,
        Tmax_slot=T_max,
        chosen_cfg=chosen_cfg,
        slot_jobs=slot_jobs,
        reconfigs=reconfigs,
        starts=starts,
        finish=finish,
        avg_jct=avg_jct,
        job_rows=job_rows,
    )

def add_no_good_cut(m, data, vars, sol):
    y, s = vars["y"], vars["s"]
    J, T, C = data["J"], data["T"], data["C"]

    terms = []
    for j in J:
        for t in range(1, T+1):
            val = round(pl.value(s[j, t]))
            terms.append(s[j, t] if val == 1 else (1 - s[j, t]))
    for t in range(1, T+1):
        for c in C:
            val = round(pl.value(y[c, t]))
            terms.append(y[c, t] if val == 1 else (1 - y[c, t]))

    N = len(terms)
    m += pl.lpSum(terms) <= N - 1, f"nogood_{len(m.constraints)}"

def solve_k_best(k, jobs_norm, configs, slot_minutes, delta, init_config):
    sols = []
    m, data, vars = build_model(jobs_norm, configs, slot_minutes, delta, init_config)
    for _ in range(k):
        m.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=30))
        if m.status != 1:  # not optimal/feasible
            break
        sol = extract_solution(m, data, vars)
        sols.append(sol)
        add_no_good_cut(m, data, vars, sol)
    return sols

def pick_best_by_makespan_then_avgjct(solutions, slot_minutes=10.0):
    """해 목록에서 makespan 최소 → 동률이면 avg_jct 최소 하나 선택"""
    if not solutions:
        return None
    for s in solutions:
        s["_makespan_min"] = slot_minutes * s["Tmax_slot"]
    min_mk = min(s["_makespan_min"] for s in solutions)
    cand = [s for s in solutions if abs(s["_makespan_min"] - min_mk) < 1e-9]
    best = min(cand, key=lambda s: s["avg_jct"])
    return best

# ===============================
# 여러 시나리오 실행/저장
# ===============================
def run_for_jobs_list(jobs_list, k_best=5, save_csv=False, prefix="results"):
    summary_rows, job_rows_all = [], []

    for scen_id, jobs_raw in enumerate(jobs_list, start=1):
        jobs_norm = normalize_jobs(jobs_raw)
        sols = solve_k_best(
            k=k_best,
            jobs_norm=jobs_norm,
            configs=configs,
            slot_minutes=slot_minutes,
            delta=delta,
            init_config=init_config,
        )
        best = pick_best_by_makespan_then_avgjct(sols, slot_minutes=slot_minutes)
        scen_str = json.dumps(jobs_raw, ensure_ascii=False)

        if best is None:
            summary_rows.append({
                "scenario_index": scen_id,
                "scenario": scen_str,
                "solution_id": None,
                "objective_min": None,
                "makespan_min": None,
                "avg_jct_min": None,
                "num_reconfigs": None,
                "reconfig_slots": None,
            })
            continue

        summary_rows.append({
            "scenario_index": scen_id,
            "scenario": scen_str,
            "solution_id": 1,
            "objective_min": round(best["objective"], 4),
            "makespan_min": round(slot_minutes * best["Tmax_slot"], 4),
            "avg_jct_min": round(best["avg_jct"], 4),
            "num_reconfigs": len(best["reconfigs"]),
            "reconfig_slots": best["reconfigs"],
        })
        for r in best["job_rows"]:
            job_rows_all.append({
                "scenario_index": scen_id,
                "solution_id": 1,
                **r
            })

    df_summary = pd.DataFrame(summary_rows).sort_values(
        ["scenario_index", "solution_id"], na_position="last"
    ).reset_index(drop=True)

    df_jobs = pd.DataFrame(job_rows_all).sort_values(
        ["scenario_index", "solution_id", "finish_min", "job_name"]
    ).reset_index(drop=True)

    if save_csv:
        df_summary.to_csv(f"{prefix}_solutions_summary.csv", index=False)
        df_jobs.to_csv(f"{prefix}_jobs_detail.csv", index=False)

    return df_summary, df_jobs

def pretty_print(sol, slot_minutes=10.0, init_config="1111111"):
    print(f"[STATUS] {sol['status']} | Obj(min)={sol['objective']:.2f}, "
          f"Tmax(slots)={sol['Tmax_slot']}, reconfigs={sol['reconfigs']}")
    T_max = sol["Tmax_slot"]
    chosen_cfg = sol["chosen_cfg"]
    slot_jobs = sol["slot_jobs"]
    reconfigs  = set(sol["reconfigs"])
    for t in range(1, T_max + 1):
        tag = f"(init→{chosen_cfg[t]})" if t == 1 else chosen_cfg[t]
        mark = " [a]" if t in reconfigs else ""
        print(f"  t={t:>2} ({(t-1)*slot_minutes:>3.0f}~{t*slot_minutes:>3.0f}min): "
              f"cfg={tag:<8} jobs={slot_jobs.get(t, [])}{mark}")
    print("  Finish(min):")
    for k, v in sorted(sol["finish"].items(), key=lambda x: x[1]):
        print(f"    {k:>5}: {v:.2f}")
    print(f"\n  Average JCT (min) = {sol['avg_jct']:.2f}\n")

# ===============================
# 메인: n ∈ {4,7,10}에 대해 실행
# ===============================
if __name__ == "__main__":
    all_summary, all_jobs = [], []

    # 예시로 n=7의 scenario #4 최적해 콘솔 출력
    # (원하면 임의로 하나 더 출력)
    scenarios7 = generate_scenarios(7, duration=10)
    sols = solve_k_best(5, normalize_jobs(scenarios7[3]), configs, slot_minutes, delta, init_config)
    best = pick_best_by_makespan_then_avgjct(sols, slot_minutes)
    if best:
        print("\n=== Pretty print example for n=7, scenario #4 ===")
        pretty_print(best, slot_minutes, init_config)