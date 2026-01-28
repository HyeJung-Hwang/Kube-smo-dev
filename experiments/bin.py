# pip install pulp
import math
import pulp as pl
import pandas as pd
from collections import defaultdict
from MIG_CONFIG import mig_config

slot_minutes = 10.0  
delta = 0.1          
init_config = "31111"

def build_model(jobs, configs, slot_minutes=10.0, delta=0.0, init_config="1111111"):
    J = list(range(len(jobs)))
    name = {j: jobs[j]["name"] for j in J}
    req  = {j: jobs[j]["size"] for j in J}
    pmin = {j: jobs[j]["duration"]     for j in J}
    d    = {j: int(math.ceil(pmin[j] / slot_minutes)) for j in J}

    # 시간 수 T: 보수적 상한 (모든 d 합으로 충분)
    T = sum(d.values())
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

    # (4) 슬롯별 용량 (모든 크기: 7g, 4g, 3g, 2g, 1g)
    for t in range(1, T+1):
        # 7g job capacity
        m += pl.lpSum(a[j, t] for j in J if req[j] == 7) <= \
             pl.lpSum(y[c, t] * configs[c]["7g"] for c in C), f"cap_7g_t{t}"
        # 4g job capacity
        m += pl.lpSum(a[j, t] for j in J if req[j] == 4) <= \
             pl.lpSum(y[c, t] * configs[c]["4g"] for c in C), f"cap_4g_t{t}"
        # 3g job capacity
        m += pl.lpSum(a[j, t] for j in J if req[j] == 3) <= \
             pl.lpSum(y[c, t] * configs[c]["3g"] for c in C), f"cap_3g_t{t}"
        # 2g job capacity
        m += pl.lpSum(a[j, t] for j in J if req[j] == 2) <= \
             pl.lpSum(y[c, t] * configs[c]["2g"] for c in C), f"cap_2g_t{t}"
        # 1g job capacity
        m += pl.lpSum(a[j, t] for j in J if req[j] == 1) <= \
             pl.lpSum(y[c, t] * configs[c]["1g"] for c in C), f"cap_1g_t{t}"

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

    data = dict(J=J, name=name, req=req, d=d, T=T, C=C, configs=configs,
                slot_minutes=slot_minutes, delta=delta, init_config=init_config)
    vars = dict(y=y, s=s, a=a, z=z, Tmax=Tmax)
    return m, data, vars
def extract_solution(m, data, vars):
    y, s, a, z, Tmax = vars["y"], vars["s"], vars["a"], vars["z"], vars["Tmax"]
    J, name, d, T, C, configs = data["J"], data["name"], data["d"], data["T"], data["C"], data["configs"]
    slot_minutes, delta, init_config = data["slot_minutes"], data["delta"], data["init_config"]

    chosen_cfg = {}
    for t in range(1, T+1):
        for c in C:
            if pl.value(y[c, t]) > 0.5:
                chosen_cfg[t] = c
                break

    starts = {}
    from collections import defaultdict
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

    # 오버헤드 누계로 완료시간 계산
    cum_rc, acc = {}, 0
    for t in range(1, T_max + 1):
        if t in reconfigs:
            acc += 1
        cum_rc[t] = acc

    finish = {}
    for j in J:
        t0 = starts[j]
        tend = t0 + d[j] - 1
        finish[name[j]] = slot_minutes * tend + delta * cum_rc[tend]

    # --- 여기 추가: Average JCT (릴리즈=0 가정 → 평균 완료시간)
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
        avg_jct=avg_jct,   # 추가
    )


def add_no_good_cut(m, data, vars, sol):
    """이전 해와 '완전히 동일한' 스케줄이 다시 나오지 않도록 no-good cut 추가.
       (y와 s 바이너리 전체에 대해 최소 하나는 달라지도록 강제)
    """
    y, s = vars["y"], vars["s"]
    J, T, C = data["J"], data["T"], data["C"]

    terms = []
    # s에 대한 no-good
    for j in J:
        for t in range(1, T+1):
            val = round(pl.value(s[j, t]))
            # val==1이면 s[j,t]가 1이던 해였으니 '그대로'면 1, '바뀌면' 0 되도록.
            terms.append(s[j, t] if val == 1 else (1 - s[j, t]))
    # y에 대한 no-good
    for t in range(1, T+1):
        for c in C:
            val = round(pl.value(y[c, t]))
            terms.append(y[c, t] if val == 1 else (1 - y[c, t]))

    N = len(terms)
    m += pl.lpSum(terms) <= N - 1, f"nogood_{len(m.constraints)}"

def solve_k_best(k,  AI_jobs, configs, slot_minutes=10.0, delta=0.1, init_config="1111111"):
    sols = []
    m, data, vars = build_model(AI_jobs, configs, slot_minutes, delta, init_config)
    for it in range(k):
        m.solve(pl.PULP_CBC_CMD(msg=False))
        if pl.LpStatus[m.status] != "Optimal":
            break
        sol = extract_solution(m, data, vars,)
        sols.append(sol)
        # 다음 해를 위해 no-good cut 추가
        add_no_good_cut(m, data, vars, sol)
    return sols

def pretty_print(sol, slot_minutes=10.0, init_config="1111111"):
    print(f"[STATUS] Optimal | Obj(min)={sol['objective']:.2f}, Tmax(slots)={sol['Tmax_slot']}, reconfigs={sol['reconfigs']}")
    T_max = sol["Tmax_slot"]
    chosen_cfg = sol["chosen_cfg"]
    ## Need to add RAN_jobs
    slot_jobs = sol["slot_jobs"]
    reconfigs  = set(sol["reconfigs"])
    for t in range(1, T_max + 1):
        tag = f"(init→{chosen_cfg[t]})" if t == 1 else chosen_cfg[t]
        mark = " [a]" if t in reconfigs else ""
        print(f"  t={t:>2} ({(t-1)*slot_minutes:>3.0f}~{t*slot_minutes:>3.0f}min): cfg={tag:<10} jobs={slot_jobs.get(t, [])}{mark}")
    print("  Finish(min):")
    for k, v in sorted(sol["finish"].items(), key=lambda x: x[1]):
        print(f"    {k:>3}: {v:.2f}")
    # print()
    print(f"\n  Average JCT (min) = {sol['avg_jct']:.2f}\n")

# ===============================
# 실행 예시
# ===============================
if __name__ == "__main__":
    RAN_jobs = [
        # {"name": "RAN", "req_g": 4, "p": "until new workload arrives", "type": "RAN"},
    ]
    AI_jobs = [
        {"name": "70_1", "req_g": 3, "p": 20, "type": "AI"},
        {"name": "70_2", "req_g": 3, "p": 10, "type": "AI"},
        {"name": "70_3", "req_g": 3, "p": 10, "type": "AI"},
        {"name": "70_4", "req_g": 3, "p": 10, "type": "AI"},
        {"name": "3_1",  "req_g": 1, "p": 10, "type": "AI"},
        {"name": "3_2",  "req_g": 1, "p": 10, "type": "AI"},
        {"name": "3_3",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_4",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_5",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_6",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_7",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_8",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_9",  "req_g": 1, "p": 10, "type": "AI"},
        # {"name": "3_10", "req_g": 1, "p": 10, "type": "AI"},
    ]

    if len(RAN_jobs) > 0:
        for RAN_job in RAN_jobs:
            init_config = init_config[:7-RAN_job["req_g"]]
            configs = mig_config[init_config]
            print(f"RAN job requires {RAN_job['req_g']}g, so initial config is set to '{init_config}' with {configs}.")
    else:
        init_config = init_config
        configs = mig_config[init_config]
        print("No RAN jobs in this GPU.")

    sols = solve_k_best(
        k=10,
        AI_jobs=AI_jobs,
        configs=configs,
        slot_minutes=slot_minutes,
        delta=delta,
        init_config=init_config,
    )
    for i, sol in enumerate(sols, 1):
        print(f"==== Solution #{i} ====")
        pretty_print(sol, slot_minutes, init_config)
    rows = []
    for i, sol in enumerate(sols, 1):
        rows.append({
            "solution_id": i,
            "objective_min": round(sol["objective"], 4),
            "makespan_min": round(slot_minutes * sol["Tmax_slot"], 4),
            "avg_jct_min": round(sol["avg_jct"], 4),
            "num_reconfigs": len(sol["reconfigs"]),
            "reconfig_slots": sol["reconfigs"],
        })
    df_summaries = pd.DataFrame(rows).sort_values(["objective_min", "solution_id"]).reset_index(drop=True)
    print("\n=== Solution Summaries ===")
    print(df_summaries)

    # ====== (NEW) 최적 해(기준별) 뽑기 ======
    # 1) 평균 JCT 최소 해
    idx_best_jct = df_summaries["avg_jct_min"].idxmin()
    df_best_jct = df_summaries.loc[[idx_best_jct]].copy()
    df_best_jct.insert(0, "criterion", "min_avg_jct")

    # 2) makespan 최소 해 (오버헤드 제외 순수 makespan 기준)
    idx_best_mk = df_summaries["makespan_min"].idxmin()
    df_best_makespan = df_summaries.loc[[idx_best_mk]].copy()
    df_best_makespan.insert(0, "criterion", "min_makespan")

    # 3) 두 최적 해를 하나로 묶기 (중복이면 두 행이 같은 solution_id일 수 있음)
    df_best = pd.concat([df_best_jct, df_best_makespan], ignore_index=True)

    print("\n=== Best Solutions by Criterion ===")
    print(df_best)
    df_summaries.to_csv("solutions_summary.csv", index=False)
    df_best.to_csv("best_solutions.csv", index=False)
# # pip install pulp
# import pulp as pl
# from collections import defaultdict

# # ----------------------------
# # 1) 입력 데이터 (그대로 사용)
# # ----------------------------
# slot_minutes = 10.0   # 슬롯 길이(분)
# delta = 0.0           # 재파티션 1회 오버헤드(분). 예: 0.1로 설정 가능
# init_config = "1111111"

# # 잡 큐: name, req_g(필요 g), duration_minutes
# jobs = [
#     {"name": "70_1", "req_g": 3, "p": 10},
#     {"name": "70_2", "req_g": 3, "p": 10},
#     {"name": "70_3", "req_g": 3, "p": 10},
#     {"name": "70_4", "req_g": 3, "p": 10},
#     {"name": "3_1",  "req_g": 1, "p": 20},
#     {"name": "3_2",  "req_g": 1, "p": 10},
#     {"name": "3_3",  "req_g": 1, "p": 10},
# ]

# # 허용 MIG 구성(총 7g 가정). 각 구성의 3g/1g 슬롯 수를 명시
# configs = {
#     "1111111": {"3g": 0, "1g": 7},   # 1g×7
#     "331":     {"3g": 2, "1g": 1},   # 3g×2 + 1g×1
#     "31111":   {"3g": 1, "1g": 4},   # 3g×1 + 1g×4
#     # 필요하면 더 추가 가능
# }

# # ----------------------------
# # 2) 전처리
# # ----------------------------
# J = list(range(len(jobs)))
# name = {j: jobs[j]["name"] for j in J}
# req  = {j: jobs[j]["req_g"] for j in J}
# pmin = {j: jobs[j]["p"]     for j in J}

# # 슬롯 단위 길이 d_j = ceil(p_j / slot_minutes)
# import math
# d = {j: int(math.ceil(pmin[j] / slot_minutes)) for j in J}

# # 시간 수 T (보수적 상한): 모든 d_j 합으로 두면 충분히 큼
# T = sum(d.values())

# C = list(configs.keys())

# # ----------------------------
# # 3) MILP 모델
# # ----------------------------
# m = pl.LpProblem("MIG_dynamic_partitioning", pl.LpMinimize)

# # 변수
# y = pl.LpVariable.dicts("y", ((c,t) for c in C for t in range(1, T+1)), 0, 1, pl.LpBinary)  # 구성 선택
# s = pl.LpVariable.dicts("s", ((j,t) for j in J for t in range(1, T+1)), 0, 1, pl.LpBinary)  # 시작 시점
# a = pl.LpVariable.dicts("a", ((j,t) for j in J for t in range(1, T+1)), 0, 1, pl.LpBinary)  # 실행 중
# z = pl.LpVariable.dicts("z", (t for t in range(1, T+1)), 0, 1, pl.LpBinary)                 # 재파티션
# Tmax = pl.LpVariable("Tmax", lowBound=0, upBound=T, cat=pl.LpInteger)                       # 마지막 슬롯

# # (1) 슬롯당 구성 1개
# for t in range(1, T+1):
#     m += pl.lpSum(y[c, t] for c in C) == 1, f"one_cfg_t{t}"

# # (2) 각 잡은 정확히 1회 시작
# for j in J:
#     m += pl.lpSum(s[j, t] for t in range(1, T+1)) == 1, f"start_once_j{j}"

# # (3) 활성(연속 실행) 선형화: a[j,t] = sum_{tau in window} s[j,tau]
# for j in J:
#     dj = d[j]
#     for t in range(1, T+1):
#         lo = max(1, t - dj + 1)
#         # 상한
#         m += a[j, t] <= pl.lpSum(s[j, tau] for tau in range(lo, t+1)), f"a_ub_j{j}_t{t}"
#         # 하한(창 안의 s가 1이면 a도 1)
#         for tau in range(lo, t+1):
#             m += a[j, t] >= s[j, tau], f"a_lb_j{j}_t{t}_tau{tau}"

# # (4) 슬롯별 용량(3g/1g)
# for t in range(1, T+1):
#     # 3g 수요
#     m += pl.lpSum(a[j, t] for j in J if req[j] == 3) <= \
#          pl.lpSum(y[c, t] * configs[c]["3g"] for c in C), f"cap_3g_t{t}"
#     # 1g 수요
#     m += pl.lpSum(a[j, t] for j in J if req[j] == 1) <= \
#          pl.lpSum(y[c, t] * configs[c]["1g"] for c in C), f"cap_1g_t{t}"

# # (5) 재파티션 검출
# # t=1: 초기 구성과 다르면 z[1]=1
# for c in C:
#     if c != init_config:
#         m += z[1] >= y[c, 1], f"reconfig_from_init_to_{c}_t1"
# # t>=2: 이전과 다르면 z[t]=1
# for t in range(2, T+1):
#     for c_now in C:
#         for c_prev in C:
#             if c_now != c_prev:
#                 m += z[t] >= y[c_now, t] + y[c_prev, t-1] - 1, f"reconfig_t{t}_{c_prev}_to_{c_now}"

# # (6) makespan 하한: 시작 슬롯 t에서 길이 d_j를 고려
# for j in J:
#     dj = d[j]
#     for t in range(1, T+1):
#         m += Tmax >= (t + dj - 1) * s[j, t], f"Tmax_lb_j{j}_t{t}"

# # 목적함수
# m += slot_minutes * Tmax + delta * pl.lpSum(z[t] for t in range(1, T+1)), "Minimize_makespan_plus_reconfig"

# # 풀기
# m.solve(pl.PULP_CBC_CMD(msg=False))
# print(f"[STATUS] {pl.LpStatus[m.status]}")

# # ----------------------------
# # 4) 해석/출력
# # ----------------------------
# # 채택된 구성
# chosen_cfg = {}
# for t in range(1, T+1):
#     for c in C:
#         if pl.value(y[c, t]) > 0.5:
#             chosen_cfg[t] = c
#             break

# # 슬롯별 실행 잡
# slot_jobs = defaultdict(list)
# starts = {}
# for j in J:
#     for t in range(1, T+1):
#         if pl.value(s[j, t]) > 0.5:
#             starts[j] = t
#             break
#     # 실행 창 표시
#     dj = d[j]
#     for tt in range(starts[j], starts[j] + dj):
#         slot_jobs[tt].append(name[j])

# # 재파티션 슬롯
# reconfigs = [t for t in range(1, T+1) if pl.value(z[t]) > 0.5]
# num_recfg = len(reconfigs)
# T_max = int(pl.value(Tmax))
# makespan_min = slot_minutes * T_max + delta * num_recfg

# print("\n=== 슬롯별 구성/배치 ===")
# for t in range(1, T_max + 1):
#     jobs_here = slot_jobs.get(t, [])
#     if t == 1:
#         tag = "(init→" + chosen_cfg[t] + (")" if chosen_cfg[t] != init_config else ")")
#     else:
#         tag = chosen_cfg[t]
#     rmark = " [a]" if t in reconfigs else ""
#     print(f"t={t:>2}: cfg={tag:<10} jobs={jobs_here}{rmark}")

# print("\n=== 지표 ===")
# print(f"T_max(slot)      = {T_max}")
# print(f"#reconfig        = {num_recfg} at slots {reconfigs}")
# print(f"Makespan (min)   = {makespan_min:.2f}")

# # 완료시각(분)과 평균 JCT
# # 재파티션 누계(슬롯 t까지 몇 번 발생했는지)
# cum_rc = {}
# acc = 0
# for t in range(1, T_max + 1):
#     if t in reconfigs:
#         acc += 1
#     cum_rc[t] = acc

# finish = {}
# for j in J:
#     t0 = starts[j]
#     tend = t0 + d[j] - 1
#     finish[name[j]] = slot_minutes * tend + delta * cum_rc[tend]

# print("\n=== 완료시각(분) ===")
# for k, v in sorted(finish.items(), key=lambda x: x[1]):
#     print(f"{k:>3}: {v:.2f}")

# avg_jct = sum(finish.values()) / len(finish)
# print(f"\nAvg JCT (min)    = {avg_jct:.2f}")