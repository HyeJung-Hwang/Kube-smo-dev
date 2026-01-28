#!/usr/bin/env python3
"""
RAN HP Scaling 스크립트
RAN 예측 데이터 기반으로 HP-scale-up/down 배포

Usage:
    python deploy_ran_scaling.py --start 09:40 --end 10:10 [--endpoint URL] [--speed SPEED] [--threshold 4.0]
"""

import argparse
import time
import requests
import pandas as pd

CSV_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/RealData_inference_with_uncertainty_group15.csv"
TARGET_JOB_ID = "hp-ran-001"

def parse_time(time_str):
    """HH:MM 형식 파싱 → 초"""
    parts = time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1])
    return hour * 3600 + minute * 60

def submit_initial_hp(endpoint):
    """초기 HP-RAN 배포"""
    data = {
        "job_id": TARGET_JOB_ID,
        "name": "RAN-L1",
        "job_type": "HP",
        "workload_type": "RAN",
        "req": 3,
        "duration": 60.0
    }
    try:
        resp = requests.post(f"{endpoint}/submit", json=data, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def submit_scale_up(endpoint, new_req=4):
    """RAN Scale-Up 제출"""
    job_id = f"hp-ran-scaleup-{int(time.time()*1000)}"
    data = {
        "job_id": job_id,
        "name": "RAN-ScaleUp",
        "job_type": "HP-scale-up",
        "workload_type": "RAN",
        "req": new_req,
        "duration": 60.0,
        "target_job_id": TARGET_JOB_ID
    }
    try:
        resp = requests.post(f"{endpoint}/submit", json=data, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def submit_scale_down(endpoint, new_req=3):
    """RAN Scale-Down 제출"""
    job_id = f"hp-ran-scaledown-{int(time.time()*1000)}"
    data = {
        "job_id": job_id,
        "name": "RAN-ScaleDown",
        "job_type": "HP-scale-down",
        "workload_type": "RAN",
        "req": new_req,
        "duration": 60.0,
        "target_job_id": TARGET_JOB_ID
    }
    try:
        resp = requests.post(f"{endpoint}/submit", json=data, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="RAN HP Scaling based on prediction")
    parser.add_argument("--start", required=True, help="Start time (HH:MM)")
    parser.add_argument("--end", required=True, help="End time (HH:MM)")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--speed", type=int, default=1, help="1초=N초 시뮬레이션")
    parser.add_argument("--threshold", type=float, default=0.5, help="Scale-up 임계값 (y_pred)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    print("=" * 50)
    print(" RAN HP Scaling Script")
    print(f" Time: {args.start} ~ {args.end}")
    print(f" Threshold: {args.threshold} (y_pred)")
    print(f" Speed: {args.speed}x")
    print("=" * 50)

    # CSV 로드
    df = pd.read_csv(CSV_PATH)
    # TIMESTAMP 형식: "2024-05-18 09:40:00+00:00"
    df['time_sec'] = df['TIMESTAMP'].apply(lambda x: parse_time(x.split()[1][:5]))

    # 실행 범위 (start ~ end)
    filtered = df[(df['time_sec'] >= start_sec) & (df['time_sec'] <= end_sec)].copy()
    filtered = filtered.sort_values('time_sec').reset_index(drop=True)

    # 15분 lookahead용 확장 데이터 (start ~ end + 15분)
    lookahead_sec = 15 * 60  # 15분
    extended = df[(df['time_sec'] >= start_sec) & (df['time_sec'] <= end_sec + lookahead_sec)].copy()
    extended = extended.sort_values('time_sec').reset_index(drop=True)

    print(f"\n실행 범위: {len(filtered)}개, 확장 데이터(+15분): {len(extended)}개")
    print("-" * 70)
    print(f"{'Time':<8} {'y_pred':<8} {'upper_95':<10} | {'15분뒤':<8} {'y_pred':<8} {'upper_95':<10} {'Size':<6} {'Action'}")
    print("-" * 70)

    # 미리보기용 상태 시뮬레이션
    preview_size = 3
    preview_last_scaleup_time = None
    scale_down_threshold = args.threshold * 0.7

    for idx, row in filtered.iterrows():
        ts = row['TIMESTAMP'].split()[1][:5]
        current_time_sec = row['time_sec']

        # 15분 뒤 데이터 찾기
        future_time_sec = row['time_sec'] + lookahead_sec
        future_row = extended[extended['time_sec'] == future_time_sec]

        if len(future_row) > 0:
            future_row = future_row.iloc[0]
            future_ts = future_row['TIMESTAMP'].split()[1][:5]
            future_y_pred = future_row['y_pred']
            future_upper = future_row['upper_bound_95']

            # Cooldown 체크
            in_cooldown = False
            if preview_last_scaleup_time is not None:
                if current_time_sec - preview_last_scaleup_time < lookahead_sec:
                    in_cooldown = True

            # Action 결정
            if future_y_pred > args.threshold and preview_size < 4:
                action = "SCALE-UP"
                preview_size = 4
                preview_last_scaleup_time = current_time_sec
            elif in_cooldown:
                action = "MAINTAIN"
            elif future_upper <= scale_down_threshold and preview_size > 3:
                action = "SCALE-DOWN"
                preview_size = 3
                preview_last_scaleup_time = None
            else:
                action = "-"

            size_str = f"{preview_size}g"
            print(f"{ts:<8} {row['y_pred']:<8.2f} {row['upper_bound_95']:<10.2f} | {future_ts:<8} {future_y_pred:<8.2f} {future_upper:<10.2f} {size_str:<6} {action}")
        else:
            size_str = f"{preview_size}g"
            print(f"{ts:<8} {row['y_pred']:<8.2f} {row['upper_bound_95']:<10.2f} | {'(no data)':<30} {size_str:<6} -")

    print("-" * 50)
    input("\nEnter를 눌러 시작...")

    # 초기 HP-RAN 배포
    print("\n>> [초기] HP-RAN (3G) 배포")
    if not args.dry_run:
        result = submit_initial_hp(args.endpoint)
        print(f"   → {result.get('message', result)}")
    else:
        print("   → [DRY-RUN]")
    time.sleep(2)

    # 실행
    base_time = filtered.iloc[0]['time_sec']
    prev_time = base_time
    current_size = 3  # 초기 RAN 크기
    last_scale_up_time = None  # 마지막 scale-up 시간

    for idx in range(len(filtered)):
        row = filtered.iloc[idx]
        wait_sim = row['time_sec'] - prev_time
        wait_real = wait_sim / args.speed

        if wait_real > 0:
            print(f"\n... 대기 {wait_real:.1f}초...")
            if not args.dry_run:
                time.sleep(wait_real)

        ts = row['TIMESTAMP'].split()[1][:5]
        current_time_sec = row['time_sec']

        # 15분 뒤 예측값 참조 (long-term prediction)
        future_time_sec = row['time_sec'] + lookahead_sec
        future_data = extended[extended['time_sec'] == future_time_sec]

        if len(future_data) > 0:
            future_row = future_data.iloc[0]
            future_ts = future_row['TIMESTAMP'].split()[1][:5]
            y_pred = future_row['y_pred']
            upper_95 = future_row['upper_bound_95']
            print(f"   [{ts}] 현재 | 15분뒤({future_ts}) 예측: y_pred={y_pred:.4f}, upper_95={upper_95:.4f}")
        else:
            # 15분 뒤 데이터 없으면 스킵
            print(f"   [{ts}] (15분뒤 데이터 없음, 스킵)")
            prev_time = row['time_sec']
            continue

        scale_down_threshold = args.threshold * 0.7

        # Cooldown 체크: scale-up 후 15분간은 scale-down 금지
        in_cooldown = False
        if last_scale_up_time is not None:
            elapsed_since_scaleup = current_time_sec - last_scale_up_time
            if elapsed_since_scaleup < lookahead_sec:
                in_cooldown = True
                remaining = (lookahead_sec - elapsed_since_scaleup) / 60
                print(f"   [COOLDOWN] Scale-up 유지 중 (남은 시간: {remaining:.1f}분)")

        if y_pred > args.threshold and current_size < 4:
            print(f"\n>> [{ts}] RAN Scale-Up! (15분뒤 y_pred={y_pred:.4f} > {args.threshold}) 3G->4G")
            if not args.dry_run:
                result = submit_scale_up(args.endpoint, new_req=4)
                print(f"   -> {result.get('message', result)}")
            current_size = 4
            last_scale_up_time = current_time_sec  # cooldown 시작
        elif upper_95 <= scale_down_threshold and current_size > 3 and not in_cooldown:
            print(f"\n<< [{ts}] RAN Scale-Down! (15분뒤 upper_95={upper_95:.4f} <= {scale_down_threshold:.4f}) 4G->3G")
            if not args.dry_run:
                result = submit_scale_down(args.endpoint, new_req=3)
                print(f"   -> {result.get('message', result)}")
            current_size = 3
            last_scale_up_time = None  # cooldown 해제

        prev_time = row['time_sec']

    print("\n완료!")

if __name__ == "__main__":
    main()
