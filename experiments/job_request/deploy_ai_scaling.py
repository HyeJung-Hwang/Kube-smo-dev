#!/usr/bin/env python3
"""
AI HP Scaling 스크립트
AI 예측 데이터 기반으로 HP-scale-out/in 배포

Usage:
    python deploy_ai_scaling.py --start 09:40 --end 10:10 [--endpoint URL] [--speed SPEED] [--threshold 0.35]
"""

import argparse
import time
import requests
import pandas as pd

CSV_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/Azure_0518_inference_with_uncertainty_calibrated.csv"
TARGET_JOB_ID = "hp-ai-001"

def parse_time(time_str):
    """HH:MM 형식 파싱 → 초"""
    parts = time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1])
    return hour * 3600 + minute * 60

def submit_initial_hp(endpoint):
    """초기 HP-AI 배포"""
    data = {
        "job_id": TARGET_JOB_ID,
        "name": "AI-Inference",
        "job_type": "HP",
        "workload_type": "AI",
        "req": 1,
        "duration": 60.0  # test 시간 duration 만큼
    }
    try:
        resp = requests.post(f"{endpoint}/submit", json=data, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def submit_scale_out(endpoint, req=1):
    """AI Scale-Out 제출"""
    job_id = f"{TARGET_JOB_ID}-scaleout-{int(time.time()*1000)}"
    data = {
        "job_id": job_id,
        "name": "AI-ScaleOut",
        "job_type": "HP-scale-out",
        "workload_type": "AI",
        "req": req,
        "duration": 60.0,
        "target_job_id": TARGET_JOB_ID
    }
    try:
        resp = requests.post(f"{endpoint}/submit", json=data, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="AI HP Scaling based on prediction")
    parser.add_argument("--start", required=True, help="Start time (HH:MM)")
    parser.add_argument("--end", required=True, help="End time (HH:MM)")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--speed", type=int, default=1, help="1초=N초 시뮬레이션")
    parser.add_argument("--threshold", type=float, default=5, help="Scale-out 임계값 (upper_bound_95)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    print("=" * 50)
    print(" AI HP Scaling Script")
    print(f" Time: {args.start} ~ {args.end}")
    print(f" Threshold: {args.threshold} (upper_bound_95)")
    print(f" Speed: {args.speed}x")
    print("=" * 50)

    # CSV 로드
    df = pd.read_csv(CSV_PATH)
    df['time_sec'] = df['TIMESTAMP'].apply(lambda x: parse_time(x.split()[1][:5]))

    # 실행 범위 (start ~ end)
    filtered = df[(df['time_sec'] >= start_sec) & (df['time_sec'] <= end_sec)].copy()
    filtered = filtered.sort_values('time_sec').reset_index(drop=True)

    # 5분 lookahead용 확장 데이터 (start ~ end + 5분)
    lookahead_sec = 5 * 60  # 5분
    extended = df[(df['time_sec'] >= start_sec) & (df['time_sec'] <= end_sec + lookahead_sec)].copy()
    extended = extended.sort_values('time_sec').reset_index(drop=True)

    print(f"\n실행 범위: {len(filtered)}개, 확장 데이터(+5분): {len(extended)}개")
    print("-" * 50)
    print(f"{'Time':<8} {'y_pred':<8} {'upper_95':<10} | {'5분뒤':<8} {'y_pred':<8} {'upper_95':<10} {'Action'}")
    print("-" * 50)

    for idx, row in filtered.iterrows():
        ts = row['TIMESTAMP'].split()[1][:5]
        # 5분 뒤 데이터 찾기
        future_time_sec = row['time_sec'] + lookahead_sec
        future_row = extended[extended['time_sec'] == future_time_sec]

        if len(future_row) > 0:
            future_row = future_row.iloc[0]
            future_ts = future_row['TIMESTAMP'].split()[1][:5]
            future_upper = future_row['upper_bound_95']
            action = "SCALE-OUT" if future_upper > args.threshold else "-"
            print(f"{ts:<8} {row['y_pred_scaled']:<8.2f} {row['upper_bound_95']:<10.2f} | {future_ts:<8} {future_row['y_pred_scaled']:<8.2f} {future_upper:<10.2f} {action}")
        else:
            print(f"{ts:<8} {row['y_pred_scaled']:<8.2f} {row['upper_bound_95']:<10.2f} | (no data)")

    print("-" * 50)
    input("\nEnter를 눌러 시작...")

    # 초기 HP-AI 배포
    print("\n>> [초기] HP-AI (1G) 배포")
    if not args.dry_run:
        result = submit_initial_hp(args.endpoint)
        print(f"   → {result.get('message', result)}")
    else:
        print("   → [DRY-RUN]")
    time.sleep(2)

    # 실행
    base_time = filtered.iloc[0]['time_sec']
    prev_time = base_time
    scaled_out = False

    for idx in range(len(filtered)):
        row = filtered.iloc[idx]
        wait_sim = row['time_sec'] - prev_time
        wait_real = wait_sim / args.speed

        if wait_real > 0:
            print(f"\n... 대기 {wait_real:.1f}초...")
            if not args.dry_run:
                time.sleep(wait_real)

        ts = row['TIMESTAMP'].split()[1][:5]

        # 5분 뒤 예측값 참조 (short-term prediction)
        future_time_sec = row['time_sec'] + lookahead_sec
        future_data = extended[extended['time_sec'] == future_time_sec]

        if len(future_data) > 0:
            future_row = future_data.iloc[0]
            future_ts = future_row['TIMESTAMP'].split()[1][:5]
            upper = future_row['upper_bound_95']
            print(f"   [{ts}] 현재 | 5분뒤({future_ts}) 예측: upper_95={upper:.4f}")
        else:
            print(f"   [{ts}] (5분뒤 데이터 없음, 스킵)")
            prev_time = row['time_sec']
            continue

        if upper > args.threshold and not scaled_out:
            print(f"\n>> [{ts}] AI Scale-Out! (5분뒤 upper_95={upper:.4f} > {args.threshold})")
            if not args.dry_run:
                result = submit_scale_out(args.endpoint)
                print(f"   -> {result.get('message', result)}")
            scaled_out = True

        prev_time = row['time_sec']

    print("\n완료!")

if __name__ == "__main__":
    main()
