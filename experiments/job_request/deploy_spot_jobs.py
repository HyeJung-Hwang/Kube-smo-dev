#!/usr/bin/env python3
"""
Spot Job 배포 스크립트
CSV에서 지정한 시간 범위의 Spot job들을 시간 순서대로 배포

Usage:
    python deploy_spot_jobs.py --start 09:40 --end 10:10 [--endpoint URL] [--speed SPEED]
"""

import argparse
import time
import requests
import pandas as pd
from datetime import datetime

CSV_PATH = "/home/skt6g/AI-RAN/KubeSMO/data/single_gpu_a100_singleworker_day79_with_mig_capped.csv"

def parse_time(time_str):
    """HH:MM 또는 HH:MM:SS 형식 파싱"""
    parts = time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    return hour * 3600 + minute * 60 + second

def format_time(seconds):
    """초를 HH:MM:SS 형식으로"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def submit_job(endpoint, job_id, name, job_type, workload_type, req, duration):
    """Job 제출"""
    url = f"{endpoint}/submit"
    data = {
        "job_id": job_id,
        "name": name,
        "job_type": job_type,
        "workload_type": workload_type,
        "req": req,
        "duration": duration
    }
    try:
        resp = requests.post(url, json=data, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Deploy Spot jobs from CSV")
    parser.add_argument("--start", required=True, help="Start time (HH:MM or HH:MM:SS)")
    parser.add_argument("--end", required=True, help="End time (HH:MM or HH:MM:SS)")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--speed", type=int, default=1, help="Time compression (default: 60, 1초=1분)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually send requests")
    parser.add_argument("--job-type", default="Spot", help="Filter by job_type (default: Spot, use 'all' for all types)")
    args = parser.parse_args()

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    print("=" * 50)
    print(f" Spot Job Deployment Script")
    print(f" Time Range: {args.start} ~ {args.end}")
    print(f" Endpoint: {args.endpoint}")
    print(f" Speed: {args.speed}x (1 real second = {args.speed} sim seconds)")
    print(f" Job Type Filter: {args.job_type}")
    print(f" Dry Run: {args.dry_run}")
    print("=" * 50)

    # CSV 로드
    df = pd.read_csv(CSV_PATH)

    # time_of_day 컬럼을 초 단위로 변환
    df['time_sec'] = df['time_of_day'].apply(parse_time)

    # 시간 범위 필터링
    mask = (df['time_sec'] >= start_sec) & (df['time_sec'] <= end_sec)

    # job_type 필터링
    if args.job_type.lower() != 'all':
        mask = mask & (df['job_type'] == args.job_type)

    filtered_df = df[mask].sort_values('time_sec').reset_index(drop=True)

    print(f"\n총 {len(filtered_df)}개 job 발견")
    print("-" * 50)

    if len(filtered_df) == 0:
        print("해당 시간대에 job이 없습니다.")
        return

    # 첫 번째 job 시간을 기준으로 상대 시간 계산
    base_time_sec = filtered_df.iloc[0]['time_sec']

    print(f"{'No':<4} {'Time':<12} {'Type':<6} {'MIG':<4} {'Duration':<10} {'Job Name'}")
    print("-" * 50)
    for idx, row in filtered_df.iterrows():
        rel_time = row['time_sec'] - base_time_sec
        duration_min = row['duration'] / 60
        print(f"{idx:<4} {row['time_of_day']:<12} {row['job_type']:<6} {int(row['mig_size'])}g   {duration_min:>6.1f}min   {row['job_name']}")

    print("-" * 50)
    input("\nEnter를 눌러 배포 시작...")

    # 배포 시작
    experiment_start = time.time()
    prev_time_sec = base_time_sec

    for idx, row in filtered_df.iterrows():
        current_time_sec = row['time_sec']

        # 대기 시간 계산
        wait_sim_sec = current_time_sec - prev_time_sec
        wait_real_sec = wait_sim_sec / args.speed

        if wait_real_sec > 0:
            print(f"\n... 대기 {wait_real_sec:.1f}초 (sim: {wait_sim_sec}초)...")
            if not args.dry_run:
                time.sleep(wait_real_sec)

        # Job 배포
        job_id = f"spot-{int(time.time() * 1000)}"
        mig_size = int(row['mig_size'])
        duration_min = row['duration'] / 60

        elapsed = time.time() - experiment_start
        print(f"\n[{elapsed:>6.1f}s] >> {row['time_of_day']} | {row['job_type']} | {mig_size}g | {duration_min:.1f}min")

        if not args.dry_run:
            result = submit_job(
                endpoint=args.endpoint,
                job_id=job_id,
                name=f"Spot-{row['job_name']}",
                job_type="Spot",
                workload_type="AI",
                req=mig_size,
                duration=duration_min
            )
            print(f"         → {result.get('message', result)}")
        else:
            print(f"         → [DRY-RUN] job_id={job_id}, req={mig_size}g, duration={duration_min:.1f}min")

        prev_time_sec = current_time_sec

    total_elapsed = time.time() - experiment_start
    print("\n" + "=" * 50)
    print(f" 완료! 총 소요시간: {total_elapsed:.1f}초")
    print(f" 배포된 job: {len(filtered_df)}개")
    print("=" * 50)

if __name__ == "__main__":
    main()
