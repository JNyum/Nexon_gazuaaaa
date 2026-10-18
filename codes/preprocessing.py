#!/usr/bin/env python3
"""
preprocess.py
GAN-Aimbots data_collection_3 JSON -> 샷 단위 / 경기 단위 테이블

사용법:
    python preprocess.py --data_dir ./data_collection_3 --out_dir ./processed

출력:
    processed/shots.csv   샷 1개 = 1행 (원시 궤적 50개 + 엔지니어링 특징 + 메타/라벨)
    processed/games.csv   경기 1개 = 1행 (메타/라벨, 샷 수, 명중률)

라벨 컬럼(label, is_cheat)은 군집 학습에 쓰지 않고 평가에만 사용한다.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 상수 (원본 feature_extraction.py와 동일하게 맞춤)
# ---------------------------------------------------------------------------
FPS = 35
FRAMES_BEFORE = int(0.5 * FPS)   # 17 프레임 (사격 전 0.5초)
FRAMES_AFTER = int(0.25 * FPS)   # 8 프레임 (사격 후 0.25초)
WINDOW = FRAMES_BEFORE + FRAMES_AFTER  # 25

YAW_IDX, PITCH_IDX = 4, 5        # actions 안의 좌우/상하 회전량 위치

# 무기 번호 -> 1발당 소모 탄약 (슈퍼샷건은 2발씩 소모)
AMMO_PER_SHOT = {1: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1}
SKIP_WEAPONS = {5, 6}            # 로켓런처, 플라즈마건: 조준 방식이 달라 제외 (원본과 동일)

# 3~6판은 라벨이 무작위 순서로 배정된 구간 -> 메인 분석용
RANDOMIZED_EPISODES = {"episode2", "episode3", "episode4", "episode5"}

LABEL_MAP = {"none": "normal", "null": "normal",
             "ease_light": "light", "ease_strong": "strong",
             "gan_group0": "gan", "gan_group1": "gan"}

YAW_COLS = [f"yaw_{i:02d}" for i in range(WINDOW)]
PITCH_COLS = [f"pitch_{i:02d}" for i in range(WINDOW)]


# ---------------------------------------------------------------------------
# 1. 로드
# ---------------------------------------------------------------------------
def load_game(path):
    with open(path) as f:
        d = json.load(f)
    name = os.path.basename(path).replace(".json", "")
    episode = name.split("_")[-1]
    raw_label = str(d.get("aimbot")).lower()
    meta = {
        "game_id": name,
        "hwid": str(d["hwid"]),
        "episode": episode,
        "randomized": episode in RANDOMIZED_EPISODES,
        "label": LABEL_MAP.get(raw_label, raw_label),
        "fps_familiarity": d.get("player-fps-familiarity"),
        "n_frames": len(d["actions"]),
        "final_frags": d["frags"][-1],
        "final_deaths": d["deaths"][-1],
    }
    meta["is_cheat"] = int(meta["label"] != "normal")
    return d, meta


# ---------------------------------------------------------------------------
# 2. 사격 시점 탐지 + 궤적 창 자르기
# ---------------------------------------------------------------------------
def find_shots(d):
    """탄약이 무기별 1발 소모량만큼 줄어든 프레임 = 사격."""
    ammos, weaps = d["ammos"], d["weapons"]
    shots = []
    for i in range(1, len(ammos)):
        w = int(weaps[i])
        if w in SKIP_WEAPONS:
            continue
        if ammos[i - 1] - ammos[i] == AMMO_PER_SHOT.get(w, 1):
            shots.append(i)
    return shots


def extract_windows(d, meta):
    actions = np.asarray(d["actions"], dtype=np.float32)
    yaw_all, pitch_all = actions[:, YAW_IDX], actions[:, PITCH_IDX]
    damages = np.asarray(d["damages"])
    weapons = np.asarray(d["weapons"])

    rows, prev = [], None
    for k, s in enumerate(find_shots(d)):
        lo, hi = s - FRAMES_BEFORE, s + FRAMES_AFTER
        if lo < 0 or hi > len(actions):
            continue
        row = dict(meta)
        row.update({
            "shot_idx": k,
            "frame": s,
            "time_sec": s / FPS,
            "weapon": int(weapons[s]),
            "hit": int(damages[s] != damages[s - 1]),
            # 연사 시 창이 겹치므로 직전 사격과의 간격을 함께 기록
            "frames_since_prev_shot": (s - prev) if prev is not None else np.nan,
        })
        row.update(dict(zip(YAW_COLS, yaw_all[lo:hi])))
        row.update(dict(zip(PITCH_COLS, pitch_all[lo:hi])))
        rows.append(row)
        prev = s
    return rows


# ---------------------------------------------------------------------------
# 3. 궤적 -> 특징 (벡터화)
# ---------------------------------------------------------------------------
def _lag1_corr(x):
    """행별 lag-1 자기상관. 움직임이 없어 분산이 0이면 0."""
    a, b = x[:, :-1], x[:, 1:]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.divide((a * b).sum(1), den, out=np.zeros(len(x)), where=den > 0)


def _row_corr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))
    return np.divide((x * y).sum(1), den, out=np.zeros(len(x)), where=den > 0)


def _sign_flips(x):
    """방향 전환 횟수 (0은 건너뛰고 부호가 바뀐 횟수). 오버슈트/보정 동작 지표."""
    out = np.zeros(len(x))
    for i, r in enumerate(x):
        nz = np.sign(r[r != 0])
        out[i] = (nz[1:] != nz[:-1]).sum() if len(nz) > 1 else 0
    return out


def add_features(df):
    Y = df[YAW_COLS].to_numpy(dtype=np.float64)
    P = df[PITCH_COLS].to_numpy(dtype=np.float64)
    speed = np.sqrt(Y ** 2 + P ** 2)              # 프레임별 2D 이동량
    pre, post = slice(0, FRAMES_BEFORE), slice(FRAMES_BEFORE, WINDOW)
    last5 = slice(FRAMES_BEFORE - 5, FRAMES_BEFORE)

    f = pd.DataFrame(index=df.index)
    # 속도
    f["yaw_speed_mean"] = np.abs(Y).mean(1)
    f["pitch_speed_mean"] = np.abs(P).mean(1)
    f["speed_max"] = speed.max(1)
    f["pre_speed_mean"] = speed[:, pre].mean(1)
    f["post_speed_mean"] = speed[:, post].mean(1)
    # 사격 직전 감속: 마지막 5프레임 속도 / 사격 전 전체 평균 (작을수록 멈추고 쏨)
    f["pre_shot_decel"] = np.divide(speed[:, last5].mean(1), f["pre_speed_mean"],
                                    out=np.ones(len(df)), where=f["pre_speed_mean"].to_numpy() > 0)
    # 가속 변화(부드러움). 봇은 한 프레임에 큰 각도를 꺾는 경향
    f["jerk_mean"] = np.abs(np.diff(speed, n=2, axis=1)).mean(1)
    f["snap_ratio"] = np.divide(speed.max(1), speed.sum(1),
                                out=np.zeros(len(df)), where=speed.sum(1) > 0)
    # 연속성 (논문 Table VII: 사람 pitch 0.66 vs 휴리스틱 봇 0.04)
    f["yaw_step_corr"] = _lag1_corr(Y)
    f["pitch_step_corr"] = _lag1_corr(P)
    f["yaw_pitch_corr"] = _row_corr(Y, P)
    # 경로 형태
    path = speed.sum(1)
    net = np.sqrt(Y.sum(1) ** 2 + P.sum(1) ** 2)
    f["straightness"] = np.divide(net, path, out=np.zeros(len(df)), where=path > 0)
    f["yaw_dir_changes"] = _sign_flips(Y)
    f["pitch_dir_changes"] = _sign_flips(P)
    f["idle_ratio"] = (speed == 0).mean(1)
    return pd.concat([df, f], axis=1)


FEATURE_COLS = ["yaw_speed_mean", "pitch_speed_mean", "speed_max", "pre_speed_mean",
                "post_speed_mean", "pre_shot_decel", "jerk_mean", "snap_ratio",
                "yaw_step_corr", "pitch_step_corr", "yaw_pitch_corr", "straightness",
                "yaw_dir_changes", "pitch_dir_changes", "idle_ratio", "hit"]


# ---------------------------------------------------------------------------
# 4. 실행
# ---------------------------------------------------------------------------
def build(data_dir):
    rows, games = [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        d, meta = load_game(path)
        r = extract_windows(d, meta)
        rows.extend(r)
        games.append(meta)

    shots = add_features(pd.DataFrame(rows))
    # 이상값 정리: 원시 궤적에 NaN/inf가 없는지 확인
    assert np.isfinite(shots[YAW_COLS + PITCH_COLS + FEATURE_COLS].to_numpy()).all()

    g = pd.DataFrame(games)
    agg = shots.groupby("game_id").agg(n_shots=("hit", "size"), hit_rate=("hit", "mean"))
    g = g.merge(agg, on="game_id", how="left").fillna({"n_shots": 0})
    return shots, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="processed")
    args = ap.parse_args()

    shots, games = build(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    shots.to_csv(os.path.join(args.out_dir, "shots.csv"), index=False)
    games.to_csv(os.path.join(args.out_dir, "games.csv"), index=False)

    main_set = shots[shots.randomized]
    print(f"games {len(games)}  shots {len(shots)}  (메인: 3~6판 {main_set.game_id.nunique()}경기 / {len(main_set)}샷)")
    print("\n[메인 구간 라벨별 샷 수]")
    print(main_set.label.value_counts().to_string())
    print("\n[메인 구간 라벨별 특징 평균]  ※ 군집 전 H1 sanity check")
    order = ["normal", "light", "strong", "gan"]
    print(main_set.groupby("label")[FEATURE_COLS].mean().reindex(order).T.round(3).to_string())


if __name__ == "__main__":
    main()
