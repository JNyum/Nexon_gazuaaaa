"""원본 JSON 로드, 스키마 검증, 사격 구간 추출."""
import json
from pathlib import Path

import numpy as np

from . import config as C


class SchemaError(ValueError):
    pass


def validate(d: dict, name: str) -> None:
    missing = [k for k in C.REQUIRED_KEYS if k not in d]
    if missing:
        raise SchemaError(f"{name}: 필수 키 누락 {missing}")
    lengths = {k: len(d[k]) for k in C.PER_FRAME_KEYS}
    if len(set(lengths.values())) != 1:
        raise SchemaError(f"{name}: 프레임 길이 불일치 {lengths}")
    if lengths["actions"] == 0:
        raise SchemaError(f"{name}: 빈 기록")
    width = {len(a) for a in d["actions"]}
    if min(width) <= C.PITCH_IDX:
        raise SchemaError(f"{name}: actions 차원 부족 {width}")


def game_meta(d: dict, path: Path) -> dict:
    """파일명 형식: {unix_timestamp}_{hwid}_recording_{episodeN}.json"""
    stem = path.stem
    episode = stem.split("_")[-1]
    raw_label = str(d.get("aimbot")).lower()
    label = C.LABEL_MAP.get(raw_label)
    if label is None:
        raise SchemaError(f"{stem}: 알 수 없는 aimbot 라벨 '{raw_label}'")
    return {
        "game_id": stem,
        "hwid": str(d["hwid"]),
        "episode": episode,
        "randomized": episode in C.RANDOMIZED_EPISODES,
        "label": label,
        "is_cheat": int(label != "normal"),
        "fps_familiarity": d.get("player-fps-familiarity"),
        "n_frames": len(d["actions"]),
        "duration_sec": round(len(d["actions"]) / C.FPS, 1),
        "final_frags": d["frags"][-1],
        "final_deaths": d["deaths"][-1],
    }


def find_shots(d: dict) -> list[int]:
    """탄약이 무기별 1발 소모량만큼 줄어든 프레임 = 사격."""
    ammos, weaps = d["ammos"], d["weapons"]
    shots = []
    for i in range(1, len(ammos)):
        w = int(weaps[i])
        if w in C.SKIP_WEAPONS:
            continue
        if ammos[i - 1] - ammos[i] == C.AMMO_PER_SHOT.get(w, 1):
            shots.append(i)
    return shots


def extract_shots(d: dict, meta: dict) -> tuple[list[dict], int]:
    """사격마다 전후 궤적 창을 잘라 행으로 만든다. (행 목록, 경계에 걸려 버린 수)"""
    actions = np.asarray(d["actions"], dtype=np.float32)
    yaw, pitch = actions[:, C.YAW_IDX], actions[:, C.PITCH_IDX]
    damages = np.asarray(d["damages"])
    weapons = np.asarray(d["weapons"])

    rows, dropped, prev = [], 0, None
    for k, s in enumerate(find_shots(d)):
        lo, hi = s - C.FRAMES_BEFORE, s + C.FRAMES_AFTER
        if lo < 0 or hi > len(actions):
            dropped += 1
            continue
        row = {
            "game_id": meta["game_id"],
            "shot_idx": k,
            "frame": s,
            "time_sec": round(s / C.FPS, 3),
            "weapon": int(weapons[s]),
            "hit": int(damages[s] != damages[s - 1]),
            # 연사 시 창이 겹치므로 직전 사격과의 간격을 함께 기록
            "frames_since_prev_shot": (s - prev) if prev is not None else np.nan,
        }
        row.update(zip(C.YAW_COLS, yaw[lo:hi].tolist()))
        row.update(zip(C.PITCH_COLS, pitch[lo:hi].tolist()))
        rows.append(row)
        prev = s
    return rows, dropped


def load_file(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
