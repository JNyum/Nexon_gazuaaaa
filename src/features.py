"""샷 궤적(yaw/pitch 25프레임) -> 행동 특징. 모든 계산은 행 단위로 벡터화."""
import numpy as np
import pandas as pd

from . import config as C

FEATURE_COLS = [
    "yaw_speed_mean", "pitch_speed_mean", "speed_max", "pre_speed_mean",
    "post_speed_mean", "pre_shot_decel", "jerk_mean", "snap_ratio",
    "yaw_step_corr", "pitch_step_corr", "yaw_pitch_corr", "straightness",
    "yaw_dir_changes", "pitch_dir_changes", "idle_ratio", "hit",
]


def _safe_div(a, b, fill):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(a.shape, float(fill)), where=b > 0)


def _corr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    return _safe_div((x * y).sum(1), np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1)), 0)


def _lag1_corr(x):
    """연속 프레임 간 상관. 논문 Table VII의 step correlation."""
    return _corr(x[:, :-1], x[:, 1:])


def _sign_flips(x):
    """방향 전환 횟수(0 프레임은 건너뜀). 오버슈트·보정 동작 지표."""
    s = np.sign(x)
    out = np.zeros(len(x))
    for i, r in enumerate(s):
        nz = r[r != 0]
        out[i] = np.count_nonzero(nz[1:] != nz[:-1])
    return out


def add_features(shots: pd.DataFrame) -> pd.DataFrame:
    Y = shots[C.YAW_COLS].to_numpy(float)
    P = shots[C.PITCH_COLS].to_numpy(float)
    speed = np.hypot(Y, P)
    pre = slice(0, C.FRAMES_BEFORE)
    post = slice(C.FRAMES_BEFORE, C.WINDOW)
    last5 = slice(C.FRAMES_BEFORE - 5, C.FRAMES_BEFORE)
    path = speed.sum(1)
    pre_mean = speed[:, pre].mean(1)

    f = pd.DataFrame(index=shots.index)
    f["yaw_speed_mean"] = np.abs(Y).mean(1)
    f["pitch_speed_mean"] = np.abs(P).mean(1)
    f["speed_max"] = speed.max(1)
    f["pre_speed_mean"] = pre_mean
    f["post_speed_mean"] = speed[:, post].mean(1)
    f["pre_shot_decel"] = _safe_div(speed[:, last5].mean(1), pre_mean, 1)   # 작을수록 멈추고 쏨
    f["jerk_mean"] = np.abs(np.diff(speed, n=2, axis=1)).mean(1)
    f["snap_ratio"] = _safe_div(speed.max(1), path, 0)                      # 한 프레임에 몰린 이동 비중
    f["yaw_step_corr"] = _lag1_corr(Y)
    f["pitch_step_corr"] = _lag1_corr(P)
    f["yaw_pitch_corr"] = _corr(Y, P)
    f["straightness"] = _safe_div(np.hypot(Y.sum(1), P.sum(1)), path, 0)
    f["yaw_dir_changes"] = _sign_flips(Y)
    f["pitch_dir_changes"] = _sign_flips(P)
    f["idle_ratio"] = (speed == 0).mean(1)
    return pd.concat([shots, f], axis=1)
