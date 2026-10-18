"""
K-Medoids 군집 (비지도). 라벨(label, is_cheat)은 이 파일에서 읽지 않는다.

    python -m src.clustering            # K 자동 선택 (실루엣)
    python -m src.clustering --k 4      # K 고정

입력  data/processed/shots.parquet, games.parquet(episode 메타만 사용)
출력  data/processed/clusters.parquet   샷별 클러스터, medoid까지 거리, 의심 여부
      data/processed/cluster_model.json  K, 가중치, medoid 샷, 클러스터별 기준선 비중, K 탐색 결과

의심 집단 판정 (라벨 없이):
  episode0(정상 10분 고정 판)을 "평소 행동" 기준선으로 둔다. 기준선은 군집 학습에 넣지 않고
  학습된 medoid에 할당만 한다. 메인 구간에서 차지하는 비중에 비해 기준선 샷이
  거의 들어오지 않는 클러스터 = 평소 유저에게서 보기 드문 행동 = 의심 집단.
"""
import argparse
import json
import logging

import kmedoids
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import RobustScaler

from . import config as C
from .features import FEATURE_COLS

log = logging.getLogger("clustering")
META_COLS = ["game_id", "episode", "randomized"]   # 라벨 컬럼은 의도적으로 제외


def load():
    shots = pd.read_parquet(C.PROCESSED_DIR / "shots.parquet")
    games = pd.read_parquet(C.PROCESSED_DIR / "games.parquet", columns=META_COLS)
    df = shots.merge(games, on="game_id", how="left")
    main = df[df.randomized].reset_index(drop=True)
    base = df[df.episode == C.BASELINE_EPISODE].reset_index(drop=True)
    return main, base


def embed(main, base, weights):
    """RobustScaler(메인 구간 기준 학습) 후 특징별 가중치 곱."""
    w = np.array([weights[c] for c in FEATURE_COLS])
    scaler = RobustScaler().fit(main[FEATURE_COLS])
    Xm = scaler.transform(main[FEATURE_COLS]) * w
    Xb = scaler.transform(base[FEATURE_COLS]) * w
    return Xm.astype(np.float32), Xb.astype(np.float32)


def fit_kmedoids(X_sample, k, seed):
    D = pairwise_distances(X_sample, metric="euclidean", n_jobs=-1)
    res = kmedoids.fasterpam(D, k, init="build", random_state=seed)
    return np.asarray(res.medoids), np.asarray(res.labels), float(res.loss), D


def assign(X, M):
    d = pairwise_distances(X, M)
    return d.argmin(1), d.min(1)


def run(k=None, weights=None, sample=C.FIT_SAMPLE, seed=C.SEED, save=True, data=None):
    """save=False면 파일을 쓰지 않고 (model, clusters) 반환 (대시보드 재군집용)."""
    weights = weights or C.FEATURE_WEIGHTS
    main, base = data if data is not None else load()
    main, base = main.copy(), base.copy()
    Xm, Xb = embed(main, base, weights)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xm), size=min(sample, len(Xm)), replace=False)
    Xs = Xm[idx]

    # ---- K 탐색: 라벨 없이 실루엣 점수로 선택 ----
    sweep, fits = [], {}
    for kk in ([k] if k else C.K_RANGE):
        med, lab, loss, D = fit_kmedoids(Xs, kk, seed)
        sil = float(silhouette_score(D, lab, metric="precomputed"))
        sweep.append({"k": kk, "silhouette": round(sil, 4), "loss": round(loss, 1)})
        fits[kk] = med
        log.info("k=%d silhouette=%.4f", kk, sil)
    best_k = k or max(sweep, key=lambda r: r["silhouette"])["k"]
    medoid_rows = idx[fits[best_k]]
    M = Xm[medoid_rows]

    # ---- 전체 할당 ----
    main["cluster"], main["dist_to_medoid"] = assign(Xm, M)
    base["cluster"], base["dist_to_medoid"] = assign(Xb, M)

    # ---- 의심 집단 판정: 기준선 대비 과대표 클러스터 ----
    main_share = main.cluster.value_counts(normalize=True).reindex(range(best_k), fill_value=0)
    base_share = base.cluster.value_counts(normalize=True).reindex(range(best_k), fill_value=0)
    ratio = (base_share / main_share.replace(0, np.nan)).fillna(0)
    suspicious = sorted(int(c) for c in ratio.index[ratio < C.SUSPICIOUS_RATIO])
    main["suspicious"] = main.cluster.isin(suspicious).astype(int)
    base["suspicious"] = base.cluster.isin(suspicious).astype(int)

    # ---- 저장 ----
    main["split"], base["split"] = "main", "baseline"
    keep = ["game_id", "frame", "shot_idx", "split", "cluster", "dist_to_medoid", "suspicious"]
    out = pd.concat([main[keep], base[keep]], ignore_index=True)

    model = {
        "k": int(best_k),
        "weights": weights,
        "fit_sample": int(len(idx)),
        "seed": seed,
        "suspicious_ratio_threshold": C.SUSPICIOUS_RATIO,
        "suspicious_clusters": suspicious,
        "k_sweep": sweep,
        "clusters": [
            {
                "cluster": c,
                "medoid_game_id": main.loc[medoid_rows[c], "game_id"],
                "medoid_frame": int(main.loc[medoid_rows[c], "frame"]),
                "main_share": round(float(main_share[c]), 4),
                "baseline_share": round(float(base_share[c]), 4),
                "baseline_ratio": round(float(ratio[c]), 3),
                "suspicious": c in suspicious,
                "medoid_features": {f: round(float(main.loc[medoid_rows[c], f]), 4) for f in FEATURE_COLS},
            }
            for c in range(best_k)
        ],
    }
    if save:
        out.to_parquet(C.PROCESSED_DIR / "clusters.parquet", index=False)
        with open(C.PROCESSED_DIR / "cluster_model.json", "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
    return model, out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--sample", type=int, default=C.FIT_SAMPLE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    m, _ = run(k=a.k, sample=a.sample, seed=a.seed)
    log.info("선택된 K=%d, 의심 클러스터=%s", m["k"], m["suspicious_clusters"])
    for c in m["clusters"]:
        log.info("  cluster %d  메인 %.1f%%  기준선 %.1f%%  ratio %.2f %s", c["cluster"],
                 100 * c["main_share"], 100 * c["baseline_share"], c["baseline_ratio"],
                 "← 의심" if c["suspicious"] else "")


if __name__ == "__main__":
    main()
