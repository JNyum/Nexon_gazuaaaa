"""
군집 결과 평가. 이 단계에서 처음으로 라벨을 조인한다.

    python -m src.evaluate

입력  data/processed/clusters.parquet, games.parquet
출력  data/processed/metrics.json       H1(샷 단위) / H2(경기 단위) 지표, 실제 치터 비율 시나리오
      data/processed/game_scores.parquet 경기별 의심 샷 비율과 판정 (대시보드 검토 대기열용)

H1  샷 단위: 의심 클러스터 소속 = 치팅 예측. ARI/NMI, precision/recall/F1, 봇 유형별 탐지율
    + 상한선: 라벨 다수결로 클러스터를 매핑했을 때의 F1 (비지도 판정이 얼마나 근접하는지 비교용)
H2  경기 단위: 경기별 의심 샷 비율. 임계값은 기준선(episode0) 경기 분포의 상위 분위수로 정하며
    라벨을 쓰지 않는다. ROC-AUC는 임계값과 무관한 순위 성능.
실제 비율: 데이터의 치팅 비율(75%)은 실서비스보다 훨씬 높으므로, 같은 TPR/FPR에서
    치터 비율 p일 때의 precision = TPR*p / (TPR*p + FPR*(1-p)) 를 계산한다.
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, f1_score,
                             normalized_mutual_info_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from . import config as C

log = logging.getLogger("evaluate")
GAME_THRESHOLD_QUANTILE = 0.95
PREVALENCES = [0.01, 0.05, 0.10, 0.25, 0.50]
BOT_TYPES = ["light", "strong", "gan"]


def binary_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "precision": round(precision_score(y, p, zero_division=0), 4),
        "recall": round(recall_score(y, p, zero_division=0), 4),
        "f1": round(f1_score(y, p, zero_division=0), 4),
        "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def precision_at(tpr, fpr, p):
    den = tpr * p + fpr * (1 - p)
    return round(float(tpr * p / den), 4) if den else 0.0


def shot_level(main):
    y, pred = main.is_cheat.to_numpy(), main.suspicious.to_numpy()
    res = {
        "n_shots": int(len(main)),
        "ari_label4": round(adjusted_rand_score(main.label, main.cluster), 4),
        "nmi_label4": round(normalized_mutual_info_score(main.label, main.cluster), 4),
        "ari_cheat": round(adjusted_rand_score(y, main.cluster), 4),
        "unsupervised": binary_metrics(y, pred),
        "detection_rate_by_label": main.groupby("label").suspicious.mean().round(4).to_dict(),
        "cluster_composition": pd.crosstab(main.cluster, main.label).to_dict(orient="index"),
    }
    # 상한선: 각 클러스터를 다수 라벨(치팅/정상)로 매핑
    majority = main.groupby("cluster").is_cheat.mean().gt(0.5).astype(int)
    res["oracle_majority_mapping"] = binary_metrics(y, main.cluster.map(majority).to_numpy())
    # 봇 유형별 (해당 봇 vs 정상만 남겨서)
    res["by_bot_type"] = {}
    for b in BOT_TYPES:
        sub = main[main.label.isin(["normal", b])]
        res["by_bot_type"][b] = binary_metrics(sub.is_cheat, sub.suspicious)
    return res


def game_level(clusters, games):
    g = (clusters.groupby(["game_id", "split"])
         .agg(n_shots=("suspicious", "size"), suspicious_ratio=("suspicious", "mean"),
              mean_dist=("dist_to_medoid", "mean"))
         .reset_index()
         .merge(games[["game_id", "hwid", "episode", "label", "is_cheat", "hit_rate"]], on="game_id"))

    base = g[g.split == "baseline"]
    thr = float(base.suspicious_ratio.quantile(GAME_THRESHOLD_QUANTILE))
    g["flagged"] = (g.suspicious_ratio > thr).astype(int)
    g = g.sort_values("suspicious_ratio", ascending=False).reset_index(drop=True)
    g["review_rank"] = np.where(g.split == "main", g.groupby("split").cumcount() + 1, np.nan)

    main = g[g.split == "main"]
    y, s = main.is_cheat.to_numpy(), main.suspicious_ratio.to_numpy()
    fpr_c, tpr_c, _ = roc_curve(y, s)
    res = {
        "n_games": int(len(main)),
        "threshold": round(thr, 4),
        "threshold_rule": f"baseline(episode0) 경기 의심비율 {int(GAME_THRESHOLD_QUANTILE*100)}% 분위수",
        "baseline_ratio_mean": round(float(base.suspicious_ratio.mean()), 4),
        "roc_auc": round(roc_auc_score(y, s), 4),
        "roc_curve": {"fpr": fpr_c.round(4).tolist(), "tpr": tpr_c.round(4).tolist()},
        "unsupervised": binary_metrics(y, main.flagged.to_numpy()),
        "detection_rate_by_label": main.groupby("label").flagged.mean().round(4).to_dict(),
        "ratio_by_label": main.groupby("label").suspicious_ratio.mean().round(4).to_dict(),
        "by_bot_type": {},
    }
    for b in BOT_TYPES:
        sub = main[main.label.isin(["normal", b])]
        r = binary_metrics(sub.is_cheat, sub.flagged)
        r["roc_auc"] = round(roc_auc_score(sub.is_cheat, sub.suspicious_ratio), 4)
        res["by_bot_type"][b] = r
    return res, g


def prevalence_table(metrics):
    tpr, fpr = metrics["recall"], metrics["fpr"]
    return [{"cheater_rate": p, "precision": precision_at(tpr, fpr, p)} for p in PREVALENCES]


def run():
    clusters = pd.read_parquet(C.PROCESSED_DIR / "clusters.parquet")
    games = pd.read_parquet(C.PROCESSED_DIR / "games.parquet")
    labeled = clusters.merge(games[["game_id", "label", "is_cheat"]], on="game_id")
    main = labeled[labeled.split == "main"]

    shot = shot_level(main)
    game, game_scores = game_level(clusters, games)
    shot["prevalence"] = prevalence_table(shot["unsupervised"])
    game["prevalence"] = prevalence_table(game["unsupervised"])

    model = json.load(open(C.PROCESSED_DIR / "cluster_model.json", encoding="utf-8"))
    metrics = {"k": model["k"], "suspicious_clusters": model["suspicious_clusters"],
               "H1_shot_level": shot, "H2_game_level": game}
    with open(C.PROCESSED_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    game_scores.to_parquet(C.PROCESSED_DIR / "game_scores.parquet", index=False)
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = run()
    s, g = m["H1_shot_level"], m["H2_game_level"]
    u = s["unsupervised"]
    log.info("[H1 샷] F1 %.3f  P %.3f  R %.3f  FPR %.3f  (상한선 F1 %.3f)  ARI %.3f",
             u["f1"], u["precision"], u["recall"], u["fpr"],
             s["oracle_majority_mapping"]["f1"], s["ari_cheat"])
    log.info("       라벨별 의심 판정률 %s", s["detection_rate_by_label"])
    u = g["unsupervised"]
    log.info("[H2 경기] F1 %.3f  P %.3f  R %.3f  FPR %.3f  AUC %.3f  (임계값 %.3f)",
             u["f1"], u["precision"], u["recall"], u["fpr"], g["roc_auc"], g["threshold"])
    log.info("       라벨별 적발률 %s", g["detection_rate_by_label"])
    log.info("       봇별 AUC %s", {b: v["roc_auc"] for b, v in g["by_bot_type"].items()})
    log.info("       치터 비율별 precision %s", g["prevalence"])


if __name__ == "__main__":
    main()
