"""
원본 JSON -> 정제 테이블 파이프라인

    python -m src.preprocess                       # 기본 경로 사용
    python -m src.preprocess --raw_dir <경로> --out_dir <경로>

출력 (data/processed/)
    games.parquet           경기 1개 = 1행 (메타, 라벨, 샷 수, 명중률)
    shots.parquet           샷 1개 = 1행 (경기 키, 원시 궤적 50컬럼, 특징 16개)
    quality_report.json     파일·샷 단위 품질 점검 결과

라벨(label, is_cheat)은 games 테이블에만 둔다. 군집 단계는 shots만 읽고,
평가 단계에서 game_id로 조인한다 (학습 단계의 라벨 누수 방지).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_COLS, add_features
from .loader import SchemaError, extract_shots, game_meta, load_file, validate

log = logging.getLogger("preprocess")


def run(raw_dir: Path, out_dir: Path) -> dict:
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{raw_dir} 에 JSON 파일이 없습니다. README의 데이터 준비 단계를 확인하세요.")

    games, shot_rows, rejected, dropped_edge = [], [], [], 0
    for path in files:
        try:
            d = load_file(path)
            validate(d, path.stem)
            meta = game_meta(d, path)
        except (SchemaError, json.JSONDecodeError) as e:
            log.warning("제외: %s", e)
            rejected.append({"file": path.name, "reason": str(e)})
            continue
        rows, dropped = extract_shots(d, meta)
        dropped_edge += dropped
        meta["n_shots"] = len(rows)
        meta["hit_rate"] = float(np.mean([r["hit"] for r in rows])) if rows else np.nan
        games.append(meta)
        shot_rows.extend(rows)

    games_df = pd.DataFrame(games)
    shots_df = add_features(pd.DataFrame(shot_rows))

    # ---- 품질 점검 ----
    num = shots_df[C.YAW_COLS + C.PITCH_COLS + FEATURE_COLS].to_numpy(float)
    checks = {
        "files_total": len(files),
        "files_loaded": len(games_df),
        "files_rejected": rejected,
        "duplicate_game_id": int(games_df.game_id.duplicated().sum()),
        "duplicate_shot_key": int(shots_df.duplicated(["game_id", "frame"]).sum()),
        "non_finite_values": int((~np.isfinite(num)).sum()),
        "shots_total": len(shots_df),
        "shots_dropped_at_edge": dropped_edge,
        "games_with_few_shots(<30)": games_df.loc[games_df.n_shots < 30, "game_id"].tolist(),
        "hwid_count": int(games_df.hwid.nunique()),
        "games_by_label": games_df.label.value_counts().to_dict(),
        "main_set": {
            "games": int(games_df.randomized.sum()),
            "games_by_label": games_df[games_df.randomized].label.value_counts().to_dict(),
            "shots": int(shots_df.game_id.isin(games_df.game_id[games_df.randomized]).sum()),
        },
    }
    assert checks["duplicate_game_id"] == 0 and checks["duplicate_shot_key"] == 0
    assert checks["non_finite_values"] == 0

    out_dir.mkdir(parents=True, exist_ok=True)
    games_df.to_parquet(out_dir / "games.parquet", index=False)
    shots_df.to_parquet(out_dir / "shots.parquet", index=False)
    with open(out_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    return checks


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=C.RAW_DIR)
    ap.add_argument("--out_dir", type=Path, default=C.PROCESSED_DIR)
    args = ap.parse_args()

    q = run(args.raw_dir, args.out_dir)
    log.info("경기 %d/%d 로드, 샷 %d개 (메인 구간 %d경기 / %d샷)",
             q["files_loaded"], q["files_total"], q["shots_total"],
             q["main_set"]["games"], q["main_set"]["shots"])
    log.info("저장 위치: %s", args.out_dir)


if __name__ == "__main__":
    main()
