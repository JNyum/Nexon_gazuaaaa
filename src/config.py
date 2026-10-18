"""경로와 상수를 한 곳에서 관리한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "data_collection_3"
PROCESSED_DIR = ROOT / "data" / "processed"

# ---- 게임/녹화 사양 (원본 feature_extraction.py와 동일) ----
FPS = 35
FRAMES_BEFORE = int(0.5 * FPS)    # 사격 전 0.5초 = 17프레임
FRAMES_AFTER = int(0.25 * FPS)    # 사격 후 0.25초 = 8프레임
WINDOW = FRAMES_BEFORE + FRAMES_AFTER

YAW_IDX, PITCH_IDX = 4, 5         # actions 안의 좌우/상하 회전량 위치

# 무기 번호 -> 1발당 소모 탄약 (슈퍼샷건은 2발씩 소모)
AMMO_PER_SHOT = {1: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1}
SKIP_WEAPONS = {5, 6}             # 로켓런처, 플라즈마건 제외

# ---- 실험 설계 ----
# episode0 = 정상 10분, episode1 = light 10분 (순서 고정)
# episode2~5 = 2.5분, 네 라벨이 무작위 순서로 배정 -> 메인 분석 구간
RANDOMIZED_EPISODES = {"episode2", "episode3", "episode4", "episode5"}

LABEL_MAP = {
    "none": "normal", "null": "normal",
    "ease_light": "light", "ease_strong": "strong",
    "gan_group0": "gan", "gan_group1": "gan",
}

# 원본 JSON 필수 키 (README의 enemy_velocities는 실제 파일에서 enemy_vels)
REQUIRED_KEYS = ["hwid", "aimbot", "actions", "damages", "ammos", "weapons", "frags", "deaths"]
PER_FRAME_KEYS = ["actions", "damages", "ammos", "weapons", "frags", "deaths"]

YAW_COLS = [f"yaw_{i:02d}" for i in range(WINDOW)]
PITCH_COLS = [f"pitch_{i:02d}" for i in range(WINDOW)]

# ---- 군집 (clustering.py) ----
BASELINE_EPISODE = "episode0"   # 정상 10분 고정 판 -> "평소 행동" 기준선 (군집 학습에는 미사용)

# 특징 가중치: 라벨을 보고 튜닝하지 않고, 논문 근거와 도메인 가정으로 사전에 고정한다.
#  - pitch 계열: 논문 Table VII에서 사람/휴리스틱 봇 차이가 가장 큰 축 -> 3.0
#  - 멈춤 비율, 방향 전환: 봇은 쉬지 않고 미세 보정한다는 가정 -> 2.0
#  - yaw 계열, 속도: 사람 간 개인차가 큰 축 -> 0.5~1.0
#  - hit: 궤적이 아닌 결과값이고 실력과 섞임 -> 0.5
FEATURE_WEIGHTS = {
    "yaw_speed_mean": 0.5, "pitch_speed_mean": 2.0, "speed_max": 0.5,
    "pre_speed_mean": 0.5, "post_speed_mean": 0.5, "pre_shot_decel": 1.0,
    "jerk_mean": 1.0, "snap_ratio": 1.0,
    "yaw_step_corr": 1.0, "pitch_step_corr": 3.0, "yaw_pitch_corr": 1.0,
    "straightness": 1.0, "yaw_dir_changes": 1.0, "pitch_dir_changes": 3.0,
    "idle_ratio": 2.0, "hit": 0.5,
}

K_RANGE = range(2, 9)
FIT_SAMPLE = 5000               # medoid 탐색용 샘플 수 (거리행렬 5000^2)
SUSPICIOUS_RATIO = 0.5          # 기준선 비중 / 메인 비중 < 0.5 인 클러스터 = 의심 집단
SEED = 42
