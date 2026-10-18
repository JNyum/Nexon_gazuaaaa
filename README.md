# Nexon_gazuaaaa
넥슨 슈터본부 슈터플랫폼실 데이터 엔지니어 과제 테스트

목표: 에임 이동 궤적만으로 특이 행동 집단을 찾아내고, 이를 운영 검토 대상으로 연결

가설: 
  - H1: 치팅 샷의 에임 궤적은 정상 샷과 다른 분포를 가지며, 라벨 없이도 별도 집단으로 묶인다.
    근거: 휴리스틱 에임봇은 수직축 이동의 연속성, 축 간 동시 이동 패턴에서 사람과 뚜렷한 차이를 보임 (Kanervisto et al., 2022)
    검증: 클러스터 순도, ARI/NMI, 실루엣 점수

  - H2: 샷 단위 군집 결과를 경기 단위로 누적하면 치팅 경기를 판별할 수 있다.
    근거: 단일 샷 판정은 불안정하지만, 게임 단위로 점수를 모으면 탐지율이 크게 향상됨 (Kanervisto et al., 2022)
    검증: 플레이어 단위 precision / recall / F1, 실제 치터 비율(1~5%) 조건에서의 precision

사용 데이터:
  Kanervisto, A., Hautamäki, V., & Kinnunen, T. (2022). Data for "GAN-Aimbots: The Next Generation of Cheating in Multiplayer Video Games". Zenodo. https://doi.org/10.5281/zenodo.6345323
  data_collection_3

활용 AI:
  Claude.ai - Pro plan, Co-work

reference: 
- 논문
  Kanervisto, A., Kinnunen, T., & Hautamäki, V. (2022). Gan-aimbots: Using machine learning for cheating in first person shooters. IEEE Transactions on Games, 15(4), 566-579.
- 논문 요약 
  - 배경
    기존 치팅 감지의 한계: 기존 안티치트 시스템은 플레이어의 마우스 이동 패턴이나 통계적 행동을 분석하여 에임봇 사용자를 감지
  - 목적
    생성형 대립 신경망(GAN) 등 최신 머신러닝을 활용하면 사람 플레이어와 구분할 수 없는 조준 행동(Human-like gameplay)을 생성하여 기존 데이터 기반 감지망을 우회할 수 있음을 검증하고 보안 위협을 공론화
  - 방법론
    GAN-Aimbot: 조건부 GAN (Conditional GAN): 실제 사람 플레이어의 마우스 이동 데이터셋을 기반으로 에임봇을 학습
    학습 구조: 생성자(Generator)는 목표 위치로 향하는 자연스러운 마우스 이동 경로를 생성하며, 판별자(Discriminator)는 이를 사람의 실제 마우스 궤적과 구분하도록 경쟁적으로 학습
    사람 행동 모방: 좌우/상하 이동 비율의 특성이나 목표 지점을 미세하게 넘어서는(Overshoot) 인간 특유의 조준 습관까지 학습하여 반영
  - 실험 결과
    조준 성능 입증: GAN-Aimbot을 사용했을 때 일반 플레이어의 샷 정확도(Accuracy)와 처치 수(Kills)가 통계적으로 유의미하게 향상
    자동 감지 시스템 우회: VACNet 방식의 심층 신경망(DNN) 감지 모델을 적용했을 때, 기존 룰(Heuristic) 기반 에임봇은 쉽게 적발된 반면 GAN-Aimbot은 높은 오탐/미탐 비율을 보이며 감지를 효과적으로 우회
    사람 평가자 우회: 숙련된 전문가와 FPS 게이머들이 플레이 영상을 직접 관전하며 심사했을 때도 GAN-Aimbot 사용자를 치터로 확정하기 매우 어려움
  - 결론 및 시사점
    머신러닝 기반 생성을 통한 치팅이 비디오 게임 산업에 실질적이고 심각한 위협이 됨
    짧은 구간의 마우스 움직임 분석만으로는 치트를 막기 어려움, 장기적 행동 모니터링이나 다각적 탐지 기법 등이 필요


## 프로젝트 구조

```
Nexon_gazuaaaa/
├── data/
│   ├── data_collection_3/       # 원본 JSON 132개
│   └── processed/               # 전처리·군집·평가 결과 (git 제외)
├── src/
│   ├── config.py                # 경로, 게임 사양, 라벨 매핑 등 상수
│   ├── loader.py                # JSON 로드, 스키마 검증, 사격 구간 추출
│   ├── features.py              # 궤적 -> 행동 특징 16개
│   ├── preprocess.py            # 파이프라인 실행 + 품질 리포트
│   ├── clustering.py            # K-Medoids 군집 (라벨 미사용) + 의심 집단 판정
│   └── evaluate.py              # 라벨 조인 후 H1(샷)/H2(경기) 평가
├── app.py                       # Streamlit 대시보드
└── requirements.txt
```

## 실행 방법

1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 데이터: `data/data_collection_3/`에 포함되어 있음 (출처: [Zenodo](https://doi.org/10.5281/zenodo.6345323))
3. 파이프라인 실행
   ```bash
   python -m src.preprocess   # JSON -> games/shots parquet
   python -m src.clustering   # K-Medoids 군집, 의심 집단 판정
   python -m src.evaluate     # 라벨 조인 후 평가 지표 산출
   streamlit run app.py       # 대시보드
   ```

## 전처리 결과

| 파일 | 단위 | 내용 |
|---|---|---|
| `games.parquet` | 경기 1개 = 1행 | game_id, hwid, episode, 무작위 구간 여부, label, is_cheat, FPS 숙련도, 샷 수, 명중률 |
| `shots.parquet` | 샷 1개 = 1행 | game_id, 사격 프레임, 무기, 명중 여부, 원시 궤적(yaw/pitch 각 25프레임), 행동 특징 16개 |
| `quality_report.json` | - | 스키마 오류 파일, 중복 키, 결측/무한값, 경계에서 버린 샷, 라벨 분포 |

- 사격 탐지: 탄약이 무기별 1발 소모량만큼 줄어든 프레임. 로켓런처·플라즈마건은 제외 (원본 코드와 동일)
- 궤적 창: 사격 전 0.5초(17프레임) + 사격 후 0.25초(8프레임)
- 라벨은 `games`에만 두고 `shots`에는 넣지 않는다. 군집은 `shots`만으로 학습하고, 평가할 때 `game_id`로 조인한다.
- 메인 분석 구간: episode2~5 (라벨이 무작위 순서로 배정, 라벨별 22경기씩 88경기 / 15,084샷). episode0(정상 10분), episode1(light 10분)은 순서가 고정돼 있어 보조 기준선으로만 사용한다.

## 군집과 평가

**군집 (`clustering.py`, 라벨 미사용)**
- 메인 구간 샷의 특징 16개를 RobustScaler로 맞춘 뒤 특징별 가중치를 곱한다. 가중치는 `config.py`의 `FEATURE_WEIGHTS`에 근거와 함께 사전 고정했다 (pitch 연속성·방향 전환 3.0, 멈춤 비율 2.0, yaw·속도·명중 0.5 등).
- 샘플 5,000개로 FasterPAM K-Medoids를 돌리고, K는 2~8 중 실루엣 점수로 선택한 뒤 전체 샷을 가장 가까운 medoid에 할당한다.
- **의심 집단 판정**: episode0(정상 10분 고정 판)을 "평소 행동" 기준선으로 두고, 학습에는 넣지 않고 할당만 한다. 메인 구간 비중 대비 기준선 비중이 0.5배 미만인 클러스터를 의심 집단으로 본다.

**평가 (`evaluate.py`, 라벨은 여기서 처음 조인)**
- H1 샷 단위: 의심 클러스터 소속을 치팅 예측으로 보고 precision/recall/F1, 봇 유형별 탐지율, ARI/NMI, 라벨 다수결 매핑 상한선을 계산한다.
- H2 경기 단위: 경기별 의심 샷 비율. 임계값은 기준선 경기 분포의 95% 분위수(라벨 미사용).
- 실제 치터 비율 시나리오: 같은 TPR/FPR에서 치터 비율 1~50%일 때 precision.

| 결과 파일 | 내용 |
|---|---|
| `clusters.parquet` | 샷별 클러스터, medoid까지 거리, 의심 여부 |
| `cluster_model.json` | K, 가중치, medoid 샷, 클러스터별 기준선 비중, K 탐색 결과 |
| `metrics.json` | H1/H2 지표, 봇 유형별 결과, ROC, 치터 비율별 precision |
| `game_scores.parquet` | 경기별 의심 샷 비율, 판정, 검토 우선순위 |

## 대시보드 (`app.py`)

| 탭 | 내용 |
|---|---|
| 개요 | H1/H2 핵심 지표, 라벨별 샷·경기 단위 의심 판정률 |
| 군집 구조 | K 선택(실루엣), 의심 집단 판정 근거(기준선 비중), 클러스터별 라벨 구성, medoid 대표 궤적, 라벨별 평균 pitch 이동량 |
| 탐지 성능 | 봇 유형별 경기 단위 ROC, 경기별 의심 샷 비율 분포와 임계값, 혼동행렬, 군집-라벨 일치도 |
| 실서비스 시나리오 | 치터 비율 슬라이더에 따른 precision, 1만 경기 기준 검토 대기열 규모와 검토 효율 |
| 검토 대기열 | 의심 샷 비율 순 경기 목록(정답 라벨은 토글로만 표시), 경기 타임라인, 의심 샷 2D 조준 경로 |
| 가중치 실험 | 특징 가중치·K를 바꿔 메모리에서 재군집하고 기본 설정과 지표 비교 (결과 파일은 덮어쓰지 않음) |
