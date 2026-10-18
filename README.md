# Nexon_gazuaaaa
넥슨 슈터본부 슈터플랫폼실 데이터 엔지니어 과제 테스트






reference: 
- paper
  Kanervisto, A., Kinnunen, T., & Hautamäki, V. (2022). Gan-aimbots: Using machine learning for cheating in first person shooters. IEEE Transactions on Games, 15(4), 566-579.
- data
  Kanervisto, A., Hautamäki, V., & Kinnunen, T. (2022). Data for "GAN-Aimbots: The Next Generation of Cheating in Multiplayer Video Games". Zenodo. https://doi.org/10.5281/zenodo.6345323
  data collection 3
- summary 
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

