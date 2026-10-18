"""
에임봇 탐지 PoC 대시보드

    streamlit run app.py

사전 조건: python -m src.preprocess && python -m src.clustering && python -m src.evaluate
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src import clustering as cl
from src import config as C
from src import evaluate as ev
from src.features import FEATURE_COLS

st.set_page_config(page_title="에임봇 탐지 PoC", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------------------
# 색상: 라벨(엔티티)별 고정. 검증된 categorical 팔레트 1~4번 슬롯 (adjacent CVD ΔE ≥ 9)
# 대비가 낮은 aqua/yellow가 있으므로 모든 막대에 값 라벨, 모든 차트에 표 보기를 함께 제공
# ---------------------------------------------------------------------------
LABELS = ["normal", "light", "strong", "gan"]
LABEL_KO = {"normal": "정상", "light": "light 봇", "strong": "strong 봇", "gan": "GAN 봇"}
LABEL_COLOR = {"normal": "#2a78d6", "light": "#eb6834", "strong": "#1baf7a", "gan": "#eda100"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"
MUTED = "#8a8985"
FRAME_AXIS = np.arange(-C.FRAMES_BEFORE, C.FRAMES_AFTER)   # 사격 프레임 = 0


def base_layout(fig, title=None, height=360, **kw):
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top", yref="container") if title else None,
                      height=height, margin=dict(l=10, r=10, t=90 if title else 40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal"),
                      hoverlabel=dict(namelength=-1), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
@st.cache_data
def load_all():
    P = C.PROCESSED_DIR
    shots = pd.read_parquet(P / "shots.parquet")
    games = pd.read_parquet(P / "games.parquet")
    clusters = pd.read_parquet(P / "clusters.parquet")
    scores = pd.read_parquet(P / "game_scores.parquet")
    model = json.load(open(P / "cluster_model.json", encoding="utf-8"))
    metrics = json.load(open(P / "metrics.json", encoding="utf-8"))
    return shots, games, clusters, scores, model, metrics


@st.cache_data
def load_cluster_inputs():
    return cl.load()


@st.cache_data(show_spinner="재군집 중...")
def recluster(weights_items, k, seed):
    model, clusters = cl.run(k=k, weights=dict(weights_items), seed=seed, save=False,
                             data=load_cluster_inputs())
    games = pd.read_parquet(C.PROCESSED_DIR / "games.parquet")
    metrics, scores = ev.compute(clusters, games, model)
    return model, metrics


try:
    shots, games, clusters, scores, model, metrics = load_all()
except FileNotFoundError:
    st.error("결과 파일이 없습니다. 먼저 `python -m src.preprocess && python -m src.clustering && "
             "python -m src.evaluate` 를 실행하세요.")
    st.stop()

H1, H2 = metrics["H1_shot_level"], metrics["H2_game_level"]
K = model["k"]
SUSP = set(model["suspicious_clusters"])
cluster_name = {c: f"C{c} ⚠ 의심" if c in SUSP else f"C{c}" for c in range(K)}

main_shots = (clusters[clusters.split == "main"]
              .merge(shots, on=["game_id", "frame", "shot_idx"])
              .merge(games[["game_id", "label", "is_cheat"]], on="game_id"))

# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("🎯 에임 궤적 기반 에임봇 탐지 PoC")
st.caption("GAN-Aimbots (Kanervisto et al., 2022) data collection 3 · 메인 구간 88경기 / "
           f"{H1['n_shots']:,}샷 · K-Medoids (K={K}) · 라벨은 평가 단계에서만 사용")

tabs = st.tabs(["개요", "군집 구조", "탐지 성능", "실서비스 시나리오", "검토 대기열", "가중치 실험"])

# ===========================================================================
# 1. 개요
# ===========================================================================
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("H1 · 샷 단위")
        st.markdown("치팅 샷의 에임 궤적은 정상 샷과 다른 분포를 가지며, 라벨 없이도 별도 집단으로 묶인다.")
        u = H1["unsupervised"]
        m = st.columns(4)
        m[0].metric("F1", f"{u['f1']:.3f}")
        m[1].metric("Precision", f"{u['precision']:.3f}")
        m[2].metric("Recall", f"{u['recall']:.3f}")
        m[3].metric("정상 오탐률", f"{u['fpr']:.1%}")
    with c2:
        st.subheader("H2 · 경기 단위")
        st.markdown("샷 단위 군집 결과를 경기 단위로 누적하면 치팅 경기를 판별할 수 있다.")
        u = H2["unsupervised"]
        m = st.columns(4)
        m[0].metric("F1", f"{u['f1']:.3f}")
        m[1].metric("ROC-AUC", f"{H2['roc_auc']:.3f}")
        m[2].metric("Recall", f"{u['recall']:.3f}")
        m[3].metric("정상 오탐률", f"{u['fpr']:.1%}")

    st.divider()
    det = pd.DataFrame({
        "샷 단위": pd.Series(H1["detection_rate_by_label"]),
        "경기 단위": pd.Series(H2["detection_rate_by_label"]),
    }).reindex(LABELS)
    fig = go.Figure()
    for name, color in [("샷 단위", SERIES[6]), ("경기 단위", SERIES[0])]:
        fig.add_bar(x=[LABEL_KO[l] for l in det.index], y=det[name], name=name, marker_color=color,
                    text=[f"{v:.0%}" for v in det[name]], textposition="outside",
                    hovertemplate="%{x}<br>" + name + " 의심 판정률 %{y:.1%}<extra></extra>")
    fig.update_traces(marker_line_width=0)
    base_layout(fig, "라벨별 의심 판정률 · 샷을 경기 단위로 모으면 GAN 봇도 드러나기 시작", height=380,
                barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], title=None)
    show(fig)
    st.markdown(
        f"- 휴리스틱 봇(light·strong)은 샷 단위에서도 분리되고, 경기 단위로는 거의 전부 적발됩니다.\n"
        f"- GAN 봇은 샷 단위 {H1['detection_rate_by_label']['gan']:.0%}로 거의 숨지만, 경기 단위 의심 샷 비율이 "
        f"정상의 약 {H2['ratio_by_label']['gan'] / max(H2['ratio_by_label']['normal'], 1e-9):.0f}배라 "
        f"AUC {H2['by_bot_type']['gan']['roc_auc']:.2f}까지 올라옵니다.\n"
        f"- 판정 임계값은 정상 기준선 경기(episode0)로만 정했습니다: 의심 샷 비율 > {H2['threshold']:.1%}")

# ===========================================================================
# 2. 군집 구조
# ===========================================================================
with tabs[1]:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        sw = pd.DataFrame(model["k_sweep"])
        fig = go.Figure()
        fig.add_scatter(x=sw.k, y=sw.silhouette, mode="lines+markers", name="실루엣",
                        line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                        hovertemplate="K=%{x}<br>실루엣 %{y:.3f}<extra></extra>")
        best = sw[sw.k == K]
        fig.add_scatter(x=best.k, y=best.silhouette, mode="markers+text", text=[f"선택 K={K}"],
                        textposition="top right", marker=dict(size=14, color=SERIES[0],
                        line=dict(width=2, color="white")), showlegend=False, hoverinfo="skip")
        base_layout(fig, "K 선택 (라벨 없이 실루엣 점수)", showlegend=False)
        fig.update_xaxes(title="K", dtick=1)
        fig.update_yaxes(title=None, range=[0, sw.silhouette.max() * 1.2])
        show(fig)
    with c2:
        info = pd.DataFrame(model["clusters"])
        fig = go.Figure()
        for c in range(K):
            r = info[info.cluster == c].iloc[0]
            fig.add_bar(y=[cluster_name[c]], x=[r.main_share], name="메인 구간", orientation="h",
                        marker_color=SERIES[0], text=[f"{r.main_share:.0%}"], textposition="outside",
                        legendgroup="m", showlegend=c == 0,
                        hovertemplate="%{y}<br>메인 구간 샷 비중 %{x:.1%}<extra></extra>")
            fig.add_bar(y=[cluster_name[c]], x=[r.baseline_share], name="정상 기준선 (episode0)",
                        orientation="h", marker_color=SERIES[1], text=[f"{r.baseline_share:.0%}"],
                        textposition="outside", legendgroup="b", showlegend=c == 0,
                        hovertemplate="%{y}<br>기준선 샷 비중 %{x:.1%}<extra></extra>")
        base_layout(fig, "의심 집단 판정 · 기준선 샷이 거의 오지 않는 클러스터", barmode="group",
                    bargap=0.3, bargroupgap=0.08)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.15])
        fig.update_yaxes(autorange="reversed")
        show(fig)

    st.markdown("**클러스터별 실제 라벨 구성** · 군집은 라벨을 보지 않았고, 이 그림에서 처음 대조합니다.")
    comp = (main_shots.groupby(["cluster", "label"]).size().unstack(fill_value=0)
            .reindex(columns=LABELS, fill_value=0))
    share = comp.div(comp.sum(axis=1), axis=0)
    fig = go.Figure()
    for l in LABELS:
        fig.add_bar(y=[cluster_name[c] for c in share.index], x=share[l], name=LABEL_KO[l],
                    orientation="h", marker_color=LABEL_COLOR[l],
                    marker_line=dict(width=2, color="rgba(255,255,255,0.9)"),
                    text=[f"{v:.0%}" if v >= 0.04 else "" for v in share[l]],
                    textposition="inside", insidetextanchor="middle",
                    customdata=comp[l], hovertemplate="%{y} · " + LABEL_KO[l] +
                    "<br>%{x:.1%} (%{customdata:,}샷)<extra></extra>")
    base_layout(fig, None, height=80 + 70 * K, barmode="stack")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    show(fig)

    st.markdown("**대표 궤적 (medoid)** · 각 클러스터의 중심은 실제 샷 하나입니다. 0 = 사격 프레임")
    cols = st.columns(2)
    for ax, (col, name) in enumerate([(C.YAW_COLS, "yaw (좌우)"), (C.PITCH_COLS, "pitch (상하)")]):
        fig = go.Figure()
        for c in range(K):
            r = info[info.cluster == c].iloc[0]
            row = shots[(shots.game_id == r.medoid_game_id) & (shots.frame == r.medoid_frame)].iloc[0]
            fig.add_scatter(x=FRAME_AXIS, y=row[col].to_numpy(float), mode="lines+markers",
                            name=cluster_name[c], line=dict(color=SERIES[c], width=2),
                            marker=dict(size=6),
                            hovertemplate=cluster_name[c] + "<br>프레임 %{x}<br>회전량 %{y:.2f}°<extra></extra>")
        fig.add_vline(x=0, line_dash="dot", line_color=MUTED, annotation_text="사격",
                      annotation_position="top")
        base_layout(fig, name, height=320)
        fig.update_xaxes(title="프레임")
        cols[ax].plotly_chart(fig, width="stretch", theme="streamlit")

    st.markdown("**라벨별 평균 상하 이동량** · 봇은 사격 전후로 pitch를 계속 미세 보정합니다.")
    prof = (main_shots.groupby("label")[C.PITCH_COLS].apply(lambda d: d.abs().mean())
            .reindex(LABELS))
    fig = go.Figure()
    for l in LABELS:
        fig.add_scatter(x=FRAME_AXIS, y=prof.loc[l].to_numpy(), mode="lines", name=LABEL_KO[l],
                        line=dict(color=LABEL_COLOR[l], width=2),
                        hovertemplate=LABEL_KO[l] + "<br>프레임 %{x}<br>평균 |pitch| %{y:.3f}°<extra></extra>")
        fig.add_annotation(x=FRAME_AXIS[-1], y=prof.loc[l].iloc[-1], text=LABEL_KO[l], showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=12))
    fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
    base_layout(fig, None, height=320)
    fig.update_xaxes(title="프레임", range=[FRAME_AXIS[0], FRAME_AXIS[-1] + 3])
    fig.update_yaxes(title="평균 |pitch| (°)")
    show(fig)

    with st.expander("표로 보기 · medoid 특징값과 클러스터 구성"):
        mf = pd.DataFrame({cluster_name[r["cluster"]]: r["medoid_features"] for r in model["clusters"]})
        mf["가중치"] = pd.Series(model["weights"])
        st.dataframe(mf.round(3), width="stretch")
        st.dataframe(comp.rename(index=cluster_name, columns=LABEL_KO), width="stretch")

# ===========================================================================
# 3. 탐지 성능
# ===========================================================================
with tabs[2]:
    gm = scores[scores.split == "main"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip")
        for b in ["light", "strong", "gan"]:
            sub = gm[gm.label.isin(["normal", b])]
            fpr, tpr, _ = roc_curve(sub.is_cheat, sub.suspicious_ratio)
            auc = H2["by_bot_type"][b]["roc_auc"]
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{LABEL_KO[b]} (AUC {auc:.2f})",
                            line=dict(color=LABEL_COLOR[b], width=2, shape="hv"),
                            hovertemplate=LABEL_KO[b] + "<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")
        base_layout(fig, "경기 단위 ROC (봇 유형 vs 정상)", height=400)
        fig.update_xaxes(title="정상 경기 오탐률 (FPR)", range=[-0.02, 1.02])
        fig.update_yaxes(title="치팅 경기 적발률 (TPR)", range=[-0.02, 1.05])
        show(fig)
    with c2:
        fig = go.Figure()
        rng = np.random.default_rng(0)
        for i, l in enumerate(LABELS):
            sub = gm[gm.label == l]
            fig.add_scatter(x=sub.suspicious_ratio, y=i + rng.uniform(-0.18, 0.18, len(sub)),
                            mode="markers", name=LABEL_KO[l],
                            marker=dict(size=9, color=LABEL_COLOR[l], line=dict(width=1.5, color="white")),
                            customdata=sub[["game_id", "n_shots"]],
                            hovertemplate="%{customdata[0]}<br>의심 샷 비율 %{x:.1%}"
                                          "<br>샷 %{customdata[1]}개<extra>" + LABEL_KO[l] + "</extra>")
        fig.add_vline(x=H2["threshold"], line_dash="dash", line_color=CRITICAL,
                      annotation_text=f"임계값 {H2['threshold']:.1%}", annotation_position="top right")
        base_layout(fig, "경기별 의심 샷 비율 분포", height=400, showlegend=False)
        fig.update_yaxes(tickvals=list(range(4)), ticktext=[LABEL_KO[l] for l in LABELS],
                         autorange="reversed", showgrid=False)
        fig.update_xaxes(tickformat=".0%", title="경기 내 의심 클러스터 샷 비율")
        show(fig)

    st.markdown("**봇 유형별 성능** (해당 봇 경기 vs 정상 경기)")
    rows = []
    for lvl, M in [("샷", H1), ("경기", H2)]:
        for b in ["light", "strong", "gan"]:
            r = M["by_bot_type"][b]
            rows.append({"단위": lvl, "봇": LABEL_KO[b], "Precision": r["precision"], "Recall": r["recall"],
                         "F1": r["f1"], "정상 오탐률": r["fpr"], "AUC": r.get("roc_auc", np.nan)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    for col, title, cm in [(c1, "샷 단위 혼동행렬", H1["unsupervised"]["confusion"]),
                           (c2, "경기 단위 혼동행렬", H2["unsupervised"]["confusion"])]:
        col.markdown(f"**{title}**")
        col.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                                   index=["실제 정상", "실제 치팅"], columns=["예측 정상", "예측 의심"]),
                      width="stretch")
    c3.markdown("**군집-라벨 일치도 (샷)**")
    c3.dataframe(pd.DataFrame({
        "값": [H1["ari_label4"], H1["nmi_label4"], H1["oracle_majority_mapping"]["f1"]]},
        index=["ARI (4라벨)", "NMI (4라벨)", "상한선 F1 (다수결 매핑)"]), width="stretch")

# ===========================================================================
# 4. 실서비스 시나리오
# ===========================================================================
with tabs[3]:
    st.markdown("데이터의 치팅 경기 비율은 75%지만 실서비스는 훨씬 낮습니다. 같은 적발률·오탐률에서 "
                "**치터 비율에 따라 의심 판정의 신뢰도(precision)가 어떻게 바뀌는지** 봅니다.")
    rate = st.slider("실서비스 치팅 경기 비율", 0.005, 0.5, 0.05, 0.005, format="%.3f")
    ps = np.linspace(0.005, 0.5, 100)
    fig = go.Figure()
    for name, M, color in [("샷 단위", H1, SERIES[6]), ("경기 단위", H2, SERIES[0])]:
        tpr, fpr = M["unsupervised"]["recall"], M["unsupervised"]["fpr"]
        prec = [ev.precision_at(tpr, fpr, p) for p in ps]
        fig.add_scatter(x=ps, y=prec, mode="lines", name=name, line=dict(color=color, width=2),
                        hovertemplate=name + "<br>치터 비율 %{x:.1%}<br>precision %{y:.1%}<extra></extra>")
    fig.add_vline(x=rate, line_dash="dot", line_color=MUTED)
    base_layout(fig, None, height=360)
    fig.update_xaxes(tickformat=".0%", title="치팅 경기 비율", range=[0, 0.51])
    fig.update_yaxes(tickformat=".0%", title="의심 판정 중 실제 치터 비율", range=[0, 1.02])
    show(fig)

    tpr, fpr = H2["unsupervised"]["recall"], H2["unsupervised"]["fpr"]
    n = 10_000
    tp, fp = tpr * rate * n, fpr * (1 - rate) * n
    m = st.columns(4)
    m[0].metric("1만 경기 중 치팅 경기", f"{rate * n:,.0f}")
    m[1].metric("검토 대기열에 오르는 경기", f"{tp + fp:,.0f}")
    m[2].metric("그중 실제 치팅", f"{tp:,.0f}")
    m[3].metric("정상인데 오른 경기", f"{fp:,.0f}")
    st.info(f"치터 비율 {rate:.1%}에서 대기열 경기의 {tp / (tp + fp):.0%}가 실제 치팅입니다. "
            f"무작위로 검토하면 {rate:.1%}만 치팅이므로 검토 효율이 약 {tp / (tp + fp) / rate:.0f}배입니다. "
            "자동 제재보다 **수동 검토 우선순위**로 쓰는 게 적합한 이유입니다.")

# ===========================================================================
# 5. 검토 대기열
# ===========================================================================
with tabs[4]:
    st.markdown("의심 샷 비율이 높은 순으로 정렬한 경기 목록입니다. 운영자가 위에서부터 검토합니다.")
    reveal = st.toggle("정답 라벨 보기 (평가용)", value=False)
    q = gm.sort_values("suspicious_ratio", ascending=False).reset_index(drop=True)
    q.insert(0, "순위", np.arange(1, len(q) + 1))
    q["판정"] = np.where(q.flagged == 1, "⚠ 검토 대상", "")
    view = q[["순위", "game_id", "hwid", "n_shots", "suspicious_ratio", "판정"] +
             (["label"] if reveal else [])].rename(columns={"n_shots": "샷 수", "suspicious_ratio": "의심 샷 비율"})
    if reveal:
        view["label"] = view["label"].map(LABEL_KO)
    st.dataframe(view, width="stretch", hide_index=True, height=320,
                 column_config={"의심 샷 비율": st.column_config.ProgressColumn(
                     format="percent", min_value=0, max_value=1)})

    gid = st.selectbox("경기 상세", q.game_id, format_func=lambda g: f"#{int(q[q.game_id == g].순위.iloc[0])}  {g}")
    gs = main_shots[main_shots.game_id == gid].sort_values("time_sec")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = go.Figure()
        for flag, name, color, size in [(0, "일반 샷", MUTED, 6), (1, "⚠ 의심 샷", CRITICAL, 9)]:
            s = gs[gs.suspicious == flag]
            fig.add_scatter(x=s.time_sec, y=s.pitch_dir_changes, mode="markers", name=name,
                            marker=dict(size=size, color=color, line=dict(width=1, color="white")),
                            hovertemplate="%{x:.1f}초<br>pitch 방향 전환 %{y:.0f}회<extra>" + name + "</extra>")
        base_layout(fig, "경기 타임라인 · 샷별 pitch 방향 전환", height=340)
        fig.update_xaxes(title="경기 시간 (초)")
        fig.update_yaxes(title="pitch 방향 전환 횟수")
        show(fig)
    with c2:
        susp = gs[gs.suspicious == 1]
        pick = susp if len(susp) else gs
        fig = go.Figure()
        for _, r in pick.head(5).iterrows():
            fig.add_scatter(x=np.cumsum(r[C.YAW_COLS].to_numpy(float)),
                            y=np.cumsum(r[C.PITCH_COLS].to_numpy(float)), mode="lines",
                            line=dict(width=2, color=CRITICAL if r.suspicious else MUTED),
                            opacity=0.8, showlegend=False,
                            hovertemplate=f"{r.time_sec:.1f}초 샷<br>누적 yaw %{{x:.1f}}° · pitch %{{y:.1f}}°<extra></extra>")
        base_layout(fig, f"{'의심 ' if len(susp) else ''}샷 2D 조준 경로 (최대 5개)", height=340)
        fig.update_xaxes(title="누적 yaw (°)", zeroline=True)
        fig.update_yaxes(title="누적 pitch (°)", zeroline=True)
        show(fig)

# ===========================================================================
# 6. 가중치 실험
# ===========================================================================
with tabs[5]:
    st.markdown("특징 가중치와 K를 바꿔 다시 군집합니다. 기본값은 `config.py`에 근거와 함께 고정된 값이며, "
                "여기서의 조정은 **민감도 확인용**입니다. 라벨을 보며 맞추면 비지도 설정이 깨진다는 점에 유의하세요.")
    with st.form("weights"):
        cols = st.columns(4)
        w = {}
        for i, f in enumerate(FEATURE_COLS):
            w[f] = cols[i % 4].slider(f, 0.0, 5.0, float(C.FEATURE_WEIGHTS[f]), 0.5)
        c1, c2, _ = st.columns([1, 1, 2])
        k_new = c1.selectbox("K", list(C.K_RANGE), index=list(C.K_RANGE).index(K))
        seed = c2.number_input("seed", value=C.SEED, step=1)
        go_btn = st.form_submit_button("다시 군집", type="primary")
    if go_btn:
        m2, met2 = recluster(tuple(sorted(w.items())), int(k_new), int(seed))
        st.caption(f"K={m2['k']} · 의심 클러스터 {m2['suspicious_clusters']}")
        rows = []
        for lvl, a, b in [("샷", H1, met2["H1_shot_level"]), ("경기", H2, met2["H2_game_level"])]:
            for key in ["precision", "recall", "f1", "fpr"]:
                rows.append({"단위": lvl, "지표": key, "기본": a["unsupervised"][key],
                             "실험": b["unsupervised"][key],
                             "차이": round(b["unsupervised"][key] - a["unsupervised"][key], 4)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        d1 = pd.Series(H2["detection_rate_by_label"]).reindex(LABELS)
        d2 = pd.Series(met2["H2_game_level"]["detection_rate_by_label"]).reindex(LABELS)
        fig = go.Figure()
        for name, d, color in [("기본", d1, SERIES[0]), ("실험", d2, SERIES[1])]:
            fig.add_bar(x=[LABEL_KO[l] for l in LABELS], y=d, name=name, marker_color=color,
                        text=[f"{v:.0%}" for v in d], textposition="outside",
                        hovertemplate="%{x}<br>" + name + " 경기 적발률 %{y:.1%}<extra></extra>")
        base_layout(fig, "경기 단위 라벨별 적발률 비교", barmode="group", bargap=0.35, bargroupgap=0.08)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
        show(fig)
