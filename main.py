import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------
# 한글 폰트 CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# CSV 읽기 (인코딩 자동 시도)
# ---------------------------------------------------------------
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ---------------------------------------------------------------
# 열 이름 표준화
# ---------------------------------------------------------------
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ---------------------------------------------------------------
# 학교명 추출 (파일명에서)
# ---------------------------------------------------------------
def get_school_name(filename):
    if not filename:
        return "우리 학교"
    base = filename.split(".")[0]
    parts = base.replace("-", "_").split("_")
    name = parts[0] if parts else base
    return name

# ---------------------------------------------------------------
# 환경 데이터 준비 (시간별 → 일별 집계)
# ---------------------------------------------------------------
def prep_daily(env_df):
    if "time" not in env_df.columns:
        return None
    df = env_df.dropna(subset=["time"]).copy()
    if df.empty:
        return None
    df["day"] = df["time"].dt.date
    cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in df.columns]
    if not cols:
        return None
    daily = df.groupby("day")[cols].mean().reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    return daily

# ---------------------------------------------------------------
# 일별 꺾은선 그래프
# ---------------------------------------------------------------
def make_daily_line(daily, col, title, color):
    if daily is None or col not in daily.columns:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["day"],
            y=daily[col],
            mode="lines",
            name=title,
            line=dict(color=color, width=2),
            hovertemplate="%{x|%m/%d}<br>%{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        hovermode="x unified",
        height=220,
        margin=dict(l=40, r=20, t=30, b=30),
    )
    fig.update_xaxes(tickformat="%m/%d", tickangle=0)
    return fig

# ---------------------------------------------------------------
# 단순 평균 막대 (요약용)
# ---------------------------------------------------------------
def make_bar(df, col, title, color):
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    avg = vals.mean()
    n = len(vals)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[1],
            y=[avg],
            name=title,
            marker_color=color,
            hovertemplate=f"평균 {title}<br>{avg:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title} (n={n})",
        xaxis=dict(showticklabels=False),
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=220,
        margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig

# ---------------------------------------------------------------
# 개체별 막대 그래프 (상위·하위 25% 강조)
# ---------------------------------------------------------------
def make_individual_bars(df, col, title, color):
    if col not in df.columns:
        return None
    tmp = df[["개체번호", col]].dropna().copy()
    if tmp.empty:
        return None
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna(subset=[col])
    if tmp.empty:
        return None
    tmp = tmp.sort_values(col, ascending=False).reset_index(drop=True)
    n = len(tmp)
    colors = [color] * n
    q75 = tmp[col].quantile(0.75) if n >= 4 else None
    q25 = tmp[col].quantile(0.25) if n >= 4 else None
    if q75 is not None:
        for i in range(n):
            if tmp.loc[i, col] >= q75:
                colors[i] = "#2e8b57"
            elif q25 is not None and tmp.loc[i, col] <= q25:
                colors[i] = "#cd5c5c"
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tmp["개체번호"].astype(str),
            y=tmp[col],
            name=title,
            marker_color=colors,
            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        )
    )
    if q75 is not None:
        fig.add_hline(y=q75, line_dash="dash", line_color="#2e8b57",
                      annotation_text="상위 25%", annotation_position="top right")
        fig.add_hline(y=q25, line_dash="dash", line_color="#cd5c5c",
                      annotation_text="하위 25%", annotation_position="bottom right")
    fig.update_layout(
        title=f"{title} (총 {n}개체)",
        xaxis_title="개체번호",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=280,
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis_tickangle=0,
    )
    return fig

# ---------------------------------------------------------------
# 바이올린 플롯 (생중량 분포)
# ---------------------------------------------------------------
def make_violin(df):
    if "생중량" not in df.columns:
        return None
    vals = pd.to_numeric(df["생중량"], errors="coerce").dropna()
    if vals.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=vals,
            name="생중량(g)",
            fillcolor="rgba(100, 180, 120, 0.5)",
            line_color="#2e8b57",
            hovertemplate="생중량: %{y:.2f} g<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"생중량 분포 (n={len(vals)}, 평균={vals.mean():.2f}g)",
        yaxis_title="생중량 (g)",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=260,
        margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig

# ---------------------------------------------------------------
# 산점도 + 추세선 + 상관계수
# ---------------------------------------------------------------
def make_scatter(df, x_col, y_col, x_label, y_label, color):
    if x_col not in df.columns or y_col not in df.columns:
        return None, None
    tmp = df[[x_col, y_col]].dropna().copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < 2:
        return None, None
    x = tmp[x_col].values
    y = tmp[y_col].values
    r = np.corrcoef(x, y)[0, 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"{x_label} vs {y_label}",
            marker=dict(color=color, size=9, opacity=0.75),
            hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
        )
    )
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"추세선 (r={r:.3f})",
            line=dict(color="#333", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"{x_label} vs {y_label} (r = {r:.3f})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=280,
        margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig, r

# ---------------------------------------------------------------
# 사이드바 · 페이지 설정
# ---------------------------------------------------------------
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
st.caption("CSV를 업로드하면 학교별 데이터로 바로 분석됩니다.")

env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("📊 생육 결과 데이터 CSV 업로드", type="csv")

with st.sidebar:
    st.divider()
    growth_cols_raw = [
        "잎 수", "지상부 길이", "지하부 길이", "생중량",
        "지상부 질량", "지하부 질량",
    ]
    selected_growth = st.multiselect(
        "생육 측정 항목 선택 (비교할 항목)",
        growth_cols_raw,
        default=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    )
    st.info("📎 CSV를 업로드하면 위 선택 항목으로 막대 그래프가 그려집니다.")

if env_file is not None:
    env_df = read_csv(env_file)
    env_df = std_cols(env_df)
    school_name = get_school_name(env_file.name)
else:
    env_df = pd.DataFrame()
    school_name = "우리 학교"

if growth_file is not None:
    growth_df = read_csv(growth_file)
    growth_df = std_cols(growth_df)
else:
    growth_df = pd.DataFrame()

if env_file is not None or growth_file is not None:
    st.sidebar.success(f"✅ {school_name} 데이터 로드 완료")
    if growth_file is not None:
        st.sidebar.caption(f"개체 수: {len(growth_df)}개")
    if env_file is not None:
        st.sidebar.caption(f"환경 데이터: {len(env_df)}행")

# ===============================================================
# 탭 1 — 📖 실험 개요
# ===============================================================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            """
**연구 배경**
극지식물은 낮은 온도, 강한 바람, 짧은 생장 기간 등 가혹한 환경에서 살아남기 위해
개체마다 다른 생존 전략을 보입니다. 같은 학교 실험실에서 키운 극지식물이라도
개체에 따라 잎 수, 줄기 길이, 무게 등이 크게 달라질 수 있습니다.

**연구 목적**
- 우리 학교 극지식물 개체별 생육 차이를 수치로 비교합니다.
- 평균·표준편차·변동계수(CV)로 개체 간 차이 정도를 파악합니다.
- 잎 수, 지상부 길이와 생중량 사이의 관계를 상관분석으로 살펴봅니다.
- 생중량이 특히 큰 개체(상위 25%)와 작은 개체(하위 25%)를 비교합니다.

> **용어 설명**
> - **평균**: 모든 값을 더해 개수로 나눈 대표값
> - **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 수
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 것 → 평균이 달라도 퍼짐 정도를 비교 가능
> - **상관계수(r)**: 두 변수가 함께 움직이는 정도 (-1 ~ +1), 0에 가까울수록 관계 약함
"""
        )

        st.divider()
        st.header("데이터 요약")

        n_growth = len(growth_df)
        n_env = len(env_df)

        rows = []
        if n_growth > 0:
            gcols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
            rows.append(["생육 결과", f"{n_growth}개체",
                         f"{len(gcols)}개 항목", ", ".join(gcols)])
        if n_env > 0:
            ecols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if "time" in env_df.columns:
                tmin = env_df["time"].min()
                tmax = env_df["time"].max()
                period = f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"
            else:
                period = "기간 정보 없음"
            rows.append(["환경 데이터", f"{n_env}행", period, ", ".join(ecols)])
        if not rows:
            st.info("📎 CSV를 업로드하면 데이터 요약이 표시됩니다.")
        else:
            summary_df = pd.DataFrame(
                rows,
                columns=["데이터 종류", "크기", "측정 기간", "측정 항목"]
            )
            st.dataframe(summary_df, width="stretch", use_container_width=False)

        st.divider()
        st.header("주요 지표")

        c1, c2, c3, c4 = st.columns(4)

        if n_growth > 0 and "생중량" in growth_df.columns:
            bw = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
            avg_bw = bw.mean() if not bw.empty else None
        else:
            avg_bw = None

        if n_env > 0 and "temperature" in env_df.columns:
            tv = pd.to_numeric(env_df["temperature"], errors="coerce").dropna()
            avg_temp = tv.mean() if not tv.empty else None
        else:
            avg_temp = None

        if n_env > 0 and "humidity" in env_df.columns:
            hv = pd.to_numeric(env_df["humidity"], errors="coerce").dropna()
            avg_hmd = hv.mean() if not hv.empty else None
        else:
            avg_hmd = None

        def metric_card(col, label, value, help_text=""):
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                col.metric(label, value, help_text)
            else:
                col.metric(label, "—")

        metric_card(c1, "총 개체수", n_growth if n_growth > 0 else "—",
                    "업로드된 생육 CSV 기준")
        metric_card(c2, "평균 온도 (°C)", f"{avg_temp:.1f}" if avg_temp is not None else "—")
        metric_card(c3, "평균 습도 (%)", f"{avg_hmd:.1f}" if avg_hmd is not None else "—")
        metric_card(c4, "평균 생중량 (g)", f"{avg_bw:.2f}" if avg_bw is not None else "—",
                    "생중량 열이 없을 경우 계산 불가")

        st.caption("※ 데이터가 업로드되지 않은 지표는 '—'으로 표시됩니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===============================================================
# 탭 2 — 🌡️ 환경 데이터
# ===============================================================
with tab2:
    try:
        if env_df.empty:
            st.info("📎 왼쪽 사이드바에서 환경 데이터 CSV를 업로드해 주세요.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터")

            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                env_num = env_df[env_cols].apply(pd.to_numeric, errors="coerce").dropna()
                if not env_num.empty:
                    summary = pd.DataFrame({
                        "항목": env_cols,
                        "평균": [round(env_num[c].mean(), 2) for c in env_cols],
                        "최소": [round(env_num[c].min(), 2) for c in env_cols],
                        "최대": [round(env_num[c].max(), 2) for c in env_cols],
                    })
                    st.subheader("환경 요약 (평균 · 최소 · 최대)")
                    st.dataframe(summary, width="stretch", use_container_width=False)
                else:
                    st.info("환경 데이터에서 숫자 열을 찾을 수 없습니다.")

            st.divider()

            daily = prep_daily(env_df)
            if daily is not None:
                ts_col1, ts_col2 = st.columns(2)
                fig_temp = make_daily_line(daily, "temperature",
                                           "일별 평균 온도 변화 (°C)", "#e74c3c")
                if fig_temp:
                    ts_col1.plotly_chart(fig_temp, width="stretch")
                else:
                    ts_col1.info("온도 데이터가 없습니다.")
                fig_hmd = make_daily_line(daily, "humidity",
                                           "일별 평균 습도 변화 (%)", "#3498db")
                if fig_hmd:
                    ts_col2.plotly_chart(fig_hmd, width="stretch")
                else:
                    ts_col2.info("습도 데이터가 없습니다.")

                ph_ec_avail = [c for c in ["ph", "ec"] if c in daily.columns]
                if ph_ec_avail:
                    ph_ec_col1, ph_ec_col2 = st.columns(2)
                    fig_ph = make_daily_line(daily, "ph", "일별 평균 pH 변화", "#9b59b6")
                    if fig_ph:
                        ph_ec_col1.plotly_chart(fig_ph, width="stretch")
                    else:
                        ph_ec_col1.info("pH 데이터가 없습니다.")
                    fig_ec = make_daily_line(daily, "ec", "일별 평균 EC 변화 (mS/cm)", "#16a085")
                    if fig_ec:
                        ph_ec_col2.plotly_chart(fig_ec, width="stretch")
                    else:
                        ph_ec_col2.info("EC 데이터가 없습니다.")
            else:
                st.info("환경 데이터의 시간 열 형식이 인식되지 않아 시계열 그래프를 그릴 수 없습니다.")

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch", use_container_width=False, height=300)
                csv_bytes_e = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 환경 데이터 CSV 다운로드",
                    data=csv_bytes_e,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===============================================================
# 탭 3 — 📊 생육 결과
# ===============================================================
with tab3:
    try:
        if growth_df.empty:
            st.info("📎 왼쪽 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.header(f"📊 {school_name} 생육 결과 분석")

            bw_col = "생중량" if "생중량" in growth_df.columns else None
            if bw_col:
                bw = pd.to_numeric(growth_df[bw_col], errors="coerce").dropna()
                if not bw.empty:
                    avg_bw_val = bw.mean()
                    max_idx = bw.idxmax()
                    min_idx = bw.idxmin()
                    max_val = bw.max()
                    min_val = bw.min()
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]

                    kg1, kg2, kg3 = st.columns(3)
                    kg1.metric("🥇 평균 생중량", f"{avg_bw_val:.2f} g",
                               delta=f"총 {len(bw)}개체 측정")
                    kg2.metric("🔝 최대 생중량 개체",
                               f"#{max_id} ({max_val:.2f} g)")
                    kg3.metric("🔻 최소 생중량 개체",
                               f"#{min_id} ({min_val:.2f} g)")

                    st.caption(
                        "※ 개체번호는 업로드한 CSV 그대로 표시됩니다 (숫자·글자 모두 가능).\n"
                        "※ 생중량 열이 없고 지상부·지하부 질량 열이 있으면 자동 합산됩니다."
                    )
                else:
                    st.info("생중량 데이터에서 유효한 숫자 값을 찾을 수 없습니다.")
                    avg_bw_val = None
            else:
                st.info("생중량 열이 없습니다. (지상부 질량 + 지하부 질량 열이 있으면 자동 계산됩니다.)")
                avg_bw_val = None

            st.divider()
            st.subheader("개체별 생육 비교")

            if bw_col:
                fig_bw = make_individual_bars(growth_df, "생중량", "생중량 (g)", "#27ae60")
                if fig_bw:
                    st.plotly_chart(fig_bw, width="stretch")
                else:
                    st.info("생중량 그래프로 표시할 데이터가 부족합니다.")
            else:
                st.info("생중량 열이 없어 해당 그래프를 그릴 수 없습니다.")

            avail_growth = [c for c in selected_growth if c in growth_df.columns]
            if avail_growth:
                n_graphs = len(avail_growth)
                cols_per_row = min(4, n_graphs)
                for i in range(0, n_graphs, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < n_graphs:
                            col_name = avail_growth[idx]
                            fig_g = make_bar(growth_df, col_name, col_name, "#8e44ad")
                            if fig_g:
                                cols[j].plotly_chart(fig_g, width="stretch")
                            else:
                                cols[j].info(f"{col_name}: 표시할 데이터가 부족합니다.")
            else:
                st.info("선택한 생육 항목 중 데이터에 존재하는 열이 없습니다. 사이드바에서 항목을 다시 선택해 주세요.")

            st.divider()
            st.subheader("생중량 분포")
            fig_violin = make_violin(growth_df)
            if fig_violin:
                st.plotly_chart(fig_violin, width="stretch")
            else:
                st.info("생중량 분포 그래프를 그릴 데이터가 부족합니다.")

            st.divider()
            st.subheader("상관관계 분석 (산점도 + 추세선)")
            sc1, sc2 = st.columns(2)

            fig_ls, r_ls = make_scatter(growth_df, "잎 수", "생중량",
                                         "잎 수 (장)", "생중량 (g)", "#2980b9")
            if fig_ls:
                sc1.plotly_chart(fig_ls, width="stretch")
                level = ("강한 양의 상관" if r_ls > 0.7
                         else "중간 양의 상관" if r_ls > 0.4
                         else "약한 관계" if abs(r_ls) < 0.4
                         else "음의 관계")
                sc1.caption(f"잎 수와 생중량의 상관계수 r = {r_ls:.3f}  ({level})")
            else:
                sc1.info("잎 수 또는 생중량 데이터가 충분하지 않아 그래프를 그릴 수 없습니다.")

            fig_ss, r_ss = make_scatter(growth_df, "지상부 길이", "생중량",
                                         "지상부 길이 (mm)", "생중량 (g)", "#d35400")
            if fig_ss:
                sc2.plotly_chart(fig_ss, width="stretch")
                level2 = ("강한 양의 상관" if r_ss > 0.7
                          else "중간 양의 상관" if r_ss > 0.4
                          else "약한 관계" if abs(r_ss) < 0.4
                          else "음의 관계")
                sc2.caption(f"지상부 길이와 생중량의 상관계수 r = {r_ss:.3f}  ({level2})")
            else:
                sc2.info("지상부 길이 또는 생중량 데이터가 충분하지 않아 그래프를 그릴 수 없습니다.")

            st.caption(
                "※ 상관계수(r)는 -1 ~ +1 사이 값이며, +1에 가까울수록 두 값이 함께 증가함을 의미합니다.\n"
                "※ 추세선은 np.polyfit으로 직접 계산한 직선입니다 (statsmodels 불필요)."
            )

            with st.expander("📋 생육 결과 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", use_container_width=False, height=300)
                csv_bytes_g = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 생육 결과 CSV 다운로드",
                    data=csv_bytes_g,
                    file_name=f"{school_name}_생육결과.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
