import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 한글 폰트 & Streamlit 스타일
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ─────────────────────────────────────────────
# CSV 읽기 (인코딩 대응)
# ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 시도, 실패 시 빈 DataFrame 반환"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 열 이름 표준화 (KeyError 방지)
# ─────────────────────────────────────────────
@st.cache_data
def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 통일"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        # 괄호 안 한글 제거: 'time(시간)' → 'time'
        base = re.sub(r"\(.+?\)", "", c.lower().replace(" ", ""))
        k = base
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
    # 숫자 변환 가능한 열은 숫자로
    for col in df.columns:
        if col in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            num = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").
                str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[col] = num
    # 생중량 열이 없으면 지상부+지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    # 시간 파싱 (날짜_시각 형식도 지원)
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(
                parts[0], format="%Y%m%d"
            ) + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


# ─────────────────────────────────────────────
# 사이드바: 파일 업로드 & 학교명 추출
# ─────────────────────────────────────────────
st.sidebar.header("📎 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

school_name = "우리 학교"
if env_file is not None:
    name = env_file.name.split("_")[0] if "_" in env_file.name else env_file.name.split(".")[0]
    school_name = name

if growth_file is not None:
    name = growth_file.name.split("_")[0] if "_" in growth_file.name else growth_file.name.split(".")[0]
    if school_name == "우리 학교":
        school_name = name

st.sidebar.markdown(f"🏫 현재 학교: **{school_name}**")

growth_options = []
if growth_file is not None:
    growth_df_raw = read_csv(growth_file)
    growth_df_std = std_cols(growth_df_raw)
    # 분석 대상 열 목록 (개체번호 제외)
    possible = ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                "지상부 질량", "지하부 질량"]
    growth_options = [c for c in possible if c in growth_df_std.columns]

selected_items = st.sidebar.multiselect(
    "생육 측정 항목 선택",
    growth_options,
    default=growth_options[:3] if len(growth_options) >= 3 else growth_options,
)

# ─────────────────────────────────────────────
# 데이터 로딩 (상단에서 한 번만)
# ─────────────────────────────────────────────
with st.spinner("데이터를 불러오고 있습니다…"):
    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    if env_file is not None:
        env_df = std_cols(read_csv(env_file))
    if growth_file is not None:
        growth_df = std_cols(read_csv(growth_file))

# ─────────────────────────────────────────────
# 공통 헬퍼: 열 존재 확인
# ─────────────────────────────────────────────
def has_col(df, col):
    return col in df.columns and df[col].notna().sum() > 0


# ─────────────────────────────────────────────
# 공통 헬퍼: CSV 다운로드 버튼
# ─────────────────────────────────────────────
def download_csv(df, label):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv_bytes, f"{label}.csv",
                       mime="text/csv", use_container_width=True)


# ─────────────────────────────────────────────
# 공통 헬퍼: Plotly 한글 폰트 적용
# ─────────────────────────────────────────────
def set_kr_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )


# ─────────────────────────────────────────────
# 공통 헬퍼: 추세선 추가 (np.polyfit)
# ─────────────────────────────────────────────
def add_trendline(fig, x_vals, y_vals, color="red"):
    clean = pd.concat([x_vals, y_vals], axis=1).dropna()
    if len(clean) < 2:
        return
    x_clean = clean.iloc[:, 0].values
    y_clean = clean.iloc[:, 1].values
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    fig.add_trace(
        go.Scatter(
            x=x_clean,
            y=slope * x_clean + intercept,
            mode="lines",
            line=dict(color=color, dash="dash", width=2),
            name=f"추세선 (r={np.corrcoef(x_clean, y_clean)[0,1]:.2f})",
            showlegend=True,
        )
    )


# ─────────────────────────────────────────────
# 공통 헬퍼: 지표 카드 (st.metric 사용)
# ─────────────────────────────────────────────
def metric_card(label, value, delta=None, delta_desc=None):
    st.metric(label, value, delta=delta, delta_description=delta_desc)


# ─────────────────────────────────────────────
# Tab 1: 실험 개요
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            f"""
**{school_name}**에서 재배 중인 극지식물의 **개체별 생육 차이**를 분석합니다.

- 같은 환경에서 자라더라도 개체마다 잎 수, 줄기 길이, 생체중이 다를 수 있습니다.
- 이 대시보드는 각 개체의 생육 데이터를 비교하고, 어떤 개체가 잘 자랐는지,
  잎 수와 줄기 길이가 생중량과 어떤 관계가 있는지 살펴봅니다.

📌 **주요 통계 용어**
- **평균**: 모든 값을 더해 개수로 나눈 대표값
- **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 수치 (클수록 들쭉날쭉)
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값, 단위가 다른 항목 간 비교에 사용
- **상관계수(r)**: 두 변수가 함께 움직이는 정도 (−1~1, 1에 가까울수록 강한 양의 관계)
            """
        )

        # 데이터 요약 표
        st.subheader("📋 데이터 요약")
        summary_rows = []
        if len(growth_df) > 0:
            n_ind = growth_df["개체번호"].nunique() if "개체번호" in growth_df.columns else len(growth_df)
            summary_rows.append({"항목": "총 개체 수", "값": f"{n_ind}개체"})
        if has_col(env_df, "time"):
            t_min = env_df["time"].min()
            t_max = env_df["time"].max()
            summary_rows.append({"항목": "측정 기간", "값": f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}"})
        if has_col(env_df, "temperature"):
            summary_rows.append({"항목": "측정 항목 (환경)", "값": "온도, 습도, pH, EC"})
        growth_items = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                     "지상부 질량", "지하부 질량"]
                        if c in growth_df.columns]
        if growth_items:
            summary_rows.append({"항목": "측정 항목 (생육)", "값": ", ".join(growth_items)})
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
        else:
            st.info("데이터가 아직 업로드되지 않았습니다. 사이드바에서 CSV 파일을 올려주세요.")

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if has_col(growth_df, "개체번호"):
                n = growth_df["개체번호"].nunique() if "개체번호" in growth_df.columns else len(growth_df)
                metric_card("총 개체 수", f"{n}개체")
            else:
                st.info("개체번호 열을 찾을 수 없어요.")

        with col2:
            if has_col(env_df, "temperature"):
                t = env_df["temperature"].dropna()
                if len(t) >= 2:
                    metric_card("평균 온도", f"{t.mean():.1f}°C",
                                delta=f"범위 {t.min():.1f} ~ {t.max():.1f}°C")
                else:
                    st.info("온도 데이터가 부족해요.")
            else:
                st.info("온도 열이 없는 환경 CSV예요.")

        with col3:
            if has_col(env_df, "humidity"):
                h = env_df["humidity"].dropna()
                if len(h) >= 2:
                    metric_card("평균 습도", f"{h.mean():.1f}%",
                                delta=f"범위 {h.min():.1f} ~ {h.max():.1f}%")
                else:
                    st.info("습도 데이터가 부족해요.")
            else:
                st.info("습도 열이 없는 환경 CSV예요.")

        with col4:
            if has_col(growth_df, "생중량"):
                g = growth_df["생중량"].dropna()
                if len(g) >= 2:
                    metric_card("평균 생중량",
                                f"{g.mean():.2f}g",
                                delta=f"±{g.std():.2f}g (SD)")
                else:
                    st.info("생중량 데이터가 부족해요.")
            else:
                st.info("생중량 열이 없어요. (지상부+지하부 질량으로 자동 계산되지 않았어요.)")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ─────────────────────────────────────────────
# Tab 2: 환경 데이터
# ─────────────────────────────────────────────
with tab2:
    try:
        if len(env_df) == 0:
            st.info("🌡️ 환경 CSV를 사이드바에 올려주세요. 이 탭의 내용이 표시됩니다.")
        else:
            # 환경 요약 표
            st.subheader("🌡️ 환경 요약")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns]
            if env_cols:
                env_numbers = env_df[env_cols].dropna()
                if len(env_numbers) > 0:
                    summary = pd.DataFrame({
                        "항목": [c for c in env_cols],
                        "평균": [env_numbers[c].mean() for c in env_cols],
                        "최소": [env_numbers[c].min() for c in env_cols],
                        "최대": [env_numbers[c].max() for c in env_cols],
                    })
                    summary["평균"] = summary["평균"].map(lambda x: f"{x:.2f}")
                    summary["최소"] = summary["최소"].map(lambda x: f"{x:.2f}")
                    summary["최대"] = summary["최대"].map(lambda x: f"{x:.2f}")
                    # 단위 추가
                    label_map = {
                        "temperature": "온도 (°C)",
                        "humidity": "습도 (%)",
                        "ph": "pH",
                        "ec": "EC (mS/cm)",
                    }
                    summary["항목"] = [label_map.get(c, c) for c in summary["항목"]]
                    st.dataframe(summary, width="stretch")
                else:
                    st.info("환경 수치 데이터가 없어요.")
            else:
                st.info("온도/습도/pH/EC 열이 모두 없는 환경 CSV예요.")

            # 날짜별 시계열
            if has_col(env_df, "time"):
                st.subheader("📅 날짜별 환경 변화 (일별 평균)")

                daily = env_df.copy()
                daily["date"] = daily["time"].dt.date
                daily_grouped = daily.groupby("date")

                env_plot_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                                 if c in daily.columns]
                if env_plot_cols:
                    n_plots = len(env_plot_cols)
                    fig_daily = make_subplots(
                        rows=n_plots, cols=1,
                        subplot_titles=[
                            {"temperature": "온도 (°C)", "humidity": "습도 (%)",
                             "ph": "pH", "ec": "EC (mS/cm)"}[c]
                            for c in env_plot_cols
                        ],
                        vertical_spacing=0.12,
                    )
                    colors = {
                        "temperature": "#FF6B6B",
                        "humidity": "#4ECDC4",
                        "ph": "#9B59B6",
                        "ec": "#F39C12",
                    }
                    for i, col in enumerate(env_plot_cols):
                        grp = daily_grouped[col].mean().dropna()
                        if len(grp) >= 2:
                            fig_daily.add_trace(
                                go.Scatter(
                                    x=grp.index,
                                    y=grp.values,
                                    mode="lines+markers",
                                    name=col,
                                    line=dict(color=colors.get(col, "#3498DB"),
                                              width=2),
                                    marker=dict(size=4),
                                    showlegend=False,
                                ),
                                row=i + 1, col=1,
                            )
                    fig_daily.update_layout(
                        height=280 * n_plots,
                        showlegend=False,
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    set_kr_font(fig_daily)
                    st.plotly_chart(fig_daily, width="stretch")
                else:
                    st.info("그래프를 그릴 환경 수치 열이 없어요.")

            # expander: 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 보기 & 다운로드"):
                if has_col(env_df, "time"):
                    display_df = env_df.copy()
                    display_df["time"] = display_df["time"].dt.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(display_df, width="stretch", height=300)
                else:
                    st.dataframe(env_df, width="stretch", height=300)
                if len(env_df) > 0:
                    download_csv(env_df, "환경데이터_CSV다운로드")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ─────────────────────────────────────────────
# Tab 3: 생육 결과
# ─────────────────────────────────────────────
with tab3:
    try:
        if len(growth_df) == 0:
            st.info("📊 생육 결과 CSV를 사이드바에 올려주세요. 이 탭의 내용이 표시됩니다.")
        else:
            # ── 핵심 결과 카드 ──────────────────────
            st.subheader("🥇 핵심 결과")
            c1, c2, c3 = st.columns(3)

            if has_col(growth_df, "생중량"):
                g = growth_df["생중량"].dropna()
                if len(g) >= 2:
                    with c1:
                        metric_card("평균 생중량", f"{g.mean():.2f}g",
                                    delta=f"표준편차 {g.std():.2f}g")
                    with c2:
                        max_idx = g.idxmax()
                        max_row = growth_df.loc[max_idx]
                        max_id = max_row["개체번호"] if "개체번호" in growth_df.columns else "?"
                        # 개체번호가 float로 저장된 경우 정수처럼 표시
                        try:
                            max_id_disp = int(max_id) if float(max_id) == int(max_id) else max_id
                        except Exception:
                            max_id_disp = max_id
                        metric_card("최대 생중량 개체",
                                    f"개체 {max_id_disp}: {g.max():.2f}g")
                    with c3:
                        min_idx = g.idxmin()
                        min_row = growth_df.loc[min_idx]
                        min_id = min_row["개체번호"] if "개체번호" in growth_df.columns else "?"
                        try:
                            min_id_disp = int(min_id) if float(min_id) == int(min_id) else min_id
                        except Exception:
                            min_id_disp = min_id
                        metric_card("최소 생중량 개체",
                                    f"개체 {min_id_disp}: {g.min():.2f}g")
                else:
                    st.info("생중량 데이터가 2개 미만이라 핵심 지표를 계산할 수 없어요.")
            else:
                st.info("생중량 열이 없어서 핵심 결과 카드를 표시할 수 없어요.")

            # ── 개체별 생육 비교 (막대 그래프) ────────
            st.subheader("📊 개체별 생육 비교")
            bar_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                        if c in growth_df.columns and c != "개체번호"]
            if bar_cols:
                n_bar = len(bar_cols)
                fig_bar = make_subplots(
                    rows=n_bar, cols=1,
                    subplot_titles=bar_cols,
                    vertical_spacing=0.10,
                )
                for i, col in enumerate(bar_cols):
                    vals = growth_df[col].dropna()
                    ids = growth_df.loc[vals.index, "개체번호"] if "개체번호" in growth_df.columns else vals.index
                    # 상위 25% / 하위 25% 경계
                    if len(vals) >= 4:
                        q75 = vals.quantile(0.75)
                        q25 = vals.quantile(0.25)
                    else:
                        q75, q25 = vals.max(), vals.min()
                    colors_bar = []
                    for v in vals:
                        if v >= q75:
                            colors_bar.append("#27AE60")  # 상위: 초록
                        elif v <= q25:
                            colors_bar.append("#E74C3C")  # 하위: 빨강
                        else:
                            colors_bar.append("#3498DB")  # 중간: 파랑
                    fig_bar.add_trace(
                        go.Bar(
                            x=[str(x) for x in ids],
                            y=vals.values,
                            marker=dict(color=colors_bar),
                            name=col,
                            showlegend=(i == 0),
                        ),
                        row=i + 1, col=1,
                    )
                fig_bar.update_layout(
                    height=260 * n_bar,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=40, r=20, t=40, b=30),
                )
                set_kr_font(fig_bar)
                st.plotly_chart(fig_bar, width="stretch")
                st.caption("⭐ 막대 색: 초록 = 상위 25%, 빨강 = 하위 25%, 파랑 = 중간")
            else:
                st.info("그래프를 그릴 생육 측정 열이 없어요.")

            # ── 생중량 분포 (박스플롯) ────────────────
            if has_col(growth_df, "생중량"):
                st.subheader("📦 생중량 분포")
                g_vals = growth_df["생중량"].dropna()
                if len(g_vals) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=g_vals.values,
                            name="생중량",
                            fillcolor="#3498DB",
                            line=dict(color="#2980B9", width=2),
                            boxpoints="all",
                            jitter=0.3,
                            pointpos=-1.5,
                            marker=dict(size=5, color="#2980B9"),
                            hovertemplate="생중량: %{y:.2f}g<extra></extra>",
                        )
                    )
                    fig_box.update_layout(
                        yaxis_title="생중량 (g)",
                        height=350,
                        margin=dict(l=50, r=20, t=30, b=40),
                        showlegend=False,
                    )
                    set_kr_font(fig_box)
                    st.plotly_chart(fig_box, width="stretch")

                    # CV 표시
                    cv = (g_vals.std() / g_vals.mean()) if g_vals.mean() != 0 else 0
                    st.caption(
                        f"💧 **CV(변동계수) = {cv:.2f}**  "
                        f"(표준편차 {g_vals.std():.2f} ÷ 평균 {g_vals.mean():.2f})  "
                        f"— 값이 클수록 개체 간 차이가 크다는 뜻"
                    )
                else:
                    st.info("생중량 데이터가 부족해서 박스플롯을 그릴 수 없어요.")
            else:
                st.info("생중량 열이 없어서 분포 그래프를 표시할 수 없어요.")

            # ── 상관관계 분석 (산점도 + 추세선) ──────
            st.subheader("🔗 상관관계 분석")
            corr_cols = [c for c in ["잎 수", "지상부 길이"]
                         if c in growth_df.columns and "생중량" in growth_df.columns]
            if corr_cols and has_col(growth_df, "생중량"):
                for col in corr_cols:
                    x_col = col
                    y_col = "생중량"
                    fig_scatter = go.Figure()
                    subset = growth_df[[x_col, y_col]].dropna()
                    # 개체번호는 산점도에 표시하지 않음
                    if len(subset) >= 2:
                        # 산점도
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=subset[x_col],
                                y=subset[y_col],
                                mode="markers",
                                marker=dict(
                                    size=8,
                                    color="#3498DB",
                                    opacity=0.7,
                                    line=dict(width=1, color="white"),
                                ),
                                name=f"{x_col} vs 생중량",
                                hovertemplate=(
                                    f"{x_col}: %{{x:.1f}}<br>생중량: %{{y:.2f}}g<extra></extra>"
                                ),
                            )
                        )
                        # 추세선
                        add_trendline(fig_scatter, subset[x_col], subset[y_col])
                        # 상관계수 계산
                        r = np.corrcoef(subset[x_col], subset[y_col])[0, 1]
                        fig_scatter.update_layout(
                            xaxis_title=x_col,
                            yaxis_title="생중량 (g)",
                            height=380,
                            margin=dict(l=50, r=20, t=30, b=40),
                        )
                        set_kr_font(fig_scatter)
                        st.plotly_chart(fig_scatter, width="stretch")
                        strength = ("강한 양의 관계" if r > 0.7 else
                                    "중간 정도 양의 관계" if r > 0.4 else
                                    "약한 관계" if r > 0 else
                                    "음의 관계")
                        st.caption(
                            f"📐 **상관계수 r = {r:.2f}**  "
                            f"— {strength} (1에 가까울수록 함께 증가, "
                            f"−1에 가까울수록 한쪽이 증가할 때 다른 쪽이 감소)"
                        )
                    else:
                        st.info(f"{col} vs 생중량 데이터가 2개 미만이라 상관관계를 계산할 수 없어요.")
            else:
                st.info("잎 수·지상부 길이와 생중량 데이터가 모두 필요해요.")

            # ── expander: 원본 데이터 + 다운로드 ─────
            with st.expander("📄 생육 데이터 원본 보기 & 다운로드"):
                st.dataframe(growth_df, width="stretch", height=300)
                if len(growth_df) > 0:
                    download_csv(growth_df, "생육결과데이터_CSV다운로드")
