import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

# -----------------------------------------------------------------------------
# 0. 페이지 기본 설정 및 세션 상태 초기화
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CSV 데이터로 배우는 선형회귀 실험실",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태(st.session_state) 초기화: 탭 이동 간 데이터 및 모델 정보 유지
if "df" not in st.session_state:
    st.session_state.df = None
if "simple_model_results" not in st.session_state:
    st.session_state.simple_model_results = None
if "multi_model_results" not in st.session_state:
    st.session_state.multi_model_results = None


# -----------------------------------------------------------------------------
# 1. 주요 모듈별 함수 정의 (인공지능 및 데이터 처리 로직)
# -----------------------------------------------------------------------------

def generate_sample_data():
    """
    기온, 습도, 풍속, 강수량과 PM2.5 사이의 실제 기상 관계를 반영한 120행 샘플 데이터 생성
    """
    np.random.seed(42)
    n_samples = 120

    temperature = np.random.uniform(-5, 35, n_samples)          # 기온 (-5°C ~ 35°C)
    humidity = np.random.uniform(20, 95, n_samples)             # 습도 (20% ~ 95%)
    wind_speed = np.random.uniform(0.5, 8.0, n_samples)         # 풍속 (0.5m/s ~ 8m/s)
    rainfall = np.where(np.random.rand(n_samples) > 0.7, np.random.uniform(1, 30, n_samples), 0.0) # 강수량

    # 물리적 경향성 반영: 풍속이 셀수록, 강수량이 많을수록 PM2.5 감소 / 기온과 습도는 세정에 영향
    pm25 = (
        50.0 
        + (temperature * 0.4) 
        + (humidity * 0.2) 
        - (wind_speed * 5.5) 
        - (rainfall * 1.2) 
        + np.random.normal(0, 5, n_samples)
    )
    # PM2.5는 음수가 될 수 없음
    pm25 = np.clip(pm25, 5, 150)

    df_sample = pd.DataFrame({
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 1),
        "rainfall": np.round(rainfall, 1),
        "pm25": np.round(pm25, 1)
    })
    return df_sample


def load_csv(uploaded_file):
    """
    UTF-8 및 CP949 인코딩을 자동 지원하여 CSV 파일 읽기
    """
    try:
        # UTF-8 시도
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        return df, None
    except UnicodeDecodeError:
        try:
            # CP949(EUC-KR) 시도
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="cp949")
            return df, None
        except Exception as e:
            return None, f"파일 인코딩 오류: UTF-8 또는 CP949 형식이어야 합니다. ({str(e)})"
    except Exception as e:
        return None, f"CSV 파일을 읽는 중 오류가 발생했습니다: {str(e)}"


def validate_data(df):
    """
    업로드된 데이터의 숫자형 열 개수 및 행 수 유효성 검사
    """
    if df is None:
        return False, "데이터가 업로드되지 않았습니다."
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return False, "선형회귀 분석을 수행하려면 최소 2개 이상의 숫자형(수치형) 열이 필요합니다."
    
    if len(df) < 10:
        return False, "데이터 행 수가 너무 적습니다 (10개 미만). 최소 10개 이상의 데이터가 필요합니다."
    
    return True, f"유효한 데이터입니다. (총 {len(df)}행, 숫자형 열 {len(numeric_cols)}개)"


def calculate_metrics(y_true, y_pred, n_features):
    """
    모델 평가 지표(MAE, MSE, RMSE, R2, Adjusted R2) 계산
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    # 자유도 보정 R2 (조정된 R2)
    if n - n_features - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = r2

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2
    }


def train_simple_regression(df, x_col, y_col, test_size):
    """
    단순선형회귀 모델 학습 및 결과 반환
    """
    data = df[[x_col, y_col]].dropna()
    X = data[[x_col]]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics = calculate_metrics(y_test, y_pred_test, n_features=1)
    
    # 기울기 및 절편
    slope = model.coef_[0]
    intercept = model.intercept_

    return {
        "model": model,
        "x_col": x_col,
        "y_col": y_col,
        "slope": slope,
        "intercept": intercept,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "metrics": metrics
    }


def train_multiple_regression(df, x_cols, y_col, test_size, use_standardization=False):
    """
    다중선형회귀 모델 학습 및 결과 반환 (표준화 옵션 포함)
    """
    data = df[x_cols + [y_col]].dropna()
    X = data[x_cols]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if use_standardization:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        model = pipeline
        coefficients = pipeline.named_steps['regressor'].coef_
        intercept = pipeline.named_steps['regressor'].intercept_
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        coefficients = model.coef_
        intercept = model.intercept_

    y_pred_test = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred_test, n_features=len(x_cols))

    return {
        "model": model,
        "x_cols": x_cols,
        "y_col": y_col,
        "coefficients": coefficients,
        "intercept": intercept,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "metrics": metrics,
        "use_standardization": use_standardization
    }


def explain_coefficient(x_name, slope, y_name):
    """
    회귀계수의 의미를 학생 눈높이에 맞게 해석 문장 생성
    """
    direction = "증가" if slope > 0 else "감소"
    abs_slope = abs(slope)
    return f"💡 **해석 안내**: 다른 조건이 동일할 때, **{x_name}** 변수가 1단위 증가하면 **{y_name}** 예측값은 평균적으로 약 **{abs_slope:.2f}**만큼 **{direction}**합니다. (단, 이는 데이터 기반의 통계적 경향일 뿐 직접적인 인과관계를 의미하지는 않습니다.)"


# -----------------------------------------------------------------------------
# 2. 사이드바 구성 (학습 단계를 유도하고 주요 용어 정리)
# -----------------------------------------------------------------------------

st.sidebar.title("📌 인공지능 기초 - 선형회귀")
st.sidebar.info("CSV 데이터를 업로드하고 단순/다중 선형회귀 모델을 만든 후 직접 평가해 봅시다.")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 핵심 용어 사전")
with st.sidebar.expander("용어 설명 보기"):
    st.markdown("""
    - **독립변수(X)**: 원인이 되는 변수 (입력값)
    - **종속변수(y)**: 결과가 되는 변수 (목표 예측값)
    - **회귀계수(기울기)**: X가 1 변할 때 y가 변화하는 정도
    - **절편**: X가 0일 때 y의 기본값
    - **잔차(Residual)**: 실제값과 예측값의 차이 ($y - \hat{y}$)
    - **R² (결정계수)**: 모델이 데이터를 얼마나 잘 설명하는지 나타내는 지표 (0~1)
    """)

st.sidebar.markdown("---")
st.sidebar.caption("고등학교 인공지능 기초 수업용 실습 도구")


# -----------------------------------------------------------------------------
# 3. 메인 화면 및 6개 탭 구상
# -----------------------------------------------------------------------------

st.title("🧪 CSV 데이터로 배우는 선형회귀 실험실")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ 학습 안내",
    "2️⃣ CSV 데이터 업로드",
    "3️⃣ 데이터 탐색",
    "4️⃣ 단순선형회귀",
    "5️⃣ 다중선형회귀",
    "6️⃣ 모델 평가 및 비교"
])


# =============================================================================
# TAB 1: 학습 안내
# =============================================================================
with tab1:
    st.header("📘 선형회귀(Linear Regression) 핵심 개념 정리")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 회귀(Regression)와 선형회귀란?")
        st.write("""
        * **회귀(Regression)**: 연속된 숫자(예: 기온, 키, 집값, 미세먼지 농도 등)를 예측하는 대표적인 **지도학습** 알고리즘입니다.
        * **선형회귀(Linear Regression)**: 입력 변수($X$)와 출력 변수($y$) 사이의 관계를 가장 잘 설명하는 **직선(또는 평면)**을 찾는 기법입니다.
        """)

        st.subheader("2. 변수의 종류")
        st.markdown("""
        * **독립변수 ($X$)**: 예측에 사용되는 입력 데이터 (예: 풍속, 기온)
        * **종속변수 ($y$)**: 우리가 맞추고자 하는 결과 목표값 (예: 초미세먼지 농도)
        """)

    with col2:
        st.subheader("3. 수식으로 보는 선형회귀")
        st.markdown("**단순선형회귀** (독립변수가 1개일 때):")
        st.latex(r"\hat{y} = b_0 + b_1 x")

        st.markdown("**다중선형회귀** (독립변수가 2개 이상일 때):")
        st.latex(r"\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n")

        st.caption("※ $b_0$는 절편(y-intercept), $b_1, b_2 \dots$는 기울기(회귀계수), $\hat{y}$는 모델의 예측값입니다.")

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("4. 실제값, 예측값, 잔차")
        st.info("""
        * **실제값 ($y$)**: 실제 관측된 진짜 데이터
        * **예측값 ($\hat{y}$)**: 회귀 방정식이 계산해 낸 값
        * **잔차 (Residual)**: $\text{실제값} - \text{예측값} = y - \hat{y}$
        
        선형회귀의 목표는 모든 데이터 점들과 회귀선 사이의 **잔차의 제곱합을 최소로 만드는 선**을 찾는 것입니다.
        """)

    with col4:
        st.subheader("⚠️ 주의: 상관관계 vs 인과관계")
        st.warning("""
        * **상관관계**: 두 변수가 함께 변하는 경향성 (예: 여름철 아이스크림 판매량과 물놀이 사고 수)
        * **인과관계**: 한 변수가 다른 변수의 직접적인 원인이 되는 관계
        
        **회귀 분석에서 강한 연관성이 발견되었다고 해서 반드시 한 변수가 다른 변수의 원인이라는 뜻은 아닙니다!**
        """)

    st.markdown("---")
    with st.expander("❓ [탐구 질문 1] 학습 안내 확인하기"):
        st.markdown("""
        1. 독립변수 $X$가 1만큼 증가할 때 종속변수 $y$가 변화하는 양을 나타내는 수식의 요소는 무엇인가요?
        2. 잔차가 양수($+)라는 것은 모델의 예측값이 실제값보다 크다는 뜻일까요, 작다는 뜻일까요?
        """)


# =============================================================================
# TAB 2: CSV 데이터 업로드
# =============================================================================
with tab2:
    st.header("📂 CSV 데이터 업로드 및 준비")

    st.markdown("""
    실습할 CSV 파일을 업로드하세요. 준비된 데이터 파일이 없다면 아래 예제 데이터 다운로드 버튼을 눌러 **'미세먼지 예측 예제 데이터'**를 받아 사용해 보세요.
    """)

    # 예제 데이터 생성 및 다운로드 버튼
    sample_df = generate_sample_data()
    csv_buffer = io.BytesIO()
    sample_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    
    st.download_button(
        label="📥 예제 데이터셋 (미세먼지_예측_데이터.csv) 다운로드",
        data=csv_buffer.getvalue(),
        file_name="미세먼지_예측_데이터.csv",
        mime="text/csv"
    )

    st.markdown("---")

    uploaded_file = st.file_uploader("분석할 CSV 파일을 선택하세요 (UTF-8, CP949 지원)", type=["csv"])

    if uploaded_file is not None:
        df, err_msg = load_csv(uploaded_file)
        if err_msg:
            st.error(err_msg)
        else:
            st.session_state.df = df
            st.success("데이터가 성공적으로 업로드되었습니다!")
    else:
        if st.session_state.df is None:
            st.info("💡 위의 버튼을 통해 예제 데이터를 다운로드받거나, 소장하고 계신 CSV 파일을 업로드해 주세요.")

    # 데이터가 존재하는 경우 기본 정보 출력
    if st.session_state.df is not None:
        current_df = st.session_state.df

        st.subheader("📊 데이터 미리보기")
        st.dataframe(current_df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("전체 행 수 (Row)", f"{current_df.shape[0]} 개")
        col2.metric("전체 열 수 (Column)", f"{current_df.shape[1]} 개")

        num_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        str_cols = current_df.select_dtypes(exclude=[np.number]).columns.tolist()
        col3.metric("숫자형 변수 개수", f"{len(num_cols)} 개")

        # 데이터 세부 정보 표
        st.subheader("📋 변수별 데이터 유형 및 결측치 현황")
        info_df = pd.DataFrame({
            "데이터 타입": current_df.dtypes.astype(str),
            "결측치(Missing Value) 수": current_df.isnull().sum(),
            "구분": ["숫자형(수치형)" if col in num_cols else "문자형/기타" for col in current_df.columns]
        })
        st.table(info_df)

        # 데이터 유효성 경고
        is_valid, msg = validate_data(current_df)
        if not is_valid:
            st.error(f"⚠️ {msg}")
        else:
            if len(current_df) < 30:
                st.warning("⚠️ 데이터 개수가 30개 미만입니다. 모델의 평가 결과가 부정확할 수 있으니 주의하세요.")
            else:
                st.success(f"✅ {msg}")

    with st.expander("❓ [탐구 질문 2] 데이터 업로드 확인하기"):
        st.markdown("""
        1. 업로드한 데이터셋에서 종속변수(y)로 사용하기 적절한 숫자형 변수는 무엇인가요?
        2. 결측치(Null/NaN)가 존재하는 데이터는 머신러닝 학습 시 어떤 문제를 일으킬 수 있을까요?
        """)


# =============================================================================
# TAB 3: 데이터 탐색
# =============================================================================
with tab3:
    st.header("🔍 탐색적 데이터 분석 (EDA)")

    if st.session_state.df is None:
        st.warning("데이터를 먼저 업로드해 주세요 (2단계 탭).")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("데이터 탐색을 위해서는 최소 2개 이상의 숫자형 변수가 필요합니다.")
        else:
            st.subheader("1. 기술통계량")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            st.markdown("---")
            st.subheader("2. 변수 분포 및 관계 시각화 (산점도 & 히스토그램)")

            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X축 변수 선택", numeric_cols, index=0, key="eda_x")
            with col2:
                # 기본 y축 선택 시 X축과 다른 변수를 지정
                default_y_idx = 1 if len(numeric_cols) > 1 else 0
                y_var = st.selectbox("Y축 변수 선택", numeric_cols, index=default_y_idx, key="eda_y")

            col_fig1, col_fig2 = st.columns(2)
            with col_fig1:
                fig_hist = px.histogram(
                    df, x=x_var, title=f"[{x_var}] 변수 분포 히스토그램",
                    marginal="box", color_discrete_sequence=['#3366CC']
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_fig2:
                fig_scatter = px.scatter(
                    df, x=x_var, y=y_var, 
                    title=f"[{x_var}] vs [{y_var}] 산점도",
                    trendline="ols", trendline_color_override="red",
                    color_discrete_sequence=['#109618']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # 탐구 질의 가이드
            st.info(f"""
            📌 **산점도 분석 가이드**
            * **[{x_var}]** 와 **[{y_var}]** 는 어떤 관계를 보이나요?
            * 점들이 붉은색 회귀선 주위에 밀집해 있나요, 흩어져 있나요?
            * 다른 데이터들과 멀리 떨어져 있는 **이상치(Outlier)**가 관측되나요?
            """)

            st.markdown("---")
            st.subheader("3. 상관계수(Correlation) 분석")

            corr_matrix = df[numeric_cols].corr()

            col_corr1, col_corr2 = st.columns([1, 1])
            with col_corr1:
                st.markdown("**상관계수 표**")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.3f}"), use_container_width=True)

            with col_corr2:
                st.markdown("**상관계수 히트맵**")
                fig_heatmap = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="변수 간 피어슨 상관계수 히트맵"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            st.caption("※ 상관계수는 -1부터 1 사이의 값을 가집니다. 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 나타냅니다.")

            with st.expander("❓ [탐구 질문 3] 데이터 탐색 질문하기"):
                st.markdown("""
                1. 두 변수 사이의 상관계수가 0에 가깝다면 선형회귀 모델로 예측하기 적절할까요?
                2. 두 변수가 높게 상관되어 있다면, 한 변수가 증가함에 따라 다른 변수가 반드시 원인이 되어 변하는 것일까요?
                """)


# =============================================================================
# TAB 4: 단순선형회귀
# =============================================================================
with tab4:
    st.header("📈 단순선형회귀 (Simple Linear Regression)")

    if st.session_state.df is None:
        st.warning("데이터를 먼저 업로드해 주세요 (2단계 탭).")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("선형회귀 분석을 수행하려면 최소 2개 이상의 숫자형 변수가 필요합니다.")
        else:
            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                x_col = st.selectbox("독립변수 (X) 선택", numeric_cols, index=0, key="simple_x")
            with col_sel2:
                # y변수 자동 선택 (x와 겹치지 않게)
                y_options = [c for c in numeric_cols if c != x_col]
                y_col = st.selectbox("종속변수 (y) 선택", y_options, index=0, key="simple_y")
            with col_sel3:
                test_size = st.slider("테스트 데이터 비율 (Test Size)", 0.1, 0.4, 0.2, step=0.05, key="simple_test_size")

            if st.button("🚀 단순선형회귀 모델 학습하기", key="btn_train_simple"):
                res = train_simple_regression(df, x_col, y_col, test_size)
                st.session_state.simple_model_results = res
                st.success("모델 학습 완료!")

            # 학습 결과 표시
            if st.session_state.simple_model_results is not None:
                res = st.session_state.simple_model_results

                # 현재 선택 변수와 저장된 결과의 변수가 일치하는지 확인
                if res['x_col'] == x_col and res['y_col'] == y_col:
                    st.markdown("---")
                    st.subheader("1. 모델 학습 결과 및 회귀식")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("학습 데이터 수", f"{len(res['X_train'])}개")
                    col_m2.metric("테스트 데이터 수", f"{len(res['X_test'])}개")
                    col_m3.metric("기울기 (Slope)", f"{res['slope']:.4f}")
                    col_m4.metric("절편 (Intercept)", f"{res['intercept']:.4f}")

                    # 작성된 회귀식 표출
                    equation_str = f"**예측 {y_col}** = ({res['slope']:.4f}) × **{x_col}** + ({res['intercept']:.4f})"
                    st.success(f"📐 도출된 단순회귀식: {equation_str}")

                    # 회귀계수 해석
                    st.info(explain_coefficient(x_col, res['slope'], y_col))

                    st.markdown("---")
                    st.subheader("2. 회귀선 및 잔차 시각화")

                    # 학습 데이터와 테스트 데이터 시각화
                    fig_reg = go.Figure()

                    # Train 데이터 점
                    fig_reg.add_trace(go.Scatter(
                        x=res['X_train'][x_col], y=res['y_train'],
                        mode='markers', name='Train Data',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    # Test 데이터 점
                    fig_reg.add_trace(go.Scatter(
                        x=res['X_test'][x_col], y=res['y_test'],
                        mode='markers', name='Test Data',
                        marker=dict(color='orange', size=8)
                    ))

                    # 회귀선 그리기
                    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
                    y_range = res['slope'] * x_range + res['intercept']
                    fig_reg.add_trace(go.Scatter(
                        x=x_range, y=y_range,
                        mode='lines', name='Linear Regression Line',
                        line=dict(color='red', width=2)
                    ))

                    fig_reg.update_layout(
                        title=f"[{x_col}] vs [{y_col}] 단순선형회귀선",
                        xaxis_title=x_col, yaxis_title=y_col
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)

                    # 잔차(Residuals) 시각화 그래프
                    st.markdown("**테스트 데이터 기준 잔차(Residuals) 시각화**")
                    fig_res = go.Figure()
                    
                    # 실제 점
                    fig_res.add_trace(go.Scatter(
                        x=res['X_test'][x_col], y=res['y_test'],
                        mode='markers', name='실제값 (Actual)',
                        marker=dict(color='orange')
                    ))
                    # 예측 점
                    fig_res.add_trace(go.Scatter(
                        x=res['X_test'][x_col], y=res['y_pred_test'],
                        mode='markers', name='예측값 (Predicted)',
                        marker=dict(color='red', symbol='x')
                    ))

                    # 잔차 선 연결 (선으로 수직 오차 표시)
                    for x_val, y_real, y_pred in zip(res['X_test'][x_col], res['y_test'], res['y_pred_test']):
                        fig_res.add_shape(
                            type="line", x0=x_val, y0=y_real, x1=x_val, y1=y_pred,
                            line=dict(color="gray", width=1, dash="dot")
                        )

                    fig_res.update_layout(
                        title="실제값과 예측값 사이의 잔차(점선) 확인",
                        xaxis_title=x_col, yaxis_title=y_col
                    )
                    st.plotly_chart(fig_res, use_container_width=True)

                    st.markdown("---")
                    st.subheader("3. 새로운 값 예측해 보기")
                    min_val = float(df[x_col].min())
                    max_val = float(df[x_col].max())
                    mean_val = float(df[x_col].mean())

                    user_input_x = st.number_input(
                        f"새로운 [{x_col}] 값 입력:",
                        min_value=min_val - 100.0, max_value=max_val + 100.0,
                        value=mean_val
                    )

                    predicted_y = res['slope'] * user_input_x + res['intercept']
                    st.metric(f"🎯 예측된 [{y_col}] 값", f"{predicted_y:.2f}")

                    if predicted_y < 0:
                        st.warning("⚠️ **선형회귀 모델의 한계**: 예측값이 음수로 계산되었습니다. PM2.5나 키, 가격처럼 물리적으로 음수가 존재할 수 없는 변수라도 선형회귀 모델은 일정한 직선 기울기를 계속 따라가기 때문에 음수를 출력할 수 있습니다.")

                    st.caption("“이 값은 데이터에서 학습한 선형적인 경향을 이용한 예측값이며 실제값과 다를 수 있습니다.”")

    with st.expander("❓ [탐구 질문 4] 단순선형회귀 확인하기"):
        st.markdown("""
        1. 회귀선에 완전히 겹쳐지지 않는 데이터 점들이 존재하는 이유는 무엇일까요?
        2. 테스트 데이터 비율(Test Size)을 너무 높게 잡으면(예: 0.5 이상) 모델 학습 과정에 어떤 영향을 미칠까요?
        """)


# =============================================================================
# TAB 5: 다중선형회귀
# =============================================================================
with tab5:
    st.header("📊 다중선형회귀 (Multiple Linear Regression)")

    if st.session_state.df is None:
        st.warning("데이터를 먼저 업로드해 주세요 (2단계 탭).")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            st.error("다중선형회귀를 수행하려면 최소 3개 이상의 숫자형 변수가 필요합니다. (독립변수 2개 이상 + 종속변수 1개)")
        else:
            col_m_y, col_m_test = st.columns(2)
            with col_m_y:
                y_col_multi = st.selectbox("종속변수 (y) 선택", numeric_cols, index=len(numeric_cols)-1, key="multi_y")
            with col_m_test:
                test_size_multi = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, step=0.05, key="multi_test_size")

            available_x = [col for col in numeric_cols if col != y_col_multi]
            selected_x_cols = st.multiselect("독립변수들 (X) 선택 (2개 이상 선택):", available_x, default=available_x[:2])

            use_std = st.checkbox("입력 변수 표준화(StandardScaler) 적용", value=False)
            if use_std:
                st.caption("💡 **표준화(Standardization)**: 서로 다른 단위(예: 기온°C, 강수량mm, 풍속m/s)를 평균 0, 표준편차 1로 맞추어 회귀계수의 크기를 상대적으로 비교할 수 있게 합니다.")

            if len(selected_x_cols) < 2:
                st.warning("⚠️ 다중선형회귀를 실행하려면 독립변수(X)를 최소 2개 이상 선택해야 합니다.")
            else:
                if st.button("🚀 다중선형회귀 모델 학습하기", key="btn_train_multi"):
                    res_multi = train_multiple_regression(df, selected_x_cols, y_col_multi, test_size_multi, use_std)
                    st.session_state.multi_model_results = res_multi
                    st.success("다중선형회귀 모델 학습 완료!")

            # 다중선형회귀 결과 표시
            if st.session_state.multi_model_results is not None:
                res_m = st.session_state.multi_model_results

                # 현재 선택 변수 상태 체크
                if set(res_m['x_cols']) == set(selected_x_cols) and res_m['y_col'] == y_col_multi:
                    st.markdown("---")
                    st.subheader("1. 다중선형회귀 계수 및 방정식")

                    st.write(f"**절편 (Intercept)**: `{res_m['intercept']:.4f}`")

                    # 회귀계수 데이터프레임
                    coef_df = pd.DataFrame({
                        "독립변수 (X)": res_m['x_cols'],
                        "회귀계수 (Coefficient)": res_m['coefficients']
                    })
                    
                    col_c1, col_c2 = st.columns([1, 1])
                    with col_c1:
                        st.dataframe(coef_df, use_container_width=True)

                    with col_c2:
                        fig_coef = px.bar(
                            coef_df, x="독립변수 (X)", y="회귀계수 (Coefficient)",
                            text_auto=".3f", title="변수별 회귀계수 크기 비교",
                            color="회귀계수 (Coefficient)", color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_coef, use_container_width=True)

                    st.warning("""
                    ⚠️ **회귀계수 해석 시주의사항**:
                    * 다중선형회귀의 회귀계수는 **다른 모든 입력 변수들이 일정하다고 가정했을 때**, 해당 변수가 1만큼 변할 때의 예측값 변화를 의미합니다.
                    * 변수마다 측정 단위가 다르면(예: m/s vs mm) 단순 회귀계수의 크기만으로 변수의 중요도를 직접 비교할 수 없습니다. (단위 영향을 없애려면 '표준화' 옵션을 사용하세요.)
                    """)

                    st.markdown("---")
                    st.subheader("2. 다중선형회귀 기반 시뮬레이션 (새로운 값 예측)")
                    
                    st.write("각 독립변수의 값을 입력하여 종속변수를 예측해 보세요:")
                    user_inputs = {}
                    
                    # 동적 입력 폼을 컬럼으로 나눔
                    input_cols = st.columns(min(len(res_m['x_cols']), 4))
                    for idx, col_name in enumerate(res_m['x_cols']):
                        col_target = input_cols[idx % 4]
                        default_val = float(df[col_name].mean())
                        user_inputs[col_name] = col_target.number_input(
                            f"[{col_name}]", 
                            value=default_val,
                            key=f"input_multi_{col_name}"
                        )

                    # 입력값을 데이터프레임 구조로 변환 후 예측
                    input_df = pd.DataFrame([user_inputs])
                    pred_multi_y = res_m['model'].predict(input_df)[0]

                    st.metric(f"🎯 예측된 [{y_col_multi}] 값", f"{pred_multi_y:.2f}")

                    if pred_multi_y < 0:
                        st.warning("⚠️ **선형회귀 모델의 한계**: 예측 결과가 음수로 산출되었습니다.")

    with st.expander("❓ [탐구 질문 5] 다중선형회귀 확인하기"):
        st.markdown("""
        1. 단순선형회귀에 비해 독립변수의 개수를 늘렸을 때 모델의 예측 능력은 어떻게 변했나요?
        2. 서로 매우 유사한 정보를 담고 있는 두 독립변수를 동시에 넣으면 어떤 문제가 발생할 수 있을까요? (힌트: 다중공선성)
        """)


# =============================================================================
# TAB 6: 모델 평가 및 비교
# =============================================================================
with tab6:
    st.header("⚖️ 모델 평가 및 성능 비교")

    simple_res = st.session_state.simple_model_results
    multi_res = st.session_state.multi_model_results

    if simple_res is None and multi_res is None:
        st.warning("단순선형회귀(4단계) 또는 다중선형회귀(5단계) 모델을 최소 하나 이상 학습해 주세요.")
    else:
        st.subheader("1. 모델 성능 지표 비교 표")

        comparison_data = []

        if simple_res is not None:
            m_s = simple_res['metrics']
            comparison_data.append({
                "모델 유형": "단순선형회귀",
                "사용한 독립변수": simple_res['x_col'],
                "R² (결정계수)": f"{m_s['R2']:.4f}",
                "조정된 R²": f"{m_s['Adj_R2']:.4f}",
                "MAE": f"{m_s['MAE']:.4f}",
                "MSE": f"{m_s['MSE']:.4f}",
                "RMSE": f"{m_s['RMSE']:.4f}"
            })

        if multi_res is not None:
            m_m = multi_res['metrics']
            comparison_data.append({
                "모델 유형": "다중선형회귀",
                "사용한 독립변수": ", ".join(multi_res['x_cols']),
                "R² (결정계수)": f"{m_m['R2']:.4f}",
                "조정된 R²": f"{m_m['Adj_R2']:.4f}",
                "MAE": f"{m_m['MAE']:.4f}",
                "MSE": f"{m_m['MSE']:.4f}",
                "RMSE": f"{m_m['RMSE']:.4f}"
            })

        comp_df = pd.DataFrame(comparison_data)
        st.table(comp_df)

        # 평가 지표 설명
        st.info("""
        📚 **평가 지표 가이드**
        * **MAE (Mean Absolute Error)**: 실제값과 예측값 차이의 절댓값 평균 (직관적 오차 크기)
        * **MSE (Mean Squared Error)**: 오차를 제곱하여 평균한 값 (큰 오차에 더 가혹한 벌점)
        * **RMSE (Root MSE)**: MSE에 제곱근을 씌워 원래 y 변수와 동일한 단위로 맞춰준 지표 (작을수록 좋음)
        * **R² (결정계수)**: 모델이 종속변수의 전체 변동성을 얼마나 설명하는지 비율 (1에 가까울수록 우수)
        * **조정된 R² (Adjusted R²)**: 변수를 쓸데없이 많이 추가할 때 발생하는 R² 상승 착시를 보정한 지표
        """)

        st.markdown("---")
        st.subheader("2. 실제값 vs 예측값 및 잔차 진단 그래프")

        # 분석할 모델 선택
        available_models = {}
        if simple_res is not None:
            available_models["단순선형회귀"] = simple_res
        if multi_res is not None:
            available_models["다중선형회귀"] = multi_res

        selected_model_name = st.radio("진단할 모델 선택:", list(available_models.keys()), horizontal=True)
        active_res = available_models[selected_model_name]

        y_true = active_res['y_test']
        y_pred = active_res['y_pred_test']
        residuals = y_true - y_pred

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            # 실제값 vs 예측값 산점도
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(go.Scatter(
                x=y_true, y=y_pred, mode='markers',
                marker=dict(color='purple', size=8),
                name='Data Points'
            ))
            # 이상적 기준선 y=x
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig_act_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(color='red', dash='dash'),
                name='기준선 (y=x)'
            ))
            fig_act_pred.update_layout(
                title=f"[{selected_model_name}] 실제값 vs 예측값",
                xaxis_title="실제값 (Actual y)", yaxis_title="예측값 (Predicted y)"
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with col_g2:
            # 잔차 분포 히스토그램
            fig_res_dist = px.histogram(
                residuals, nbins=20,
                title=f"[{selected_model_name}] 잔차 분포 히스토그램",
                labels={'value': '잔차 (y - y_hat)'},
                color_discrete_sequence=['#FF9900']
            )
            st.plotly_chart(fig_res_dist, use_container_width=True)

        # 잔차 산점도 (Residual Plot)
        fig_res_scatter = go.Figure()
        fig_res_scatter.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode='markers',
            marker=dict(color='teal', size=8)
        ))
        fig_res_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res_scatter.update_layout(
            title=f"[{selected_model_name}] 잔차 산점도 (Residual Plot)",
            xaxis_title="예측값 (Predicted y)", yaxis_title="잔차 (Residual)"
        )
        st.plotly_chart(fig_res_scatter, use_container_width=True)

        # 자동 해석 문장 제공
        st.success("""
        💡 **시각적 결과 자동 진단**
        * **실제값 vs 예측값**: 데이터 점들이 붉은 대각선(y=x)에 가까이 빽빽하게 모여 있을수록 예측 성능이 훌륭함을 의미합니다.
        * **잔차 산점도**: 잔차가 0(빨간 점선)을 중심으로 **특정 패턴 없이 무작위로 고르게 분포**해야 선형 모델의 가정에 부합합니다.
        * 만약 잔차가 U자 모양이나 곡선 패턴을 보인다면, 실제 데이터는 선형 관계가 아닌 비선형(2차식 등) 관계일 가능성이 높습니다.
        * 다중선형회귀의 R²가 더 높더라도, **MAE/RMSE 오차 값이 실제로 줄어들었는지**와 **조정된 R²**를 종합해서 평가하세요.
        """)

    with st.expander("❓ [탐구 질문 6] 종합 모델 평가 질문하기"):
        st.markdown("""
        1. 변수의 개수를 계속 늘리기만 하면 R² 값은 항상 증가하거나 유지됩니다. 왜 무작정 변수를 많이 추가하는 것이 좋지 않을까요?
        2. RMSE 지표와 MAE 지표 중 이상치(Outlier) 오차에 더 예민하게 반응하는 지표는 무엇이며, 그 이유는 무엇일까요?
        """)
