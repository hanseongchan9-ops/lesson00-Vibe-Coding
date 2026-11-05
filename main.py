import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# --- 1. 페이지 설정 및 디자인 스타일 적용 ---
st.set_page_config(
    page_title="Smart Power Guard",
    page_icon="⚡",
    layout="wide"
)

# 전기전자공학 컨셉에 어울리는 다스한 스타일 및 위험 단계 색상 적용 CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1a1c23;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. 타이틀 및 소개 ---
st.title("⚡ Smart Power Guard")
st.markdown("### 머신러닝 기반 가정용 전력 소비량 예측 및 전력 피크 경고 시스템")
st.write("과거 전력 사용 패턴과 시간 정보를 학습한 AI 모델을 통해 전력 소비량을 예측하고, 피크 위험 단계를 안내합니다.")

st.markdown("---")

# --- 3. 가상의 학습 데이터 생성 및 모델 학습 (실행용 데모 데이터) ---
# *실제 프로젝트에서는 전처리된 CSV 데이터를 pd.read_csv()로 불러와서 사용합니다.*
@st.cache_resource
def load_and_train_model():
    # 시뮬레이션을 위한 샘플 데이터 생성 (시간, 요일, 월, 기온 등)
    np.random.seed(42)
    n_samples = 1000
    
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)
    months = np.random.randint(1, 13, n_samples)
    temps = np.random.uniform(-10, 35, n_samples) # 기온 (겨울 한파, 여름 폭염 반영)
    
    # 전력 소비량 계산 로직 (기온이 아주 높거나 낮을 때, 저녁 시간대에 전력 소모가 크도록 설정)
    power = 1.5 + 0.05 * hours + 0.1 * np.abs(temps - 20) + np.random.normal(0, 0.3, n_samples)
    power = np.clip(power, 0.5, 6.0) # 최소 0.5kW ~ 최대 6.0kW 제한
    
    df = pd.DataFrame({
        'Hour': hours,
        'DayOfWeek': days,
        'Month': months,
        'Temperature': temps,
        'Power': power
    })
    
    # 입력값(X)과 정답값(y) 분리
    X = df[['Hour', 'DayOfWeek', 'Month', 'Temperature']]
    y = df['Power']
    
    # Random Forest 모델 학습
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, df

model, historical_df = load_and_train_model()

# --- 4. 사이드바: 사용자 입력 컨트롤러 ---
st.sidebar.header("🎛️ 예측 조건 설정")
st.sidebar.markdown("측정하고자 하는 시간대와 환경을 선택하세요.")

input_hour = st.sidebar.slider("시간대 (Hour)", 0, 23, 14)
input_temp = st.sidebar.slider("예상 기온 (°C)", -15.0, 38.0, 24.0)
input_month = st.sidebar.selectbox("월 (Month)", list(range(1, 13)), index=6) # 기본 7월(여름)
input_day = st.sidebar.selectbox("요일", ["월", "화", "수", "목", "금", "토", "일"], index=0)

day_mapping = {"월":0, "화":1, "수":2, "목":3, "금":4, "토":5, "일":6}
input_day_num = day_mapping[input_day]

# 사용자가 입력한 데이터를 데이터프레임으로 변환
input_data = pd.DataFrame([[input_hour, input_day_num, input_month, input_temp]], 
                          columns=['Hour', 'DayOfWeek', 'Month', 'Temperature'])

# --- 5. AI 모델 예측 수행 ---
predicted_power = model.predict(input_data)[0]

# 위험 단계 기준 설정 (학습 데이터의 분포 활용)
q1 = historical_df['Power'].quantile(0.4)
q2 = historical_df['Power'].quantile(0.8)

if predicted_power < q1:
    risk_level = "안전"
    risk_color = "blue"
    advice = "현재 전력 사용량이 안정적인 수준입니다. 평소처럼 전기를 이용하세요."
elif predicted_power < q2:
    risk_level = "주의"
    risk_color = "orange"
    advice = "전력 소비량이 평소보다 다소 높은 시간대입니다. 불필요한 전등이나 가전제품 대기전력을 신경 써주세요!"
else:
    risk_level = "경고"
    risk_color = "red"
    advice = "🚨 **전력 피크 위험 구간입니다!** 냉난방기 온도를 조절하고, 전력 과부하가 걸리지 않도록 각별히 주의하세요."

# --- 6. 웹앱 메인 화면 결과 출력 ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="📊 예상 전력 소비량", value=f"{predicted_power:.2f} kW")

with col2:
    if risk_level == "안전":
        st.success(f"### 전력 피크 위험 단계: {risk_level}")
    elif risk_level == "주의":
        st.warning(f"### 전력 피크 위험 단계: {risk_level}")
    else:
        st.error(f"### 전력 피크 위험 단계: {risk_level}")

st.markdown("---")

# 실천 안내 가이드 박스
st.info(f"💡 **에너지 절약 안내**: {advice}")

st.markdown("---")

# --- 7. 시각화 그래프 제공 ---
st.subheader("📈 시간대별 평균 전력 소비 패턴 및 비교")

# 시간대별 평균 전력 소비량 계산
hourly_avg = historical_df.groupby('Hour')['Power'].mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#1f77b4', linewidth=2, label='시간대별 평균 전력')
ax.axvline(x=input_hour, color='red', linestyle='--', label=f'선택한 시간 ({input_hour}시)')
ax.set_title("시간대별 전력 소비 추이", fontsize=14, color='white')
ax.set_xlabel("시간 (Hour)", color='white')
ax.set_ylabel("전력 소비량 (kW)", color='white')
ax.grid(True, linestyle='alpha=0.3')
ax.legend()

# 다크 모드 배경색 맞춤
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#1a1c23')
ax.tick_params(colors='white')

st.pyplot(fig)

# 수행평가 보고서용 안내 멘트
st.markdown("---")
st.caption("고등학교 데이터 과학 프로젝트 | 스마트 전력 가드 시스템 (Smart Power Guard)")
