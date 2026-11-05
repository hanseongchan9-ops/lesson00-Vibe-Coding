import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

# --- 1. 페이지 설정 및 디자인 스타일 적용 ---
st.set_page_config(
    page_title="Smart Power Guard",
    page_icon="⚡",
    layout="wide"
)

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

st.title("⚡ Smart Power Guard (실제 캐글 데이터 연동 버전)")
st.markdown("### UCI / Kaggle 가정용 전력 소비량 데이터 기반 AI 예측 및 전력 피크 경고 시스템")

# --- 2. 실제 데이터 불러오기 및 모델 학습 함수 ---
@st.cache_resource
def load_and_train_model():
    file_name = "household_power_consumption.txt"
    
    if not os.path.exists(file_name):
        st.error(f"⚠️ '{file_name}' 파일을 찾을 수 없습니다. 깃허브 저장소에 CSV/TXT 파일을 업로드해 주세요!")
        return None, None

    # 데이터 로드 (용량이 크므로 일부 행만 샘플링해서 로드하거나 전체 로드)
    # 캐글 데이터는 세미콜론(;) 구분자 사용
    df = pd.read_csv(file_name, sep=';', low_memory=False, na_values=['?'])
    
    # 결측치 제거
    df.dropna(subset=['Global_active_power'], inplace=True)
    
    # 날짜/시간 속성 추출
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    
    # 숫자로 변환
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])
    
    # 학습 속도를 위해 데이터가 너무 많으면 일부(예: 5000개) 샘플링 추출
    df_sample = df.sample(n=5000, random_state=42).copy()
    
    # 입력값(X)과 정답값(y) 설정 (여기서는 시간, 요일, 월을 입력으로 사용)
    X = df_sample[['Hour', 'DayOfWeek', 'Month']]
    y = df_sample['Global_active_power']
    
    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(X, y)
    
    return model, df_sample

model, historical_df = load_and_train_model()

if model is None:
    st.stop()

# --- 3. 사이드바: 사용자 입력 컨트롤러 ---
st.sidebar.header("🎛️ 예측 조건 설정")
input_hour = st.sidebar.slider("시간대 (Hour)", 0, 23, 14)
input_month = st.sidebar.selectbox("월 (Month)", list(range(1, 13)), index=6)
input_day = st.sidebar.selectbox("요일", ["월", "화", "수", "목", "금", "토", "일"], index=0)

day_mapping = {"월":0, "화":1, "수":2, "목":3, "금":4, "토":5, "일":6}
input_day_num = day_mapping[input_day]

input_data = pd.DataFrame([[input_hour, input_day_num, input_month]], 
                          columns=['Hour', 'DayOfWeek', 'Month'])

# --- 4. AI 모델 예측 수행 ---
predicted_power = model.predict(input_data)[0]

# 위험 단계 기준 설정 (실제 데이터의 분포 활용)
q1 = historical_df['Global_active_power'].quantile(0.4)
q2 = historical_df['Global_active_power'].quantile(0.8)

if predicted_power < q1:
    risk_level = "안전"
    advice = "현재 전력 사용량이 안정적인 수준입니다."
elif predicted_power < q2:
    risk_level = "주의"
    advice = "전력 소비량이 평소보다 다소 높은 시간대입니다. 불필요한 전력 소모를 줄여보세요!"
else:
    risk_level = "경고"
    advice = "🚨 **전력 피크 위험 구간입니다!** 과부하에 주의하세요."

# --- 5. 결과 화면 출력 ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="📊 실제 데이터 기반 예상 전력 소비량", value=f"{predicted_power:.2f} kW")

with col2:
    if risk_level == "안전":
        st.success(f"### 전력 피크 위험 단계: {risk_level}")
    elif risk_level == "주의":
        st.warning(f"### 전력 피크 위험 단계: {risk_level}")
    else:
        st.error(f"### 전력 피크 위험 단계: {risk_level}")

st.markdown("---")
st.info(f"💡 **에너지 절약 안내**: {advice}")
st.markdown("---")

# --- 6. 실제 데이터 기반 시각화 그래프 ---
st.subheader("📈 실제 데이터 기반 시간대별 평균 전력 소비 패턴")

hourly_avg = historical_df.groupby('Hour')['Global_active_power'].mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#1f77b4', linewidth=2, label='시간대별 평균 전력')
ax.axvline(x=input_hour, color='red', linestyle='--', label=f'선택한 시간 ({input_hour}시)')
ax.set_title("시간대별 전력 소비 추이 (실제 캐글 데이터)", fontsize=14, color='white')
ax.set_xlabel("시간 (Hour)", color='white')
ax.set_ylabel("전력 소비량 (kW)", color='white')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()

fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#1a1c23')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

st.pyplot(fig)

st.markdown("---")
st.caption("고등학교 데이터 과학 프로젝트 | 스마트 전력 가드 시스템 (Smart Power Guard)")
