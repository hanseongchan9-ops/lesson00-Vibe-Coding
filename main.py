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

# 전기전자공학 컨셉에 어울리는 스타일 및 위험 단계 색상 적용 CSS
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
st.markdown("### 가정용 전력 소비량 데이터 기반 전력 피크 예측 및 경고 시스템")
st.write("선택한 시간대, 월, 요일에 따른 맞춤형 전력 소비량 예측과 필터링된 패턴 그래프를 제공합니다.")

st.markdown("---")

# --- 3. 실제 데이터 로드 및 모델 학습 함수 ---
@st.cache_resource
def load_and_train_model():
    file_path = 'household_power_consumption.txt'
    
    # 실제 파일이 존재하는 경우 실제 데이터 사용
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, sep=';', low_memory=False)
            df.replace('?', np.nan, inplace=True)
            
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
            
            df.dropna(subset=['Global_active_power'], inplace=True)
            
            df['Hour'] = df['Datetime'].dt.hour
            df['DayOfWeek'] = df['Datetime'].dt.dayofweek
            df['Month'] = df['Datetime'].dt.month
            
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                
            X = df[['Hour', 'DayOfWeek', 'Month']]
            y = df['Global_active_power']
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            return model, df, "실제 데이터(household_power_consumption.txt)"
        except Exception as e:
            st.warning(f"실제 데이터를 읽어오는 중 문제가 발생하여 데모 데이터로 전환합니다. (오류내용: {e})")

    # 파일이 없거나 오류 발생 시 가상 데이터 생성
    np.random.seed(42)
    n_samples = 3000
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)
    months = np.random.randint(1, 13, n_samples)
    
    # 월과 요일에 따라 전력 소비량이 다르게 변하도록 시뮬레이션 반영
    power = 1.5 + 0.05 * hours + 0.1 * months + 0.2 * days + np.random.normal(0, 0.3, n_samples)
    power = np.clip(power, 0.5, 6.0)
    
    df = pd.DataFrame({
        'Hour': hours,
        'DayOfWeek': days,
        'Month': months,
        'Global_active_power': power
    })
    
    X = df[['Hour', 'DayOfWeek', 'Month']]
    y = df['Global_active_power']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, df, "데모(시뮬레이션) 데이터"

model, historical_df, data_source_name = load_and_train_model()
st.sidebar.caption(f"📂 현재 적용된 데이터 소스: **{data_source_name}**")

# --- 4. 사이드바: 사용자 입력 컨트롤러 ---
st.sidebar.header("🎛️ 예측 조건 설정")
st.sidebar.markdown("측정하고자 하는 시간대와 날짜 조건을 선택하세요.")

input_hour = st.sidebar.slider("시간대 (Hour)", 0, 23, 14)
input_month = st.sidebar.selectbox("월 (Month)", list(range(1, 13)), index=6)
input_day = st.sidebar.selectbox("요일", ["월", "화", "수", "목", "금", "토", "일"], index=0)

day_mapping = {"월":0, "화":1, "수":2, "목":3, "금":4, "토":5, "일":6}
input_day_num = day_mapping[input_day]

input_data = pd.DataFrame([[input_hour, input_day_num, input_month]], 
                          columns=['Hour', 'DayOfWeek', 'Month'])

# --- 5. AI 모델 예측 수행 ---
predicted_power = model.predict(input_data)[0]

# 위험 단계 기준 설정 (전체 데이터 분포 기반)
q1 = historical_df['Global_active_power'].quantile(0.4)
q2 = historical_df['Global_active_power'].quantile(0.8)

if predicted_power < q1:
    risk_level = "안전"
    advice = "현재 전력 사용량이 안정적인 수준입니다. 평소처럼 전기를 이용하세요."
elif predicted_power < q2:
    risk_level = "주의"
    advice = "전력 소비량이 평소보다 다소 높은 시간대입니다. 불필요한 전등이나 가전제품 대기전력을 신경 써주세요!"
else:
    risk_level = "경고"
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
st.info(f"💡 **에너지 절약 안내**: {advice}")
st.markdown("---")

# --- 7. 시각화 그래프 (선택한 월·요일 조건 필터링 반영) ---
st.subheader(f"📈 선택 조건({input_month}월, {input_day}요일)에 따른 시간대별 전력 소비 패턴")

# 사용자가 사이드바에서 선택한 '월'과 '요일'에 해당하는 데이터만 필터링
filtered_df = historical_df[
    (historical_df['Month'] == input_month) & 
    (historical_df['DayOfWeek'] == input_day_num)
]

# 만약 필터링된 데이터가 너무 적다면 전체 데이터 평균을 대체로 사용하거나 그대로 그룹화
if len(filtered_df) < 5:
    st.warning("선택하신 조건에 해당하는 데이터가 부족하여 전체 데이터 기준으로 패턴을 표시합니다.")
    plot_df = historical_df
else:
    plot_df = filtered_df

hourly_avg = plot_df.groupby('Hour')['Global_active_power'].mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#00d2ff', linewidth=2, label=f'{input_month}월 {input_day}요일 시간대별 평균')
ax.axvline(x=input_hour, color='red', linestyle='--', label=f'선택한 시간 ({input_hour}시)')
ax.set_title("조건별 시간대별 전력 소비 추이", fontsize=14, color='white')
ax.set_xlabel("시간 (Hour)", color='white')
ax.set_ylabel("전력 소비량 (kW)", color='white')

# Matplotlib 버전 호환 오류 방지 설정
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
