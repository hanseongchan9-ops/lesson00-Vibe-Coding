import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

st.title("⚡ AI 기반 전력 소비량 예측 및 피크 경고")

@st.cache_resource
def load_and_train_model():
    file_path = 'household_power_consumption.txt'
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['?'])
            df.dropna(subset=['Global_active_power'], inplace=True)
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df['Hour'] = df['Datetime'].dt.hour
            df['DayOfWeek'] = df['Datetime'].dt.dayofweek
            df['Month'] = df['Datetime'].dt.month
            df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])
            
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                
            X = df[['Hour', 'DayOfWeek', 'Month']]
            y = df['Global_active_power']
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            return model, df, "실제 UCI 전력 데이터"
        except:
            pass

    # 파일 없을 경우 데모 데이터
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame({
        'Hour': np.random.randint(0, 24, n),
        'DayOfWeek': np.random.randint(0, 7, n),
        'Month': np.random.randint(1, 13, n),
        'Global_active_power': np.random.uniform(0.5, 5.0, n)
    })
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(df[['Hour', 'DayOfWeek', 'Month']], df['Global_active_power'])
    return model, df, "데모 데이터"

model, historical_df, source_name = load_and_train_model()
st.sidebar.caption(f"📂 데이터 소스: **{source_name}**")

st.sidebar.header("🎛️ 예측 조건 설정")
input_hour = st.sidebar.slider("시간대 (Hour)", 0, 23, 14)
input_month = st.sidebar.selectbox("월 (Month)", list(range(1, 13)), index=6)
input_day = st.sidebar.selectbox("요일", ["월", "화", "수", "목", "금", "토", "일"], index=0)

day_map = {"월":0, "화":1, "수":2, "목":3, "금":4, "토":5, "일":6}
input_data = pd.DataFrame([[input_hour, day_map[input_day], input_month]], columns=['Hour', 'DayOfWeek', 'Month'])

predicted_power = model.predict(input_data)[0]

q1 = historical_df['Global_active_power'].quantile(0.4)
q2 = historical_df['Global_active_power'].quantile(0.8)

if predicted_power < q1:
    risk_level, advice = "안전", "전력 사용량이 안정적입니다."
elif predicted_power < q2:
    risk_level, advice = "주의", "평소보다 전력 소비가 다소 높은 시간대입니다."
else:
    risk_level, advice = "경고", "🚨 전력 피크 위험 구간입니다! 과부하에 주의하세요."

col1, col2 = st.columns(2)
with col1:
    st.metric(label="📊 예상 전력 소비량", value=f"{predicted_power:.2f} kW")
with col2:
    if risk_level == "안전": st.success(f"위험 단계: {risk_level}")
    elif risk_level == "주의": st.warning(f"위험 단계: {risk_level}")
    else: st.error(f"위험 단계: {risk_level}")

st.info(f"💡 안내: {advice}")
