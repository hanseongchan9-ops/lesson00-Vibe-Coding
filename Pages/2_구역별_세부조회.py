import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("🏠 X월 XX일 가전제품 구역별 전력 소비량 조회")
st.write("원하시는 날짜를 선택하여 주방, 세탁, 기후제어(에어컨·온수기) 구역별 전력 사용 패턴을 확인하세요.")

@st.cache_data
def load_data():
    file_path = 'household_power_consumption.txt'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['?'])
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df['Date_only'] = df['Datetime'].dt.date
        df['Month'] = df['Datetime'].dt.month
        df['Day'] = df['Datetime'].dt.day
        df['Hour'] = df['Datetime'].dt.hour
        
        # 숫자형 변환
        for col in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    return None

df = load_data()

if df is None:
    st.error("⚠️ 'household_power_consumption.txt' 파일이 없습니다. 깃허브에 파일을 업로드해 주세요!")
else:
    # --- 사이드바 날짜 선택 컨트롤러 ---
    st.sidebar.header("📅 날짜 선택")
    
    # 데이터에 존재하는 월 선택
    available_months = sorted(df['Month'].unique())
    selected_month = st.sidebar.selectbox("월 선택", available_months)
    
    # 해당 월에 속한 일(Day) 선택
    available_days = sorted(df[df['Month'] == selected_month]['Day'].unique())
    selected_day = st.sidebar.selectbox("일(Day) 선택", available_days)
    
    # 선택된 날짜 데이터 필터링
    target_data = df[(df['Month'] == selected_month) & (df['Day'] == selected_day)]
    
    st.markdown(f"### 📌 선택한 날짜: **{selected_month}월 {selected_day}일** 전력 사용 현황")
    
    if target_data.empty:
        st.warning("선택하신 날짜에 해당하는 데이터가 없습니다.")
    else:
        # 구역별 총 사용량 혹은 평균 지표 카드 출력
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🍳 주방 구역 (Sub 1)", f"{target_data['Sub_metering_1'].sum():,.1f} Wh")
            st.caption("식기세척기, 오븐, 전자레인지 등")
        with col2:
            st.metric("🧺 세탁 구역 (Sub 2)", f"{target_data['Sub_metering_2'].sum():,.1f} Wh")
            st.caption("세탁기, 냉장고 등")
        with col3:
            st.metric("❄️ 기후제어/온수 (Sub 3)", f"{target_data['Sub_metering_3'].sum():,.1f} Wh")
            st.caption("에어컨, 전기 온수기 등")
            
        st.markdown("---")
        
        # 시간대별 구역별 소비 추이 시각화 그래프
        st.subheader(f"📈 {selected_month}월 {selected_day}일 시간대별 구역별 전력 소비 추이")
        
        hourly_sub = target_data.groupby('Hour')[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hourly_sub.index, hourly_sub['Sub_metering_1'], label='주방 (Sub 1)', color='#ff9999', linewidth=2)
        ax.plot(hourly_sub.index, hourly_sub['Sub_metering_2'], label='세탁 (Sub 2)', color='#66b3ff', linewidth=2)
        ax.plot(hourly_sub.index, hourly_sub['Sub_metering_3'], label='기후제어/온수 (Sub 3)', color='#99ff99', linewidth=2)
        
        ax.set_title(f"{selected_month}월 {selected_day}일 가전 구역별 전력 변화", fontsize=14, color='white')
        ax.set_xlabel("시간 (Hour)", color='white')
        ax.set_ylabel("전력 소비량 (Wh)", color='white')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        # 다크모드 디자인 맞춤
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1c23')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        
        st.pyplot(fig)
