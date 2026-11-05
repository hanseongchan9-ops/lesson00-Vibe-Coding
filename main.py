import streamlit as st

st.set_page_config(
    page_title="Smart Power Guard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Power Guard - 가정용 전력 관리 시스템")
st.markdown("### 고등학교 데이터 과학 수행평가 프로젝트")
st.write("좌측 사이드바의 **페이지 메뉴**를 통해 기능을 선택해 주세요.")

st.info("""
- **1_전력_예측_및_피크**: 머신러닝을 활용한 시간대별 전력 소비량 예측 및 피크 경고
- **2_구역별_세부조회**: X월 XX일 날짜 선택 및 주방/세탁/에어컨 등 구역별 세부 전력 사용량 분석
""")
