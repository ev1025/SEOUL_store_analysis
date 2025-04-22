import streamlit as st
import numpy as np

def popup(store_name, ref_data, data):
    """상권 정보를 팝업으로 표시합니다."""
    area_to_label = dict(zip(ref_data["상권_코드_명"], ref_data["상권_라벨"]))
    selected_label = area_to_label[store_name]

    filtered_data = (
        data.query("상권_라벨 == @selected_label")
        .merge(ref_data[["서비스_라벨", "서비스_업종_코드_명"]], on="서비스_라벨", how="left")
        .drop_duplicates()
    )

    filtered_data['가게별 매출'] = round(filtered_data['당월_매출_금액'] / filtered_data['유사_업종_점포_수'])
    filtered_data['평균 매출 (원)'] = np.where(filtered_data['가게별 매출'] <= 2_000_000,
                                            filtered_data['당월_매출_금액'],
                                            filtered_data['가게별 매출'])

    filtered_data = filtered_data.rename(columns={"서비스_업종_코드_명": "업종명",
                                                "유사_업종_점포_수": "유사 업종 수"})

    store_options = ["주변 정보", "상가 정보", "업종 정보"]
    selected_store = st.selectbox(" 확인할 정보를 선택하세요:", store_options)

    if selected_store == store_options[0]:
        st.write(f" 인근 지하철 역 수 : {filtered_data['지하철_역_수'].iloc[0]} 개")
        st.write(f" 인근 버스 정류장 수 : {filtered_data['버스_정거장_수'].iloc[0]} 개")
        st.write(f"‍‍ 인근 유동인구 수 : {filtered_data['총_유동인구_수'].iloc[0]} 명")
        st.write(f" 인근 가구 수 : {filtered_data['총_가구_수'].iloc[0]} 가구")
        st.write(f" 인근 평균 월소득 : {filtered_data['월_평균_소득_금액'].iloc[0]:,}원")

    if selected_store == store_options[1]:
        st.write(f" **개업율**: {filtered_data['개업_율'].iloc[0]}%")
        st.write(f" **폐업율**: {filtered_data['폐업_률'].iloc[0]}%")
        st.write(f" **평당 평균 임대료**: {filtered_data['동별_임대료'].iloc[0]:,}원")

    if selected_store == store_options[2]:
        st.dataframe(filtered_data[['업종명', '유사 업종 수', '평균 매출 (원)']].set_index('업종명'), use_container_width=True)