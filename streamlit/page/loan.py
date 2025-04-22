import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os



def render_loan(): 
    st.subheader("대출 추천은 준비 중입니다. 너른 양해 부탁드립니다. ^^")
    # 이미지 파일 경로
    image_path = 'img/image.jpg'  
    # 이미지 표시
    st.image(image_path, caption='WE WILL BE BACK')
    # data = st.session_state['data']   
    # ref_data = st.session_state['ref_data']           # 각 라벨 참조 테이블

    # gu_list = st.session_state['gu_list']    # 행정구 리스트
    # dong_list = st.session_state['dong_list']  # 행정동 리스트
    # store_list = st.session_state['store_list']  # 상권 리스트

    # dong_dict = st.session_state['dong_dict']  # 행정구 : [행정동 리스트]   
    # store_dict = st.session_state['store_dict'] # 행정동 : [상권 리스트]
    # cat_dict = st.session_state['cat_dict'] # 행정동 : [업종 리스트]

    # dong_name = st.session_state['dong_name']
    # store_name = st.session_state['store_name']
    # cat_name = st.session_state['cat_name']
    

    # # 서비스 라벨별 평균 매출 계산
    # 서비스별_평균_매출 = data.groupby('서비스_라벨')['당월_매출_금액'].mean()
    # 서비스별_평균_매출 = 서비스별_평균_매출.reset_index()

    # # 상권 라벨별 평균 매출 계산
    # 상권별_평균_매출 = data.groupby('상권_라벨')['당월_매출_금액'].mean()
    # 상권별_평균_매출 = 상권별_평균_매출.reset_index()

    # # 행정동 라벨별 평균 매출 계산
    # 행정동별_평균_매출 = data.groupby('행정동_라벨')['당월_매출_금액'].mean()
    # 행정동별_평균_매출 = 행정동별_평균_매출.reset_index()

    # # 그룹별 평균 임대료 금액 계산
    # 행정동별_평균_임대료 = data.groupby('행정동_라벨')['동별_임대료'].mean()
    # st.session_state.cat_name = 'PC방'
    # 선택값 = {'서비스_라벨': ref_data[ref_data['서비스_업종_코드_명']==st.session_state.cat_name]['서비스_라벨'].unique(),
    #        '상권_라벨' : ref_data[ref_data['상권_코드_명']==st.session_state.store_name]['상권_라벨'].unique(),
    #        '행정동_라벨' :ref_data[ref_data['행정동_코드_명']==st.session_state.dong_name]['행정동_라벨'].unique()
    #        }

    # def calculate_survival_rate(폐업률, 개업률, 유사점포수, 매출비율, 유동인구, 상주인구, 임대료비율, 그룹, level="서비스_라벨"):
    #     # 그룹별 가중치 선택
    #     if level == "서비스_라벨":
    #         weights_df = pd.read_csv("../data/group_average_weights_서비스_라벨.csv", index_col=0)
    #     elif level == "상권_라벨":
    #         weights_df = pd.read_csv("../data/group_average_weights_상권_라벨.csv", index_col=0)
    #     elif level == "행정동_라벨":
    #         weights_df = pd.read_csv("../data/group_average_weights_행정동_라벨.csv", index_col=0)
    #     else:
    #         raise ValueError("올바른 그룹 기준을 입력하세요. (서비스_라벨 / 상권_라벨 / 행정동_라벨)")

    
    #     weights_df.index = weights_df.index.astype(str)
    #     그룹 = str(그룹)  # 그룹 변수를 문자열로 변환

    #     if 그룹 not in weights_df.index:
    #         print(f"그룹 '{그룹}'의 데이터가 없습니다. 기본 가중치를 사용합니다.")
    #         weights = np.array([-0.3, 0.2, -0.15, 0.25, 0.1, -0.2, 0.15, 0.1])  # 기본 가중치
    #     else:
    #         weights = weights_df.loc[그룹].values
        
    #     if len(weights) < 7:
    #         raise ValueError(f"가중치 개수가 부족합니다: {weights}")
        
        
    #     if level == "서비스_라벨":
    #         그룹별_평균_매출 = 서비스별_평균_매출.set_index("서비스_라벨")["당월_매출_금액"]
    #     elif level == "상권_라벨":
    #         그룹별_평균_매출 = 상권별_평균_매출.set_index("상권_라벨")["당월_매출_금액"]
    #     else:
    #         그룹별_평균_매출 = 행정동별_평균_매출.set_index("행정동_라벨")["당월_매출_금액"]

    #     # 인덱스를 문자열로 변환 후 그룹별 평균 매출 가져오기
    #     그룹별_평균_매출.index = 그룹별_평균_매출.index.astype(str)
    #     그룹_평균_매출 = 그룹별_평균_매출.get(그룹, 1000000)
    #     print(f"그룹 '{그룹}'의 평균 매출: {그룹_평균_매출}")

    #     # 매출비율 계산 
    #     매출비율 = np.clip(매출비율 / 그룹_평균_매출, 0, 1)

    #     # 행정동별 평균 임대료 가져오기
    #     행정동별_평균_임대료.index = 행정동별_평균_임대료.index.astype(str)
    #     그룹_평균_임대료 = 행정동별_평균_임대료.get(그룹, 1000000)
    #     print(f"그룹 '{그룹}'의 평균 임대료: {그룹_평균_임대료}")

    #     # 임대료 비율 계산
    #     임대료비율 = np.clip(임대료비율 / 그룹_평균_임대료, 0, 1)

    #     # 생존 확률 계산
    #     W1, W2, W3, W4, W5, W6, W8 = weights[:7]
    #     S = (W1 * (1 - 폐업률) +
    #         W2 * 개업률 +
    #         W3 * min(유사점포수 / 100, 1) +
    #         W4 * 매출비율 +        #(매출비율 / 1e9)
    #         W5 * 유동인구 / 10000000 +
    #         W6 * 상주인구 / 1000000 +
    #         W8 * (1 - 임대료비율))
        
    #     S = np.clip(S, 0, 1)
    #     survival_rate = np.clip(S * 100, 0, 100)
        
    #     return round(survival_rate, 2)


    # # 행정동, 상권, 업종 종합
    # def calculate_combined_survival_rate(선택값):
    #     for level, 그룹 in 선택값.items():
    #         if level == '서비스_라벨':
    #             # 서비스 라벨 생존 확률 계산
    #             service_info = data[data['서비스_라벨']==그룹][['폐업_률', '개업_율', '유사_업종_점포_수','당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist()
    #             service_survival_rate = calculate_survival_rate(*service_info, 그룹, level=level)
    #         elif level == '상권_라벨':
    #             # 상권 라벨 생존 확률 계산
    #             market_info = data[data['상권_라벨']==그룹][['폐업_률', '개업_율', '유사_업종_점포_수','당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist()
    #             market_survival_rate = calculate_survival_rate(*market_info, 그룹, level=level)
    #         elif level == '행정동_라벨':
    #             # 행정동 라벨 생존 확률 계산
    #             district_info = data[data['행정동_라벨']==그룹][['폐업_률', '개업_율', '유사_업종_점포_수','당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist()
    #             district_survival_rate = calculate_survival_rate(*district_info, 그룹, level=level)

    #     # 각 라벨별 가중치
    #     service_weight = 0.5
    #     market_weight = 0.3
    #     district_weight = 0.2
        
    #     combined_survival_rate = (service_weight * service_survival_rate +
    #                               market_weight * market_survival_rate +
    #                               district_weight * district_survival_rate)
        
    #     print(f"최종 종합 생존 확률: {combined_survival_rate}%")
        
    #     return round(combined_survival_rate, 2)

    
    # st.write(calculate_survival_rate(*data[data['서비스_라벨'] == 1][['폐업_률', '개업_율', '유사_업종_점포_수', '당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist(), 1, level='서비스_라벨'))
    # st.write(st.session_state.cat_name)
    # st.write(calculate_combined_survival_rate(선택값))
    # # st.write(ref_data[ref_data['행정동_코드_명']==st.session_state.dong_name]['행정동_라벨'].unique())
