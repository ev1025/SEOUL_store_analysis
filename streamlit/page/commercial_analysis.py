import streamlit as st
import pandas as pd
import numpy as np
import folium # type: ignore
import plotly.express as px
from streamlit_folium import st_folium # type: ignore
from folium.plugins import MarkerCluster # type: ignore
import os


def render_analysis():
    st.set_page_config(page_title="상권분석", layout="wide")
    
    """ 
    데이터 정의
    
    """
    data = st.session_state['data']                   # 202304분기 전체 데이터 (범주데이터 라벨 인코딩 되어 이씀)
    data_23 = data[data['기준_년분기_코드']==20234]
    ref_data = st.session_state['ref_data']           # 각 라벨 참조 테이블

    gu_list = st.session_state['gu_list']    # 행정구 리스트
    dong_list = st.session_state['dong_list']
    store_list = st.session_state['store_list']

    dong_dict = st.session_state['dong_dict']  # 행정구 : [행정동 리스트]   
    store_dict = st.session_state['store_dict'] # 행정동 : [상권 리스트]
    cat_dict = st.session_state['cat_dict'] # 상권 : [업종 리스트]

    # 아래 참조를 위한 초기값 None 지정
    gu_name = st.session_state['gu_name']
    dong_name = st.session_state['dong_name']



    """ 
    지도 생성 함수
    
    """
    def create_map(store_location = st.session_state.store_location,
                    selected_store=st.session_state.store_name, 
                    map_center=st.session_state.map_center, 
                    map_zoom=st.session_state.map_zoom, 
                    ):
        m = folium.Map(location=map_center, zoom_start=map_zoom)
        mc = MarkerCluster().add_to(m)

        # 전체 상권 마커
        for store, coords in store_location.items():
            folium.Marker(
                (coords['latitude'], coords['longitude']), 
                tooltip=f"상권 이름: {store}",
                icon=folium.Icon(color='blue', icon='star')
            ).add_to(mc)
        # 선택된 상권 마커
        if selected_store:
            location = store_location.get(selected_store)
            st.session_state.map_center = [location['latitude'], location['longitude']]
            st.session_state.map_zoom = 17
            folium.Marker(st.session_state.map_center, popup=None, tooltip=f"상권 이름: {selected_store}",
                        icon=folium.Icon(color='red', icon='star')).add_to(m)
            folium.Circle(st.session_state.map_center, popup=None, tooltip=None, 
                          radius=200, color = 'red', fill_color ='red').add_to(m)
            return m # 전체 상권 보여주는경우           
        return m # 전체 상권 보여주는경우


    
    """
    사이드바 영역
    
    """
    # 행정구 선택
    gu_name = st.sidebar.selectbox('행정구', gu_list, index=gu_list.index(st.session_state.gu_name) if st.session_state.gu_name else None, placeholder='구 선택')
    # 구가 바뀌면 
    if 'gu_name' in st.session_state and st.session_state.gu_name != gu_name:
        st.session_state.dong_name = None
        st.session_state.store_name = None
        st.session_state.cat_name = None
        st.session_state.gu_name = gu_name

    # 행정동 선택
    if gu_name:
        dong_name = st.sidebar.selectbox('행정동', dong_dict[gu_name], index=dong_dict[gu_name].index(st.session_state.dong_name) if st.session_state.dong_name else None, placeholder='동 선택')
        if 'dong_name' in st.session_state and st.session_state.dong_name != dong_name:
            st.session_state.store_name = None
            st.session_state.cat_name = None
            st.session_state.dong_name = dong_name

    # 상권 선택
        if dong_name:
            store_name = st.sidebar.selectbox('상권', store_dict[dong_name], index=store_dict[dong_name].index(st.session_state.store_name) if st.session_state.store_name else None, placeholder='상권 선택')
            if 'store_name' in st.session_state and st.session_state.store_name != store_name:
                st.session_state.cat_name = None
                st.session_state.store_name = store_name

    """
    상권분석 본문 영역

    """
    st.subheader("상권 분석 서비스")
    left_col, right_col = st.columns([1.7,1.3])

    # 지도 컬럼
    with left_col:
        
        with st.spinner(f"{len(st.session_state.store_list):,}개의 상권을 분석 중입니다. 잠시만 기다려주세요!"):
            st_folium(create_map(), height= 500, use_container_width=True)

    # 상세 정보 컬럼
    with right_col:      
        st.markdown(
            """
            <style>
                /* expander 제목 스타일 */
                .st-emotion-cache-89jlt8 e121c1cl0 {
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #333;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.store_name:
            '''
            상권 상세 정보
            
            '''
            bt_tab1, bt_tab2 = st.tabs(['상권 분석', '100일 생존 예측'])
            with bt_tab1:
                # store_name이 존재하는 경우에만 처리 
                selected_label = st.session_state.area_to_label.get(store_name, None)

                if selected_label is not None:
                    filtered_data = data_23[data_23["상권_라벨"] == selected_label]
                    ref_filtered = ref_data[["서비스_라벨", "서비스_업종_코드_명"]]
                    filtered_data = filtered_data.merge(ref_filtered, on="서비스_라벨", how="left").drop_duplicates()
                else:
                    filtered_data = None  # 기본값 설정

                # 평균 매출 칼럼 생성
                filtered_data['가게별 매출'] = round(filtered_data['당월_매출_금액'] / filtered_data['유사_업종_점포_수'])

                # 원본데이터의 문제로 200만 원 이하인 경우 평균 값이 아닌 당월 매출 금액을 사용
                filtered_data['평균 매출 (원)'] = np.where(filtered_data['가게별 매출'] <= 2000000,  # 200만원 이하인 경우
                                                        filtered_data['당월_매출_금액'],              # 당월_매출_금액 사용
                                                        filtered_data['가게별 매출']                  # 아닌 경우 가게별 매출 사용
                )

                # 칼럼 이름 변경
                filtered_data = filtered_data.rename(columns={"서비스_업종_코드_명":"업종명",
                                                                "유사_업종_점포_수":"유사 업종 수"
                                                                })

                # 상세 정보 옵션
                store_options = ["주변 정보", "상가 정보", "업종 정보"]
                selected_option = st.selectbox("🔍 확인할 정보를 선택하세요:", store_options)
                
                if selected_option == store_options[0]:
                    st.session_state.option1 = [f"🚇 인근 지하철 역 수 : {filtered_data['지하철_역_수'][0]:,} 개",
                                                f"🚏 인근 버스 정류장 수 : {filtered_data['버스_정거장_수'][0]:,} 개",
                                                f"🧑‍🤝‍🧑 인근 유동인구 수 : {filtered_data['총_유동인구_수'][0]:,} 명",
                                                f"🏠 인근 가구 수 : {filtered_data['총_가구_수'][0]:,} 가구",
                                                f"💰 인근 평균 월소득 : {filtered_data['월_평균_소득_금액'][0]:,}원"]
                    for i in st.session_state.option1:
                        st.write(i)

                if selected_option == store_options[1]:
                    st.session_state.option2 = [f"🏢 **개업율**: {filtered_data['개업_율'][0]}%",
                                                f"🚪 **폐업률**: {filtered_data['폐업_률'][0]}%",
                                                f"💰 **평당 평균 임대료**: {filtered_data['동별_임대료'][0]:,}원"]
                    for i in st.session_state.option2:
                        st.write(i)

                if selected_option == store_options[2]:
                    st.session_state.option3 = filtered_data[['업종명','유사 업종 수','평균 매출 (원)']].set_index('업종명')
                    st.dataframe(st.session_state.option3, use_container_width = True)


            '''
            폐업률 예측 구간
                
            '''
            with bt_tab2:
                # 업종 선택 버튼
                cat_name = st.selectbox('업종 선택 (지역을 먼저 설정해주세요.)', cat_dict[st.session_state['store_name']], index=cat_dict[st.session_state['store_name']].index(st.session_state.cat_name) if st.session_state.cat_name else None, placeholder='업종 선택')
                st.session_state.cat_name = cat_name
                
                # 서비스 라벨별 평균 매출 계산
                서비스별_평균_매출 = data.groupby('서비스_라벨')['당월_매출_금액'].mean()
                서비스별_평균_매출 = 서비스별_평균_매출.reset_index()

                # 상권 라벨별 평균 매출 계산
                상권별_평균_매출 = data.groupby('상권_라벨')['당월_매출_금액'].mean()
                상권별_평균_매출 = 상권별_평균_매출.reset_index()

                # 행정동 라벨별 평균 매출 계산
                행정동별_평균_매출 = data.groupby('행정동_라벨')['당월_매출_금액'].mean()
                행정동별_평균_매출 = 행정동별_평균_매출.reset_index()

                # 그룹별 평균 임대료 금액 계산
                행정동별_평균_임대료 = data.groupby('행정동_라벨')['동별_임대료'].mean()

                선택값 = {'서비스_라벨': ref_data[ref_data['서비스_업종_코드_명']==st.session_state.cat_name]['서비스_라벨'].unique(),
                    '상권_라벨' : ref_data[ref_data['상권_코드_명']==st.session_state.store_name]['상권_라벨'].unique(),
                    '행정동_라벨' :ref_data[ref_data['행정동_코드_명']==st.session_state.dong_name]['행정동_라벨'].unique()
                    }
                
                def calculate_survival_rate(폐업률, 개업률, 유사점포수, 매출, 유동인구, 상주인구, 임대료비율, 그룹, level="서비스_라벨"):
                    # 그룹별 가중치 선택
                    if level == "서비스_라벨":
                        weights_df = pd.read_csv("data/group_average_weights_서비스_라벨.csv", index_col=0)
                    elif level == "상권_라벨":
                        weights_df = pd.read_csv("data/group_average_weights_상권_라벨.csv", index_col=0)
                    elif level == "행정동_라벨":
                        weights_df = pd.read_csv("data/group_average_weights_행정동_라벨.csv", index_col=0)
                    else:
                        raise ValueError("올바른 그룹 기준을 입력하세요. (서비스_라벨 / 상권_라벨 / 행정동_라벨)")

                
                    weights_df.index = weights_df.index.astype(str)
                    그룹 = str(그룹)  # 그룹 변수를 문자열로 변환

                    if 그룹 not in weights_df.index:
                        print(f"그룹 '{그룹}'의 데이터가 없습니다. 기본 가중치를 사용합니다.")
                        weights = np.array([-0.3, 0.2, -0.15, 0.25, 0.1, -0.2, 0.15, 0.1])  # 기본 가중치
                    else:
                        weights = weights_df.loc[그룹].values
                    
                    if len(weights) < 7:
                        raise ValueError(f"가중치 개수가 부족합니다: {weights}")
                    
                    
                    if level == "서비스_라벨":
                        그룹별_평균_매출 = 서비스별_평균_매출.set_index("서비스_라벨")["당월_매출_금액"]
                    elif level == "상권_라벨":
                        그룹별_평균_매출 = 상권별_평균_매출.set_index("상권_라벨")["당월_매출_금액"]
                    else:
                        그룹별_평균_매출 = 행정동별_평균_매출.set_index("행정동_라벨")["당월_매출_금액"]

                    # 인덱스를 문자열로 변환 후 그룹별 평균 매출 가져오기
                    그룹별_평균_매출.index = 그룹별_평균_매출.index.astype(str)
                    그룹_평균_매출 = 그룹별_평균_매출.get(그룹, 1000000)
                    # st.write(f"그룹 '{그룹}'의 평균 매출: {그룹_평균_매출}")

                    # 매출비율 계산 
                    매출비율 = np.clip(매출 / 그룹_평균_매출, 0, 1)

                    # 행정동별 평균 임대료 가져오기
                    행정동별_평균_임대료.index = 행정동별_평균_임대료.index.astype(str)
                    그룹_평균_임대료 = 행정동별_평균_임대료.get(그룹, 1000000)
                    # st.write(f"그룹 '{그룹}'의 평균 임대료: {그룹_평균_임대료}")

                    # 임대료 비율 계산
                    임대료비율 = np.clip(임대료비율 / 그룹_평균_임대료, 0, 1)

                    # 생존 확률 계산
                    W1, W2, W3, W4, W5, W6, W8 = weights[:7]
                    S = (W1 * (1 - 폐업률) +
                        W2 * 개업률 +
                        W3 * min(유사점포수 / 100, 1) +
                        W4 * 매출비율 +        #(매출비율 / 1e9)
                        W5 * 유동인구 / 10000000 +
                        W6 * 상주인구 / 1000000 +
                        W8 * (1 - 임대료비율))
                    
                    S = np.clip(S, 0, 2)
                    survival_rate = np.clip(S * 50, 0, 100)
                    
                    return round(survival_rate, 2)


                # 행정동, 상권, 업종 종합
                def calculate_combined_survival_rate(선택값):
                    for level, 그룹 in 선택값.items():
                        if level == '서비스_라벨':
                            # 서비스 라벨 생존 확률 계산
                            service_info = data[data['서비스_라벨']==그룹[0]][['폐업_률', '개업_율', '유사_업종_점포_수','당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist()
                            service_survival_rate = round(calculate_survival_rate(*service_info, 그룹, level=level)/100,2)
                        elif level == '상권_라벨':
                            # 상권 라벨 생존 확률 계산
                            market_info = data[data['상권_라벨']==그룹[0]][['폐업_률', '개업_율', '유사_업종_점포_수','당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist()
                            market_survival_rate = round(calculate_survival_rate(*market_info, 그룹, level=level)/100,2)
                        elif level == '행정동_라벨':
                            # 행정동 라벨 생존 확률 계산
                            district_info = data[data['행정동_라벨']==그룹[0]][['폐업_률', '개업_율', '유사_업종_점포_수','당월_매출_금액', '총_유동인구_수', '총_상주인구_수', '동별_임대료']].mean().tolist()
                            district_survival_rate = round(calculate_survival_rate(*district_info, 그룹, level=level)/100,2)

                    # 각 라벨별 가중치
                    service_weight = 0.5
                    market_weight = 0.3
                    district_weight = 0.2
                    
                    combined_survival_rate = (service_weight * service_survival_rate +
                                            market_weight * market_survival_rate +
                                            district_weight * district_survival_rate)
                    
                    return combined_survival_rate, district_survival_rate,  market_survival_rate, service_survival_rate



                # 100일 생존 예측탭 버튼
                suv_col1, suv_col2 = st.columns(2)
                with suv_col1:
                    if st.button("생존 예측", use_container_width=True):
                        st.session_state.survive = True
                with suv_col2:
                    if st.button('선택 초기화', use_container_width=True):
                        st.session_state.gu_name = None
                        st.session_state.dong_name = None
                        st.session_state.store_name = None
                        st.session_state.cat_name = None
                        st.session_state.survive = None

                  

                def survive_predict():
                    all_suv, dong_suv, store_suv, cat_surv = calculate_combined_survival_rate(선택값)

                    col_per1, col_per2 = st.columns([2,1])
                    with col_per1:
                        st.write('')
                        st.progress(all_suv) 
                    with col_per2:
                        if all_suv >= 0.8:
                            st.markdown(
                                f"<div style='text-align:center;'><span style='color:#009874; font-size:24px; font-weight:bold;'>{all_suv:.2%}</span><br>"
                                f"<b>나의 생존 확률 😁 </div>",
                                unsafe_allow_html=True
                            )
                        elif all_suv > 0.4:
                            st.markdown(
                                f"<div style='text-align:center;'><span style='color:#FAC608; font-size:24px; font-weight:bold;'>{all_suv:.2%}</span><br>"
                                f"<b>나의 생존 확률 🤔 </div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div style='text-align:center;'><span style='color:#DB4455; font-size:24px; font-weight:bold;'>{all_suv:.2%}</span><br>"
                                f"<b>나의 생존 확률 😭 </div>",
                                unsafe_allow_html=True
                            )
                    # 선택 상권의 업종 매출 추이
                    selected_df = data[(data['서비스_라벨']==list(선택값.values())[0][0])&
                                       (data['상권_라벨']==list(선택값.values())[1][0])&
                                       (data['행정동_라벨']==list(선택값.values())[2][0])]

                    selected_df.loc[:, '기준_년분기_코드'] = selected_df.loc[:, '기준_년분기_코드'].astype(int).astype(str).apply(lambda x: f"{x[:4]}년 {x[4]}분기")
                    cat_df = selected_df.copy()
                    cat_sales = cat_df[['기준_년분기_코드','당월_매출_금액']].sort_values(by='기준_년분기_코드')
                    cat_sales['당월_매출_금액'] = cat_sales['당월_매출_금액'].clip(lower=1500000)
                    cat_sales = cat_sales.rename(columns={'당월_매출_금액':'매출','기준_년분기_코드':'분기'})
                    cat_sales['분류'] = f'{st.session_state.store_name}의 {st.session_state.cat_name} 매출'

                    # 해당 업종의 평균 매출 추이
                    cat_only_df = data[(data['서비스_라벨']==list(선택값.values())[0][0])]
                    cat_only_sales = cat_only_df.copy()
                    cat_only_sales = cat_only_sales[['기준_년분기_코드','당월_매출_금액']].sort_values(by='기준_년분기_코드')
                    cat_only_sales.loc[:, '기준_년분기_코드'] = cat_only_sales.loc[:, '기준_년분기_코드'].astype(int).astype(str).apply(lambda x: f"{x[:4]}년 {x[4]}분기")
                    cat_only_sales['당월_매출_금액'] = cat_only_sales['당월_매출_금액'].astype(int).clip(lower=1500000)
                    cat_avg_sales = cat_only_sales.groupby('기준_년분기_코드', as_index=False)['당월_매출_금액'].mean().round()
                    cat_avg_sales = cat_avg_sales.rename(columns={'당월_매출_금액':'매출','기준_년분기_코드':'분기'})
                    cat_avg_sales['분류'] = f'{st.session_state.cat_name} 평균 매출'

                    # 매출 종합
                    concat_graph = pd.concat([cat_sales,cat_avg_sales],axis=0)

                    etc_df = selected_df[['기준_년분기_코드','총_유동인구_수','총_상주인구_수','서울_운영_영업_개월_평균','서울_폐업_영업_개월_평균']].sort_values(by='기준_년분기_코드')
                    etc_df = etc_df.groupby('기준_년분기_코드', as_index=False)[['총_유동인구_수','총_상주인구_수','서울_운영_영업_개월_평균','서울_폐업_영업_개월_평균']].mean().round()
                    etc_df = etc_df.rename(columns={'기준_년분기_코드':'분기','총_상주인구_수':'상주인구', 
                                                    '총_유동인구_수':'유동인구', '서울_운영_영업_개월_평균':'평균 영업 개월 수',
                                                     '서울_폐업_영업_개월_평균':'평균 폐업 개월 수' })
                    pop_df = etc_df[['분기','유동인구','상주인구']]
                    other_df = etc_df[['분기', '평균 영업 개월 수', '평균 폐업 개월 수']]

                    st.write(' ')
                    with st.expander("##### 업종별 매출비교"):
                        st.line_chart(concat_graph, x='분기',y='매출', color='분류' )
                    with st.expander("##### 업종별 인구 변화"):
                        st.bar_chart(pop_df, x='분기', y=['유동인구', '상주인구'], stack=False)
                        st.dataframe(pop_df.set_index('분기'), use_container_width=True)
                    with st.expander("##### 업종별 기타 정보"):
                        st.bar_chart(other_df,x='분기', y=['평균 영업 개월 수', '평균 폐업 개월 수'], stack=False)
                        st.dataframe(other_df.set_index('분기'), use_container_width=True)
                    

                if st.session_state.survive:
                    survive_predict()
