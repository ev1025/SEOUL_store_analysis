import streamlit as st
import pandas as pd

def load_data():
    """
    데이터를 로드하고 필요한 전처리를 수행합니다.
    """
    data = pd.read_csv('data/clustering_results.csv', encoding='utf-8',)
    ref_data = pd.read_csv('data/ref_data.csv', encoding='utf-8-sig')
    return data, ref_data

def process_data(data, ref_data):
    """
    데이터를 처리하여 필요한 리스트와 딕셔너리를 생성합니다.
    """
    unique_list = data['상권_라벨'].unique()
    ref_data = ref_data[ref_data['상권_라벨'].isin(unique_list)]
    gu_list = ref_data['행정구_코드_명'].drop_duplicates().sort_values().tolist()
    dong_list = ref_data['행정동_코드_명'].drop_duplicates().tolist()
    store_list = ref_data['상권_코드_명'].drop_duplicates().tolist()
    dong_dict = {gu: sorted(ref_data[ref_data['행정구_코드_명'] == gu]['행정동_코드_명'].drop_duplicates()) for gu in gu_list}
    store_dict = {dong: sorted(ref_data[ref_data['행정동_코드_명'] == dong]['상권_코드_명'].drop_duplicates()) for dong in dong_list}
    cat_dict = {store: sorted(ref_data[ref_data['상권_코드_명'] == store]['서비스_업종_코드_명'].drop_duplicates()) for store in store_list}
    store_location = ref_data.loc[ref_data['상권_코드_명'].isin(store_list), 
                                ['상권_코드_명', 'latitude', 'longitude']].set_index('상권_코드_명').T.to_dict()
    return unique_list, ref_data, gu_list, dong_list, store_list, dong_dict, store_dict, cat_dict, store_location

def initialize_state():
    """
    초기 상태를 설정합니다.
    """
    if "data" not in st.session_state:
        data, ref_data = load_data()
        unique_list, ref_data, gu_list, dong_list, store_list, dong_dict, store_dict, cat_dict, store_location = process_data(data, ref_data)
        
        # area_to_label 딕셔너리 생성 및 저장
        area_to_label = dict(zip(ref_data["상권_코드_명"], ref_data["상권_라벨"]))
        
        st.session_state.update({
            "data": data,
            "ref_data": ref_data,
            "page": None,
            "gu_name" : None,
            "dong_name": None,
            "store_name": None,
            "cat_name" : None,
            "survive" : None,
            "unique_list": unique_list,
            "gu_list": gu_list,
            "dong_list": dong_list,
            "store_list": store_list,
            "dong_dict": dong_dict,
            "store_dict": store_dict,
            "cat_dict": cat_dict,
            "store_location": store_location,

            # 상권분석 state
            "map_center" : [37.5665, 126.978],
            "map_zoom" : 10.5,
            "area_to_label": area_to_label,  # area_to_label 딕셔너리 저장

            # 100일 생존 스테이트
            'map_loaded' : True,
            "highlighted_location" : None,
            "initialized" : True

        })