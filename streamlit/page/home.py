import streamlit as st
import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, Input # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from keras import layers # type: ignore

    
def render_home():
    # 데이터 정의
    data = st.session_state['data']   
    ref_data = st.session_state['ref_data']           # 각 라벨 참조 테이블

    gu_list = st.session_state['gu_list']    # 행정구 리스트
    dong_list = st.session_state['dong_list']  # 행정동 리스트
    store_list = st.session_state['store_list']  # 상권 리스트

    dong_dict = st.session_state['dong_dict']  # 행정구 : [행정동 리스트]   
    store_dict = st.session_state['store_dict'] # 행정동 : [상권 리스트]
    cat_dict = st.session_state['cat_dict'] # 행정동 : [업종 리스트]

    dong_name = st.session_state['dong_name']
    store_name = st.session_state['store_name']
    cat_name = st.session_state['cat_name']
    
    # with open('model/deepar_model_4.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # with open('model/scaler_X.pkl', 'rb') as scaler_X:
    #     scaler_X = pickle.load(scaler_X)
    # with open('model/scaler_y.pkl', 'rb') as scaler_y:
    #     scaler_y = pickle.load(scaler_y)
    
    # 모델과 스케일러를 세션 상태에 저장
    if 'model' not in st.session_state:
        with open('model/deepar_model_4.pkl', 'rb') as f:
            st.session_state.model = pickle.load(f)

    if 'scaler_X' not in st.session_state:
        with open('model/scaler_X.pkl', 'rb') as f:
            st.session_state.scaler_X = pickle.load(f)

    if 'scaler_y' not in st.session_state:
        with open('model/scaler_y.pkl', 'rb') as f:
            st.session_state.scaler_y = pickle.load(f)
            
    df = pd.read_csv('data/clustering_results.csv')

    st.markdown("""
        <div style="text-align: center;">
            <h4 style="margin-block-start: 0em; margin-block-end: 0em; color: #532533;">
                상권분석 데이터 전문가 NEW-MOON의 상권 분석 서비스</h4>
            <b>예비 창업자</b> 및 <b>기존 소상공인</b>이<br>
            자신의 상황에 맞는 최적의 입지를 찾고,<br>
            예상 매출과 매물 - 대출까지 한 눈에 비교할 수 있는 플랫폼입니다.
            <br><br>
        </div>
    """, unsafe_allow_html=True)

    
    left_b, middle_b, right_b = st.columns([1,1,1])
    
    with left_b:
        if st.button('📍상권 분석', use_container_width=True):
            st.switch_page(st.session_state['page_list'][1])
    with middle_b:
        if st.button('🏠 매물 추천', use_container_width=True):
            st.switch_page(st.session_state['page_list'][2])
    with right_b:
        if st.button('💸 대출 추천', use_container_width=True):
            st.switch_page(st.session_state['page_list'][3])

    # 매출 예측을 위한 expander 추가
    with st.expander("💰 내 가게의 매출은 얼마일까?", expanded=False):
        pop_col1, pop_col2 = st.columns([1, 1])
        with pop_col1:
            gu_name = st.selectbox('행정구', gu_list, index=None, placeholder='구 선택')
        with pop_col2:
            # 행정동 선택 박스
            if gu_name:
                dong_name = st.selectbox('행정동', dong_dict[gu_name], index=None, placeholder='동 선택')

        # 상권 선택 박스
        if dong_name:
            store_name = st.selectbox('상권', store_dict[dong_name], index=None, placeholder='상권 선택')
        if store_name:
            cat_name = st.selectbox('업종', cat_dict[store_name], index=None, placeholder='업종 선택')

        if cat_name:
            predict_store = ref_data[ref_data['상권_코드_명'] == store_name]['상권_라벨'].iloc[0]
            predict_service = ref_data[ref_data['서비스_업종_코드_명'] == cat_name]['서비스_라벨'].iloc[0]

            # # 예측 데이터 필터링
            # predict_data = df[(df['상권_라벨'] == predict_store) & 
            #                     (df['서비스_라벨'] == predict_service)]

            # 기준 년분기 코드를 datetime으로 변환하는 함수
            def convert_quarter_to_date(q_code):
                year = str(q_code)[:4]  # 년도 추출
                quarter = int(str(q_code)[4])  # 분기 추출
                month = (quarter - 1) * 3 + 1  # 분기를 월로 변환 (1, 4, 7, 10)
                return pd.Timestamp(year + '-' + str(month) + '-01')

            # 데이터프레임 df에서 '기준_년분기_코드' 열을 변환
            df['날짜'] = df['기준_년분기_코드'].apply(convert_quarter_to_date)

            # 날짜를 인덱스로 설정
            df.set_index('날짜', inplace=True)
            
            # 상권_라벨이 126인 행 제외
            df = df[df['상권_라벨'] != 126]
            
            # 99% 백분위값 계산 (서비스별)
            percentiles = df.groupby(['상권_라벨', '서비스_라벨'])['당월_매출_금액'].quantile(0.99).reset_index()
            percentiles.columns = ['상권_라벨', '서비스_라벨', '99백분위값']

            # 이상치 값 대체 (상권 262의 서비스 64번, 상권 1070의 서비스 59번)
            target_outliers = [(262, 64), (1070, 59)]

            for region, service in target_outliers:
                p99_value = percentiles[(percentiles['상권_라벨'] == region) &
                                        (percentiles['서비스_라벨'] == service)]['99백분위값'].values[0]

                mask = (df['상권_라벨'] == region) & (df['서비스_라벨'] == service)
                df.loc[mask & (df['당월_매출_금액'] > p99_value), '당월_매출_금액'] = p99_value

            # 업종별 평균 및 분산 계산
            grouped_stats = df.groupby('서비스_라벨')['당월_매출_금액'].agg(['mean', 'std']).reset_index()
            grouped_stats.columns = ['서비스_라벨', '평균_매출', '표준편차']

            # 정상 범위 설정
            grouped_stats['하한'] = grouped_stats['평균_매출'] - 2 * grouped_stats['표준편차']
            grouped_stats['하한'] = grouped_stats['하한'].clip(lower=0)  # 하한을 0으로 클리핑
            grouped_stats['상한'] = grouped_stats['평균_매출'] + 2 * grouped_stats['표준편차']

            # 제외할 서비스 라벨
            excluded_service_labels = [64, 59]

            # 이상치 제거
            for _, row in grouped_stats.iterrows():
                label = row['서비스_라벨']
                if label not in excluded_service_labels:
                    lower_bound = row['하한']
                    upper_bound = row['상한']
                    df = df[~((df['서비스_라벨'] == label) &
                            ((df['당월_매출_금액'] < lower_bound) |
                                (df['당월_매출_금액'] > upper_bound)))]
                    
            # 예측 데이터 필터링
            predict_data = df[(df['상권_라벨'] == predict_store) & 
                                (df['서비스_라벨'] == predict_service)] 
            
            # st.write("예측데이터:", predict_data)
            
            log_data = predict_data[['로그_당월_매출_금액']]
            n_steps = min(len(log_data) - 1, 10)  
            
            # st.write("로그데이터:", log_data)
            # st.write("로그데이터 길이:", len(log_data))
            
           # 데이터 길이가 n_steps보다 큰지 확인
            if len(log_data) < n_steps+1:
                st.write("데이터가 충분하지 않습니다. 예측을 수행할 수 없습니다.")
            else:
                # 데이터 준비 함수
                def create_dataset(data, time_step=1):
                    X, y = [], []
                    for i in range(len(data) - time_step):
                        X.append(data[i:(i + time_step)])
                        y.append(data[i + time_step])
                    return np.array(X), np.array(y)
                
                # 데이터셋 생성
                X, y = create_dataset(log_data.values, time_step=n_steps)
                
                # X의 shape 확인
                # st.write("X의 shape:", X.shape)
            
                all_features = df.values[n_steps:]
                
                # X 스케일링
                X = X.reshape(X.shape[0], X.shape[1], 1)  # (샘플 수, 타임 스텝, 1)

                # all_features를 스케일링
                all_features_scaled = st.session_state.scaler_X.transform(all_features)  # 스케일러를 통해 변환
                X = np.repeat(all_features_scaled[:, np.newaxis, :], n_steps, axis=1)
                
                #  y 스케일링
                predict_data_scaled = st.session_state.scaler_X.transform(predict_data)  # 스케일러를 통해 변환
                predict_data_scaled = np.tile(predict_data_scaled, (10, 1))  # 10배 확장
                predict_data_scaled = np.reshape(predict_data_scaled, (-1, 10, 182))  # 3D로 변환

                # 예측 수행
                y_pred_scaled = st.session_state.model.predict(predict_data_scaled)

                # y_pred_scaled가 2D 배열인지 확인 후 변환
                if y_pred_scaled.ndim == 1:
                    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

                # y_pred를 원래 스케일로 되돌리기
                y_pred = np.exp(st.session_state.scaler_y.inverse_transform(y_pred_scaled))

                # 결과 출력
                # st.write(f"업종별 예측된 매출 금액: **{y_pred.flatten()[0]:,.2f}** 원") 
                
                if not predict_data.empty and '점포_수' in predict_data.columns:
                    # "점포_수"의 평균 계산
                    average_store_count = predict_data['점포_수'].mean()

                    if average_store_count != 0: 
                        # 평균 예측된 매출 금액 계산
                        average_predicted_sales = y_pred.flatten()[0] / average_store_count

                        # 결과 출력
                        # st.write(f"업종별 평균 예측된 매출 금액: **{average_predicted_sales:,.2f}** 원")
                        st.write(f" ➡️ 업종별 예측된 평균 매출 금액: **{average_predicted_sales:,.0f}** 원")
                    else:
                        st.write("점포 수가 0 이어서 평균 매출을 구할 수 없습니다.")
                else:
                    st.write("데이터에 점포 수가 없어서 평균 매출은 구하지 못 합니다.")


    # 배경 이미지 적용 (CSS 활용)
    st.markdown(
        """
        <style>
        .stApp {
            background: url('img/main_img.jpg') no-repeat center center fixed;
            background-size: cover;
        }
        .custom-button {
            display: block;
            width: 47%;  /* 버튼 너비 조정 */
            padding: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background: #ffffff;
            border: 1px solid grey;
            border-radius: 10px;
            text-decoration: none;
            margin-top: 20px;
        }
        .custom-button:hover {
            background: #f1f1f1;
        }
        .feature-box {
            background: #f1f1f1;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .button-container {
            margin: 5%;
            display: flex;  /* Flexbox 사용하여 버튼을 양옆으로 배치 */
            justify-content: space-between;  /* 양쪽으로 정렬 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    bottom_left, bottom_right = st.columns([2, 2])

    # 왼쪽 섹션 (메인 문구 + 사용 방법)
    with bottom_left:
        # 사용 방법 푸터 (배경 색 적용)
        st.markdown(
            """
            <div class="feature-box">
                <h5>🛠 사용 방법</h5>
                <p>1. <b>간단한 정보 입력</b><br>- 상권, 업종, 필요 자금 등 기본 정보 입력</p>
                <p>2. <b>AI 기반 맞춤 추천</b><br>- 분석을 통한 예상 매출과 최적의 매물 제공 </p>
                <p>3. <b>입지 비교 및 상세 조회</b><br>- 상권, 예상 매출, 임대료 현황을 한눈에 비교 </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with bottom_right:
        # 주요 기능
        st.markdown("""
                <div class="feature-box">
                <h5>💡 주요 기능</h5>
                <p>1. <b>상권 분석</b><br>- AI 기반 데이터 분석으로 상권 파악</p>
                <p>2. <b>매출 예측</b><br>- 업종별 평균 매출과 예상 매출 제공</p>
                <p>3. <b>매물 추천</b><br>- 예산에 맞는 최적의 매물 추천</p>
                <p>4. <b>대출 추천</b><br>- 창업 자금 마련을 위한 맞춤형 금융 상품 제안</p>
            </div>
        """,     unsafe_allow_html=True)

