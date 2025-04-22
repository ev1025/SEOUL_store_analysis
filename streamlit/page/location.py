import streamlit as st
import folium  # type: ignore
from streamlit_folium import st_folium  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore
from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore
from langchain.chat_models import ChatOpenAI  # type: ignore
from langchain.chains import RetrievalQA  # type: ignore
import pandas as pd
import traceback
from chromadb import PersistentClient  # type: ignore # ChromaDB 클라이언트
import openai # type: ignore
import random

# OpenAI 임베딩 모델 초기화
embedding_function = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]) 
vector_db = Chroma(persist_directory="chroma_db3", embedding_function=embedding_function)
retriever = vector_db.as_retriever()

# ChromaDB 클라이언트 및 컬렉션 초기화
persist_directory = "chroma_db3"
chroma_client = PersistentClient(path=persist_directory)
collection = chroma_client.get_collection(name="my_collection")

# chat_model = ChatOpenAI(model_name = 'gpt-4o',
#                         api_key=st.secrets["OPENAI_API_KEY"],           
#                         temperature=0.5
#                         )

ft_model = ChatOpenAI(model_name="ft:gpt-4o-mini-2024-07-18:fininsight::BD0dE9e2", api_key=st.secrets["OPENAI_API_KEY"], temperature=0.5)

# RetrievalQA 체인 생성
R_QA1 = RetrievalQA.from_chain_type(llm=ft_model, chain_type="stuff", retriever=retriever, return_source_documents=True)

# MMR검색기 지정
# mmr_retriever = vector_db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "fetch_k": 100,   # 후보 문서 개수를 줄여 정보 과부하 방지
#         "k": 10,         # 최종 선택 문서 개수를 줄여 정밀도 개선
#         "lambda_mult": 0.5  # 유사도를 더 우선시하도록 조정
#     }
# )

# R_QA2 = RetrievalQA.from_chain_type(llm = ft_model, chain_type = "stuff", retriever = mmr_retriever, return_source_documents = True)


# OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_reason(metadata):
    prompt = f"""
    당신은 부동산 전문가입니다.
    이 매물에 대한 추천 이유를 작성해 주세요. 
    다음 정보를 기반으로 작성해 주세요:
    - 위치: {metadata['location']}
    - 보증금: {metadata['deposit']}
    - 월세: {metadata['rent']}
    - 권리금: {metadata['key_money']}
    - 면적: {metadata['area']}m²
    - 층수: {metadata['floor']}
    - 설명: {metadata['description']}
    
    특히, 추천한 매물에 대해서 다른 정보 보다 **설명** 부분을 기반으로 추천 이유를 줄글 형식으로 자세하게 생성해주세요.

    """
    
    # 추천 이유 생성
    result = R_QA1({"query": prompt})
    
    recommendation_reason = result['result']
    sources = result.get('source_documents', [])
    
    return recommendation_reason, sources

    
# 엑셀 파일 경로
csv_file_path = "data/ref_data.csv"

# 엑셀 파일에서 데이터 읽기
df = pd.read_csv(csv_file_path)

def render_location():
    #st.title("상가 추천 요정🧚")
    st.subheader("매물 추천 서비스")
    st.markdown("안녕하세요, 원하는 상권을 선택하면 위치를 보여주고, <br> **지역구 위치와 예산 조건**에 맞는 매물을 추천해주는 페이지입니다!", unsafe_allow_html=True)
    
    # 지도 관련 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'map_ready' not in st.session_state:
        st.session_state.map_ready = False
    if 'map' not in st.session_state:
        st.session_state.map = None
    if 'filtered_results' not in st.session_state:
        st.session_state.filtered_results = []
    

    # 사이드바에 셀렉트 박스 추가
    st.sidebar.header("검색 조건")
    district_options = df['행정구_코드_명'].unique().tolist()
    selected_district = st.sidebar.selectbox("지역구를 선택하세요:", district_options)

    # 선택된 자치구에 해당하는 행정동 필터링
    adstrd_options = df[df['행정구_코드_명'] == selected_district]['행정동_코드_명'].unique().tolist()
    selected_adstrd = st.sidebar.selectbox("행정동을 선택하세요:", adstrd_options)

    # 선택된 행정동에 해당하는 상권 필터링
    trdar_options = df[df['행정동_코드_명'] == selected_adstrd]['상권_코드_명'].unique().tolist()
    selected_trdar = st.sidebar.selectbox("상권을 선택하세요:", trdar_options)
    
    # 보증금 입력 박스 표시 
    deposit_budget = st.sidebar.number_input("보증금 조건을 입력하세요 (만원 단위):", min_value=0)
    
    # 월세 입력 박스 표시
    rent_budget = st.sidebar.number_input("월세 조건을 입력하세요 (만원 단위):", min_value=0)
    
    
    # 매물 검색 버튼
    if st.sidebar.button("매물 검색"):
        # 매물 검색 버튼 클릭 시 상태 초기화
        st.session_state.filtered_results = []  # 이전 결과 초기화

        try:
            # 선택한 상권에 맞는 위도와 경도 찾기
            location_info = df[df['상권_코드_명'] == selected_trdar]
            if not location_info.empty:
                lat = location_info.iloc[0]['latitude']
                lon = location_info.iloc[0]['longitude']

                # 지도 생성
                m = folium.Map(location=[lat, lon], zoom_start=15)
                folium.Marker(
                    [lat, lon],
                    popup=f"추천 매물 위치: {selected_district} - {selected_adstrd}",
                    icon=folium.Icon(color='red', icon='home', prefix='fa') # (color='blue', icon='info-sign')
                ).add_to(m)

                # Streamlit에 지도 표시
                st.session_state.map_ready = True
                st.session_state.map = m
        except:
            pass
            
    # 지도 표시
    if st.session_state.map_ready and st.session_state.map is not None:
        st_folium(st.session_state.map, width=900, height=500)      
   
        try:
            # 쿼리 실행
            query_text = "*" # 원하는 쿼리 텍스트
            results = collection.query(
                query_texts=[query_text],
                n_results=30000
            )

            filtered_results = []
            alternative_results = []    
            budget_only_results = []  # 예산만 맞는 매물 리스트
            deposit_results = []
            seen_ids = set()  # 중복 체크를 위한 집합

            # 문서와 메타데이터를 동시에 순회
            for i in range(len(results['ids'][0])):  # 첫 번째 차원 길이
                doc_id = results['ids'][0][i]
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]

                # 메타데이터에서 보증금 값 추출
                deposit = int(metadata['deposit'].replace('만원', '').replace(',', '').strip())
                location_meta = metadata.get('location', '')
                
                # 메타데이터에서 월세 값 추출
                rent = int(metadata['rent'].replace('만원', '').replace(',', '').strip())
                location_meta = metadata.get('location', '')
                
                 # 중복 체크
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                # 모든 조건을 만족하는 매물 필터링
                if (rent <= rent_budget) and (selected_district in location_meta) and (deposit <= deposit_budget ):
                    filtered_results.append((doc_content, metadata))
                # 예산 (월세+보증금) 조건을 만족하는 매물
                elif (rent <= rent_budget) and (deposit <= deposit_budget):
                    budget_only_results.append((doc_content, metadata))
                # 위치 조건을 만족하는 매물
                elif (selected_district in location_meta):
                    alternative_results.append((doc_content, metadata))
                # 보증금 조건만 만족하는 매물
                # elif (deposit <= deposit_budget):
                #    deposit_results.append((doc_content, metadata))
                        
            # 세션 상태에 결과 저장
            st.session_state.filtered_results = filtered_results

            # 조건만족 매물 출력
            # with st.container():
            with st.expander("💯**조건을 만족하는 매물:**", expanded=True):
                if filtered_results:
                    for doc_content, metadata in filtered_results[:4]:
                        item_id = metadata['id']
                        url = f"https://www.zigbang.com/home/store/items/{item_id}"

                        # 추천 이유 생성
                        recommendation_reason, sources = generate_reason(metadata)

                        # 매물 정보 포맷 출력
                        st.markdown(f"""
                        🏠 **매물 Id:** {item_id}<br>
                        **위치:** {metadata['location']}<br>
                        **보증금:** {metadata['deposit']}<br>
                        **월세:** {metadata['rent']}<br>
                        **권리금:** {metadata['key_money']}<br>
                        **면적:** {metadata['area']}m²<br>
                        **층수:** {metadata['floor']}<br>
                        **추천 이유:** {recommendation_reason}<br>
                        
                        🔗 매물 보러가기: [클릭]({url})<br>
                        """, unsafe_allow_html=True)
                        st.write("---")
                else:
                    st.write("예산 내 매물이 없습니다.")

            # 위치 조건 맞는 매물 출력
            if alternative_results:
                # with st.container():
                with st.expander("✨ **마음에 드는 매물이 없으신가요? 대신 선택한 위치에 해당하는 매물을 보여드릴게요!**", expanded=True):
                    selected_alternative_results = random.sample(alternative_results, min(4, len(alternative_results)))
                    for doc_content, metadata in selected_alternative_results:
                    # for doc_content, metadata in alternative_results[:4]:
                        item_id = metadata['id']
                        url = f"https://www.zigbang.com/home/store/items/{item_id}"

                        # 추천 이유 생성
                        recommendation_reason, sources = generate_reason(metadata)
        
                        st.markdown(f"""
                        🏠 **매물 Id:** {item_id}<br>
                        **위치:** {metadata['location']}<br>
                        **보증금:** {metadata['deposit']}<br>
                        **월세:** {metadata['rent']}<br>
                        **권리금:** {metadata['key_money']}<br>
                        **면적:** {metadata['area']}m²<br>
                        **층수:** {metadata['floor']}<br>
                        **추천 이유:** {recommendation_reason}<br>
                        
                        🔗 매물 보러가기: [클릭]({url})<br>
                        """, unsafe_allow_html=True)
                        st.write("---")
            # # 보증금만 맞는 매물 출력
            # if deposit_results:
            #     st.write("✨ 마음에 드는 매물이 없으신가요? 대신 보증금은 만족하는 매물도 있어요!")
            #     selected_deposit_results = random.sample(deposit_results, min(3, len(deposit_results)))
            #     for doc_content, metadata in selected_deposit_results:
            #         item_id = metadata['id']
            #         url = f"https://www.zigbang.com/home/store/items/{item_id}"

            #         # 추천 이유 생성
            #         recommendation_reason, sources = generate_reason(metadata)

            #         st.markdown(f"""
            #         🏠 **매물 Id:** {item_id}<br>
            #         **위치:** {metadata['location']}<br>
            #         **보증금:** {metadata['deposit']}<br>
            #         **월세:** {metadata['rent']}<br>
            #         **권리금:** {metadata['key_money']}<br>
            #         **면적:** {metadata['area']}m²<br>
            #         **층수:** {metadata['floor']}<br>
            #         **추천 이유:** {recommendation_reason}<br>
                    
            #         🔗 매물 보러가기: [클릭]({url})<br>
            #         """, unsafe_allow_html=True)
            #         st.write("---")

            # 예산 조건(월세+보증금)만 맞는 매물 출력
            if budget_only_results:
                # with st.container():
                with st.expander("✨ **마음에 드는 매물이 없으신가요? 대신 예산을 만족하는 매물도 있어요!**", expanded=True):
                    selected_budget_only_results = random.sample(budget_only_results, min(3, len(budget_only_results)))
                    for doc_content, metadata in selected_budget_only_results:
                    # for doc_content, metadata in budget_only_results[:4]:
                        item_id = metadata['id']
                        url = f"https://www.zigbang.com/home/store/items/{item_id}"

                        # 추천 이유 생성
                        recommendation_reason, sources = generate_reason(metadata)

                        st.markdown(f"""
                        🏠 **매물 Id:** {item_id}<br>
                        **위치:** {metadata['location']}<br>
                        **보증금:** {metadata['deposit']}<br>
                        **월세:** {metadata['rent']}<br>
                        **권리금:** {metadata['key_money']}<br>
                        **면적:** {metadata['area']}m²<br>
                        **층수:** {metadata['floor']}<br>
                        **추천 이유:** {recommendation_reason}<br>
                        
                        🔗 매물 보러가기: [클릭]({url})<br>
                        """, unsafe_allow_html=True)
                        st.write("---")
            
        except Exception as e:
            st.write(f"오류 발생: {e}")
            