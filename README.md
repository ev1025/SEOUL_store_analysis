## 서울시 소상공인을 위한 상권분석 서비스

### 📅 프로젝트 기간
- 2025.02.25 ~ 2025.04.02 (약 37일)

### 👥 참여자 및 역할

| **😊 김경민 (팀장)**                                   | **😁 기솔아 (기획자)**                                 |
|----------------------------------------------------|----------------------------------------------------|
| - 데이터 분석 정의서 작성                          | - 프로젝트 기획서 작성                             |
| - 데이터 수집 (직방 웹사이트 크롤링)               | - 시장조사 및 사용자 리서치                        |
| - 데이터 전처리 (이상치 처리, 스케일링)            | - 화면 UI/UX 기획 (IA, 와이어프레임, 스트림릿)     |
| - Streamlit 페이지 제작 (매출 예측, 매물 추천)     |  - Streamlit 페이지 제작 (각 페이지 기초 화면)                                                  |
| - Machine Learning을 활용한 매출 예측              |    - 발표 자료 제작 보완 (PPT, 시연 영상)              |
| &nbsp;&nbsp;&nbsp;└ 회귀분석: `RandomForest`, `XGBOOST` |                   |
| &nbsp;&nbsp;&nbsp;└ 군집분석: `KNN`                 |                                                    |
| &nbsp;&nbsp;&nbsp;└ 시계열분석: `GRU`, `LSTM`, `DEEPAR+` |                                                    |


| **😘 이지홍 (데이터 분석)**                            | **🤔 이진우 (데이터 분석)**                            |
|----------------------------------------------------|----------------------------------------------------|
| - 프로젝트 기획서 작성                             | - 데이터 분석 정의서 작성                          |
| - 데이터 수집 (매출 예측 공공데이터)               | - 데이터 전처리 (축소 및 통합)                     |
| - Advanced RAG 구현 (`Rerank`, `MMR`)                     | - Fine-tuning (`GPT-4o-mini`, `Gemma3_12b`)            |
| - Streamlit 페이지 제작 (100일 생존 확률)&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;                            | - Streamlit 구조 설계 (navigation, session)        |
| - 발표 자료 제작 및 발표 (PPT)        | - Streamlit 페이지 제작 (Home, 상권분석) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          |


<br>

### 1. 프로젝트 배경 

&nbsp;  2024년 기준, 서울시 자영업 폐업률은 125%를 기록하며 역대 최고치를 갱신했습니다. 이는 창업보다 폐업이 더 많은, 극심한 창업 리스크 시대를 반영하는 수치입니다.
과포화된 업종 구조와 경기 침체로 인해 소상공인의 생존율이 급감하고 있으며, 특히 예비 창업자들은 경험 부족과 정보 비대칭으로 인해 창업 자체를 망설이고 있는 실정입니다.

> - '묻지마 창업'을 방지하기 위해 합리적인 의사결정 필요
> - 상권분석과 적정 임대료 및 매출 데이터 정보 필요
> - 사업 운영 시 발생할 수 있는 리스크 사전 인지 및 대비 가능
> - 개인의 사업 계획에 맞는 맞춤형 매물 추천으로 안정적인 자금 계획 수립 지원

<br>

### 2. 기대효과 
&nbsp; 본 프로젝트는,
상권 분석 → 예상 매출 및 임대료 예측 → 매물 추천까지,
창업의 전 과정을 데이터 기반으로 지원하는 통합 창업 의사결정 지원 서비스를 제안합니다.

<br>


### 3. 데이터 소개

&nbsp; 공공 및 민간에서 수집한 다양한 데이터를 활용하여 상권 분석, 매출 예측, 매물 추천 등의 기능을 구현하였습니다.

#### &nbsp; 1) 👫 인구 및 생활 기반 데이터 (**서울 열린데이터 광장**)
```
- 길 단위 인구 (19,794 rows × 27 columns)
- 상주 인구 (19,576 rows × 29 columns)
- 직장 인구 (19,476 rows × 26 columns)
- 아파트 정보 (17,623 rows × 20 columns)
- 집객 시설 수 (18,936 rows × 25 columns)
```

#### &nbsp; 2) 💳 상권 및 소비 데이터  (**서울 열린데이터 광장**)
```
- 매출 데이터 (249,331 rows × 55 columns)
- 소득 소비 데이터 (19,572 rows × 17 columns)
- 상권 변화 지표 (19,800 rows × 11 columns)
- 점포 데이터 (912,939 rows × 14 columns)
```
#### &nbsp; 3) 🏢 부동산/임대 관련 데이터  (**직방, 공공데이터포털**)
```
- 상가 매물 데이터 (134,411개)  
- 행정동별 임대료 (1,231 rows × 10 columns) 
```
#### &nbsp; 4) Streamlit 폴더 구조
```
📁 Streamlit
 ├─ app.py
 ├─ state.py
 │ 
 ├─ 📁 page
 │   ├─ commercial_analysis.py
 │   ├─ home.py
 │   ├─ loan.py
 │   └─ location.py
 │      
 ├─ 📁 model
 │   ├─ deepar_model_4.pkl
 │   ├─ scaler_X.pkl
 │   └─ scaler_y.pkl
 │ 
 ├─ 📁 chroma_db3
 │   ├─ chroma.sqlite3
 │   ├─ 📁 0d585fd0-b6af-4c08-af2b-a7719d61ce99
 │   └─ 📁 2781d1b3-d913-47b4-a651-05687acb1bbe
 │          
 ├─ 📁 data
 │   ├─ group_average_weights_상권_라벨.csv
 │   ├─ group_average_weights_서비스_라벨.csv
 │   ├─ group_average_weights_행정동_라벨.csv
 │   ├─ raw_data.csv
 │   └─ ref_data.csv
 │      
 ├─ 📁 img
 │   ├─ il_best.svg
 │   ├─ il_start.svg
 │   ├─ image.jpg
 │   ├─ llogo.png
 │   ├─ main_img.jpg
 │   ├─ newmoonlogo.png
 │   ├─ NEWMOON_LOGO.png
 │   ├─ slogo.png
 │   └─ 상권분석.png
 │      
 └─ 📁 util
     ├─ creata_map.py
     └─ popup.py
```


### 4. 데이터 분석
#### 1) 데이터 전처리

 &nbsp;&nbsp;&nbsp;[전처리 과정]
- Chunking: RecursiveCharacterTextSplitter로 문장 블록 분할
- Embedding: OpenAIEmbeddings를 통해 의미기반 벡터 생성
- DB 저장: ChromaDB로 검색 최적화된 벡터 저장 구조 구현


#### 2) 데이터 모델링
① 상권 매출 예측 모델   
a) 기능 : 사용자가 선택한 상권 및 업종 정보를 바탕으로, 해당 조건에 맞는 예상 매출을 예측      
b) 사용 모델   
- 회귀 분석: XGBoostClassifier, RandomForestClassifier
- 군집 분석: KNN
- 시계열 분석: GRU, LSTM, DeepAR+   

c) 특징
- 공공데이터 기반의 시계열 분석 모델을 활용해 정량적이고 신뢰도 높은 예측 가능
- 동 단위 상권의 소비 성향과 성장 흐름을 고려하여 파악한 서울시 각 상권별 트렌드를 이용하여 매출 예측에 차별화 적용
- 업종별 특성(예: 카페 vs 식당)을 고려하여 세분화된 매출 예측 제공
  
<img width="1155" alt="image" src="https://github.com/user-attachments/assets/3f50fab2-4f89-40b1-9901-2850a04e7eaa" />   


② 맞춤형 매물 추천 모델 (RAG 기반 LLM)      
a) 기능 : 사용자가 자연어로 입력한 조건(예: “월세 300만 원 이하, 카페 가능”)을 이해하고, 해당 조건에 맞는 최적 매물 리스트를 추천   
b) 사용 모델   
- LLM 모델: GPT-4o-mini, Gemma-3B-12B 파인튜닝 모델
- Retrieval 방식: RAG(Retrieval Augmented Generation) 기반
- 검색 최적화: MMR(Maximal Marginal Relevance) 기반 유사 매물 필터링
- 랭킹 개선: Cohere Rerank를 통해 의미 기반 재정렬 및 중복 제거
- 체인 구조: LangChain의 RetrievalQA Chain, ContextualCompressionRetriever 적용

c) 특징
- 다양한 질문 방식(“싸고 넓은 곳”, “사람 많은 곳 근처”)에도 대응하는 대화형 질의 응답 지원
- AI가 사용자 조건의 의도와 표현을 파악하여 매물을 정확하게 추천
- 중복 제거 및 유사도 기반 정렬로 사용자 만족도 높은 결과 제공

2. 맞춤형 매물 추천 (RAG 기반 LLM)


💛💛매물추천 지도가 깜빡거려서 ... 경민님 캡쳐 plz


특징
- 매물 추천에 특화된 맞춤형 대화형 AI로, 사용자의 질문 방식이나 표현이 다소 애매하거나 다양하더라도, 실제 창업 조건에 맞춰 정확하게 이해하고 응답
- 유사 매물 필터링 → 의미 유사도 기반 추천 → 중복 제거 및 최적 매물 재정렬

### 유틸리티 기능(수정중..⏳)

1. 100일 생존 확률
- 기능?
2. 상권분석
- 기능?


### 모델 평가 및 검증
 
