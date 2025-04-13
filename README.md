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


### 4. 주요 기능
#### &nbsp;1) 데이터 전처리

 &emsp;[전처리 과정]
- Chunking: RecursiveCharacterTextSplitter로 문장 블록 분할
- Embedding: OpenAIEmbeddings를 통해 의미기반 벡터 생성
- DB 저장: ChromaDB로 검색 최적화된 벡터 저장 구조 구현


#### &nbsp; 2) 데이터 모델링
#### &emsp; ① 상권 매출 예측 모델  
&emsp;&emsp;a) 기능 : 사용자가 선택한 상권 및 업종 정보를 바탕으로, 해당 조건에 맞는 예상 매출을 예측      
&emsp;&emsp;b) 모델링    
| 항목     | 사용 모델                               |
|---------------|-------------------------------------------|
| 회귀 분석     | `XGBoostClassifier`, `RandomForestClassifier` |
| 군집 분석     | `KNN`                                      |
| 시계열 분석   | `GRU`, `LSTM`, `DeepAR+`                   |     
   

&emsp;&emsp;c) 특이사항   
> - 공공데이터 기반의 시계열 분석 모델을 활용해 정량적이고 신뢰도 높은 예측 가능   
> - 동 단위 상권의 소비 성향과 성장 흐름을 고려하여 파악한 서울시 각 상권별 트렌드를 이용하여 매출 예측에 차별화 적용   
> - 업종별 특성(예: 카페 vs 식당)을 고려하여 세분화된 매출 예측 제공   

&emsp;&emsp;d) 모델 검증     


<img width="1155" alt="image" src="https://github.com/user-attachments/assets/3f50fab2-4f89-40b1-9901-2850a04e7eaa" />   

<br>

#### &emsp; ② 맞춤형 매물 추천 모델   
&emsp;&emsp;a) 기능 : 유저의 자연어 입력(예: “월세 300만 원 이하, 쌍문동, 카페”)을 이해하고 조건에 맞는 매물 정보와 추천 이유 제공    

&emsp;&emsp;b) 모델링    
 | **항목** | **설명** |
 |------|------|
 | 파인튜닝(Fine-tuning) | `GPT-4o-mini`, `Gemma3_12b` 모델을 자체 Q&A 데이터(108개)로 미세조정 |
 | Retrieval-Augmented Generation | 크롤링한 13만 개 상가 데이터를 `ChromaDB`에 저장하여 검색 정확도 향상 |
 | 검색 최적화 | `MMR(Maximal Marginal Relevance)` 기반 유사 매물 필터링 |
 | 랭킹 개선 | `Cohere Rerank`를 통해 의미 기반 재정렬 및 중복 제거 |  
 | 출력 형식 개선 | `Few-shot` 기법으로 자연스럽고 일관된 답변 유도 |  

&emsp;&emsp;c) 특이사항      
> - 추천 이유가 일반적이거나 반복되는 문제 발생       
> - 프롬프트 내부에 "추천 이유는 \*\*매물 설명**을 기반으로 작성하세요."와 같은 명시적 지시 문장을 추가하여,    
> 모델이 특정 필드의 정보를 중점적으로 활용하도록 유도       
> - 그 결과, 추천 이유가 매물의 실제 특징(예: ‘넓은 실내’, ‘유동 인구 많은 위치’)을 반영하게 되어 설명력 향상
    
&emsp;&emsp;d) 모델 검증    

💛💛 매물추천 지도가 깜빡거려서 ... 경민님 캡쳐 plz    


### 5. 유틸리티 기능(수정중..⏳)
#### &nbsp;1) 데이터 전처리

 &emsp;[전처리 과정]
💛💛 지누 & 경민님 부탁해요

#### &emsp; ① 100일 생존 확률 분석
&emsp;&emsp;a) 기능 : 점포 생존 확률(매출 변동, 시각화) 예측 및 상권 경쟁력 분석

&emsp;&emsp;b) 모델링    

<img width="1393" alt="image" src="https://github.com/user-attachments/assets/f7a84907-e6b4-4b4d-93b7-5553f74d6d30" />

<img width="456" height="100" alt="image" src="https://github.com/user-attachments/assets/556f1cd6-678a-45a5-8d0a-c15897bf989b" />

<img width="456" alt="image" src="https://github.com/user-attachments/assets/60e8e3e5-6c38-4fd4-ab7d-3aa50ad0d52c" />

<img width="456" alt="image" src="https://github.com/user-attachments/assets/109605eb-f9b3-497f-bde6-a57453164779" />



#### &emsp; ② 상권분석  
&emsp;&emsp;a) 기능 : 유동인구, 평균 매출, 폐업률, 개업률, 평균 임대료, 업종별 임대료 정보 제공

&emsp;&emsp;b) 모델링   

<img width="1378" alt="image" src="https://github.com/user-attachments/assets/c20af0ea-b381-4b4d-8771-869ba7024faa" />
<img width="480" alt="image" src="https://github.com/user-attachments/assets/f0d828bf-c464-4f9a-8ba1-87e4cb7a6976" />
<img width="480" alt="image" src="https://github.com/user-attachments/assets/c3764298-8f29-434b-a52b-cc4804b24e8e" />


### 6. 프로젝트 소회

