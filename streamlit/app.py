import streamlit as st
from importlib import import_module
from page import home, commercial_analysis, location, loan
from state import initialize_state


initialize_state()

st.logo(
    image="img/NEWMOON_LOGO (1).png",    
    icon_image="img/NEWMOON_LOGO (1).png",
)

home_navi = st.Page(home.render_home, title="홈", icon=":material/home:", default=False)
analysis_navi = st.Page(commercial_analysis.render_analysis, title="상권분석", icon=":material/store:", default=False)
location_navi = st.Page(location.render_location, title="매물추천", icon=":material/search:", default=False)
loan_navi = st.Page(loan.render_loan, title="대출추천", icon=":material/money:", default=False)

page_list = [home_navi, analysis_navi, location_navi, loan_navi]
st.session_state.page_list = page_list

page = st.navigation(page_list,expanded =False)
page.run()

if page != st.session_state.page:
    st.session_state.page = page
