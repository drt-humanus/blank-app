import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import BytesIO
import pickle
import urllib.request
from numpy.random import default_rng as rng

if 'load' not in st.session_state:
    st.session_state['load'] = 0

def load_page():
    st.write("""
    # [AIAM] AI IN PREDICTION OF MORTILITY IN ACUTE MYOCARDIAL INFARCTION
    (Ứng dụng AI tiên lượng tử vong trong nhồi máu cơ tim cấp)
    """)
    st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
@st.cache_data(suppress_st_warning=True)
def get_pickle_data():
    with open("/workspaces/blank-app/.github/deployed_rfmodel_pca10_nor_smote.sav", "rb") as download:
        return pickle.load(download)

def calculate_risk():
    load_page()
    show_user_input(user_input)
    rf_model = get_pickle_data()
    # Store the models prediction in a variable
    prediction = rf_model.predict_proba(user_input)
    # Set a subheader and display the classification
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Tiên lượng tử vong của nhồi máu cơ tim là')
    with col4:
        if prediction[0,1]<0.25:
            st.markdown("<h1 style='color: green;font-weight: bold;'>" + ' ' + str(f'{prediction[0,1]*100:5.2f}') + ' ' + '%' +"</h1>", unsafe_allow_html=True)
        if prediction[0,1]>0.25 and prediction[0,1]<0.75:
            st.markdown("<h1 style='color: black;font-weight: bold;'>" + ' ' + str(f'{prediction[0,1]*100:5.2f}') + ' ' + '%' +"</h1>", unsafe_allow_html=True)
        if prediction[0,1]>0.75:
            st.markdown("<h1 style='color: red;font-weight: bold;'>" + ' ' + str(f'{prediction[0,1]*100:5.2f}') + ' ' + '%' +"</h1>", unsafe_allow_html=True)
    st.write('Mô hình Gradient Boosting' + ' (' +  'Exponential, Friedmen MSE, sqrt)')
    st.write('Độ chính xác của chẩn đoán: 89.5%' + ' ' + '('+  'AUC: 0.75)')
    st.session_state['load'] = 1

# value "giá trị xuất hiện khi khởi động chương trình", step: đơn vị nhỏ nhất của chỉ số định lượng (interger/float)
def show_user_input(user_input):
        df = pd.DataFrame(user_input)
        st.dataframe(df)
        st.markdown("""
                <style>
                    div.stButton > button:first-child {
                        background:linear-gradient(to bottom, #c1cbd7 5%, #c1cbd7 100%);
                        background-color:#c1cbd7;
                        border-radius:10px;
                        font-size:20px;
                        font-weight:bold;
                        text-shadow:-1px 1px 0px #c1cbd7;
                        height:2em;
                        width:100%;
                        color:#000000;
                    }
                </style>
            """, unsafe_allow_html=True)
        st.button("Tính khả năng tử vong", on_click=calculate_risk)

with st.sidebar.form(key ='Form1'):
    # Illustrate the left-side
    # Get User_data
    khongcolist = ["Không", "Có"]
    killiplist = ["Class I","Class II","Class III","Class IV"]
    clicnicaltypelist = ["Non-STEMI", "STEMI"]

    killip = st.selectbox('Phân độ Killip', options=['Class I','Class II','Class III','Class IV'])
    clicnicaltype = st.selectbox('Thể lâm sàng', options=['Non-STEMI','STEMI'])
    rca = st.selectbox('RCA', options = ["Không", "Có"])
    lda = st.selectbox('LDA', options = ["Không", "Có"])
    smoking = st.selectbox('Hút thuốc lá', options = ["Không", "Có"])
    aceiarb = st.selectbox('ACEiARB', options = ["Không", "Có"])
    anemia = st.selectbox('Thiếu máu', options = ["Không", "Có"])
    troponinadmission = st.number_input('Trị số Troponin (ng/mL)', min_value=1.0, max_value=1000.0, value=15.0, step=0.1, format=f'%.1f', help='Nhập đến một chữ số thập phân')
    age = st.number_input('Tuổi', min_value=10, max_value=90, value=64, step=1)
    gensiniscore = st.number_input('Điểm số Gensini', min_value=1, max_value=200, value=9, step=1)
    st.markdown("""
            <style>
                div.stButton > button:first-child {
                    background:linear-gradient(to bottom, #c1cbd7 5%, #c1cbd7 100%);
                    background-color:#c1cbd7;
                    border-radius:10px;
                    font-size:20px;
                    font-weight:bold;
                    text-shadow:-1px 1px 0px #c1cbd7;
                    height:2em;
                    width:100%;
                    color:#000000;
                }
            </style>
        """, unsafe_allow_html=True)
    submitted = st.form_submit_button(label = 'Đồng ý')
    #Store a dictionary into a variables
    user_data = {'killip': killiplist.index(killip),
                    'clicnicaltype': clicnicaltypelist.index(clicnicaltype),
                    'rca': khongcolist.index(rca),
                    'lda': khongcolist.index(lda),
                    'smoking': khongcolist.index(smoking),
                    'aceiarb': khongcolist.index(aceiarb),
                    'anemia': khongcolist.index(anemia),
                    'troponinadmission': troponinadmission,
                    'age': age,
                    'gensiniscore': gensiniscore,
                }
    # Transform the data into a data frame
    user_input = pd.DataFrame(user_data, index=[0])

if st.session_state['load'] == 0:
    if not submitted:
        load_page()
        get_pickle_data()
        st.markdown("<h4 style='color: brown;font-weight: bold; background-color: #ffdbdb'>&emsp;Hướng dẫn<h6 style='color: brown; background-color: #ffdbdb'>&emsp;&emsp;&emsp;Bước 1: Nhập/chọn thông tin bệnh nhân ở thanh bên trái<br><br>&emsp;&emsp;&emsp;Bước 2: Kiểm tra thông tin bệnh nhân trong bảng tóm tắt<br><br>&emsp;&emsp;&emsp;Bước 3: Bấm tiên lượng</h6></h4>", unsafe_allow_html=True)

if submitted:
    load_page()
    no_load = show_user_input(user_input)
