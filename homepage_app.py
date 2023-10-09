import pandas as pd
import numpy as np
import streamlit as st
import sklearn 
import imblearn
import pickle 
from PIL import Image

# load dataset
df = pd.read_csv('clean_dataset.csv')

# load model
model = pickle.load(open('xgb_fix_tuned.pkl','rb'))
    
# create title (homepage)
def main():
    load_image = Image.open('./homepage.jpg')
    st.image(load_image)
    st.title('The Student Performance in Exams Prediction')
    st.subheader('Please Input Your Data Below!')

    # choose menu input - Selectbox
    # st.sidebar.subheader('Select Your Input')
    state = st.selectbox('Select Your State!', df['state'].unique())
    international_plan = st.selectbox('Select Your International Plan Package!', df['international_plan'].unique())
    voice_mail_plan = st.selectbox("Select Your Voice Mail Plan Package!", df['voice_mail_plan'].unique())

    # subtitle for numerical feature
    st.subheader('Select Your History of Telecommunications Services!')
    
    # choose menu input - selectbox for symptoms
    account_length = st.number_input('How is Your Account Length?', min_value=0, max_value=400)
    total_day_calls = st.number_input('How is Your Total Day Calls??', min_value=0, max_value=400)
    total_day_charge = st.number_input('How is Your Total Day Charge??', min_value=0, max_value=400)
    total_eve_calls = st.number_input('How is Your Total Evening Calls??', min_value=0, max_value=400)
    total_eve_charge = st.number_input('How is Your Total Evening Charge??', min_value=0, max_value=400)
    total_night_calls = st.number_input('How is Your Total Night Calls??', min_value=0, max_value=400)
    total_night_charge = st.number_input('How is Your Total Night Charge??', min_value=0, max_value=400)
    total_intl_calls = st.number_input('How is Your Total International Calls??', min_value=0, max_value=400)
    total_intl_charge = st.number_input('How is Your Total International Charge??', min_value=0, max_value=400)
    number_customer_service_calls = st.number_input('How is Your Total Customer Service Calls??', min_value=0, max_value=400)

    # prediction - button for predict
    if st.button('Predict',help='Click to predict'):
        
    # input the data in dataframe
        input_data = pd.DataFrame({
        'state': [state],
        'international_plan': [international_plan],
        'voice_mail_plan': [voice_mail_plan],
        'account_length': [account_length],
        'total_day_calls': [total_day_calls],
        'total_day_charge': [total_day_charge],
        'total_eve_calls': [total_eve_calls],
        'total_eve_charge': [total_eve_charge],
        'total_night_calls': [total_night_calls],
        'total_night_charge': [total_night_charge],
        'total_intl_calls': [total_intl_calls],
        'total_intl_calls': [total_intl_calls],
        'number_customer_service_calls': [number_customer_service_calls]
        })
        
        # do predict with model
        prediction = model.predict(input_data)

        st.subheader('Prediction Result')
        
        if prediction[0] == 0:
            st.success("Customers Who Have No Potential to Churn!")
        else:
            st.warning("Customers Who Have The Potential to Churn!")

    
    st.write('----')
    st.write('''
    Dashboard Created by [Tyovendi Arisandy](https://www.linkedin.com/in/tyovendiarisandy/)
    ''')

if __name__=='__main__':
    main()
