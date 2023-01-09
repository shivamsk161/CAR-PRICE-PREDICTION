import streamlit as st
import pandas as pd
from pickle import load

lb=load(open('label_encoder.pkl','rb'))
slr = load(open('standard_scaler.pkl', 'rb'))
gb=load(open('gb.pkl','rb'))


st.title('Car Price Prediction')

df2=pd.read_csv('car_d2.csv')


with st.form('my_form'):
    name = st.selectbox(label='Name', options=df2.name.unique())
    fuel = st.selectbox(label='Fuel', options=df2.fuel.unique())
    stype = st.selectbox(label='Seller Type', options=df2.seller_type.unique())
    trns = st.selectbox(label='Transmission', options=df2.transmission.unique())
    own = st.selectbox(label='Owner', options=df2.owner.unique())

    yr = st.selectbox(label='Name', options=df2.year.unique())
        # yr = st.number_input('Enter model year : ')
    kmd = st.select_slider('Select Length of diamond in mm', options=df2.km_driven.unique())
        # kmd = st.number_input('Enter km driven : ')

    btn = st.form_submit_button(label='Predict')

    if btn:
        if name and fuel and stype and trns and own and yr and kmd:
            query_cat = pd.DataFrame({'name':[name], 'fuel':[fuel],'seller_type':[stype],'transmission':[trns],'owner':[own]})
            query_num = pd.DataFrame({'year':[yr], 'km_driven':[kmd]})   

            query_cat = oe.transform(query_cat)
            query_num = slr.transform(query_num)

            query_point = pd.concat([pd.DataFrame(query_cat), pd.DataFrame(query_num)], axis=1)

            price = gbr.predict(query_point)

            st.success(f"The Price is $ {round(price[0],0)}")

        else:
            st.error('Please enter all values')
