import streamlit as st
import pandas as pd
from pickle import load

lb=load(open('label_encoder.pkl','rb'))
slr = load(open('standard_scaler.pkl', 'rb'))
gb=load(open('gb.pkl','rb'))


st.title('Car Price Prediction')

df=pd.read_csv('car_df.csv')


with st.form('my_form'):
    name = st.selectbox(label='Name', options=df.name.unique())
    fuel = st.selectbox(label='Fuel', options=df.fuel.unique())
    stype = st.selectbox(label='Seller Type', options=df.seller_type.unique())
    trns = st.selectbox(label='Transmission', options=df.transmission.unique())
    own = st.selectbox(label='Owner', options=df.owner.unique())

    yr = st.selectbox(label='Name', options=df.year.unique())
    kmd = st.select_slider('Select Length of diamond in mm', options=df.km_driven.unique())

    btn = st.form_submit_button(label='Predict')

    if btn:
        if name and fuel and stype and trns and own and yr and kmd:
            query_cat = pd.DataFrame({'name':[name], 'fuel':[fuel],'seller_type':[stype],'transmission':[trns],'owner':[own]})
            query_num = pd.DataFrame({'year':[yr], 'km_driven':[kmd]})   

            query_num[['year','km_driven']]=pd.DataFrame(sc.fit_transform(query_num),columns=query_num.columns,index=query_num.index)
            query_cat['name']=lb.fit_transform(query_cat['name'])
            query_cat['fuel']=lb.fit_transform(query_cat['fuel'])
            query_cat['seller_type']=lb.fit_transform(query_cat['seller_type'])
            query_cat['transmission']=lb.fit_transform(query_cat['transmission'])
            query_cat['owner']=lb.fit_transform(query_cat['owner'])

            query=pd.concat([pd.DataFrame(query_cat),pd.DataFrame(query_num)],axis=1)

            Selling_price=gb.predict(query)

            st.success(f"Selling Price is Rs {round(Selling_price[0],0)}")

        else:
            st.error('Please enter all values')
