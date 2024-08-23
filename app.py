import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pickle

#Load the trained model

model=tf.keras.models.load_model('model.h5')

## Load the trained model, scaler pickle, onehot
model=load_model('model.h5')

## load the encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('One_hot_geo.pkl', 'rb') as file:
    one_hot_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## streamlit app

st.title("Customer Churn Prediction")

#User input
geography=st.selectbox('Geography', one_hot_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age', 18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Produts",1,4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#Input format 

geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_geo.get_feature_names_out(['Geography']))

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale input data

input_data_scale=scaler.transform(input_data)

#Prediction churn

prediction=model.predict(input_data_scale)
prediction_prob=prediction[0][0]
if(prediction_prob>0.5):
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
