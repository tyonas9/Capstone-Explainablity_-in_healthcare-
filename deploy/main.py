#pip install streamlit
#!pip install pandas
import streamlit as st
import pandas as pd


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to the Awesome project!')
    st.text('In this project I look into ... And I try ... I worked with the dataset from ...')

with dataset:
    st.header('Dataset: my dataset')
    st.text('I found this dataset at... I decided to work with it because ...')

    
    taxi_data = pd.read_csv('data/train-store.csv') 
    st.write(taxi_data.head())

    pulocation_dist = pd.DataFrame(taxi_data['Dept'].value_counts())
    st.bar_chart(pulocation_dist)
    

with features:
    st.header('New features I came up with')
    st.text('Let\'s take a look into the features I generated.')

with model_training:
    st.header('Model training')
    st.text('In this section you can select the hyperparameters!')