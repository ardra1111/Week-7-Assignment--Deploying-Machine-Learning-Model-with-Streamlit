import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

class_names = ['Setosa', 'Versicolor', 'Virginica']

st.title("🌸 Iris Flower Classifier")
st.write("""
This app predicts the **species of Iris flowers** based on user input.
Adjust the sliders below and get instant predictions.
""")

st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

    features = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }

    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"Predicted Class: **{class_names[prediction[0]]}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=class_names)
st.bar_chart(proba_df.T)
