import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Spaceship Titanic Prediction App')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Make predictions
    predictions = model.predict(data)

    # Prepare and display the output
    output = pd.DataFrame({'PassengerId': data['PassengerId'], 'Transported': predictions})
    st.write("Predictions:")
    st.dataframe(output)

    # Download link for predictions
    st.download_button(
        label="Download Predictions as CSV",
        data=output.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv',
    )
