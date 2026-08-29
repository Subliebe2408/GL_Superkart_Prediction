
import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

st.subheader("Online Prediction")

# Collect user input for product/store features
product_weight = st.number_input("Product Weight")
product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_area = st.number_input("Product Allocated Area")
product_mrp = st.number_input("Product MRP")
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Food Mart"])
product_id_char = st.selectbox("Product Id Char", ["FD", "DR", "NC"]) 
store_age = st.number_input("Store Age (Years)", min_value=0, step=1)
product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"]) 

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar,
    "Product_Allocated_Area": product_area,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,  # ✅ corrected
    "Store_Age_Years": store_age,
    "Product_Type_Category": product_type_category  # ✅ corrected
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data.to_dict(orient='records')[0])
    if response.status_code == 200:
        prediction = response.json()["Predicted Sales"]
        st.success(f"Predicted Sales: {prediction}")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")  # Added error details for debugging

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")  # Added error details for debugging
