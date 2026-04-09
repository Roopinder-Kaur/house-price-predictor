import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Housing.csv")

# Convert yes/no to 1/0
data['mainroad'] = data['mainroad'].map({'yes':1, 'no':0})
data['guestroom'] = data['guestroom'].map({'yes':1, 'no':0})
data['basement'] = data['basement'].map({'yes':1, 'no':0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes':1, 'no':0})
data['airconditioning'] = data['airconditioning'].map({'yes':1, 'no':0})
data['prefarea'] = data['prefarea'].map({'yes':1, 'no':0})

# Features
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏠 House Price Predictor")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
stories = st.number_input("Stories")
parking = st.number_input("Parking")

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]],
                              columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
    
    price = model.predict(input_data)
    
    st.success(f"Predicted Price: ₹ {int(price[0])}")