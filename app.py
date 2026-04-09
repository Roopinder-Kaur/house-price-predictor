import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Image
st.image("house.jpg", use_container_width=True)

# Title
st.title("🏠 House Price Predictor")
st.write("Enter house details below to estimate price")

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
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
          'mainroad', 'guestroom', 'basement', 'airconditioning', 'prefarea']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Layout
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (Square Footage)", min_value=0)
    bedrooms = st.number_input("Bedrooms", min_value=0)
    bathrooms = st.number_input("Bathrooms", min_value=0)
    stories = st.number_input("Stories", min_value=0)
    parking = st.number_input("Parking", min_value=0)

with col2:
    mainroad = st.selectbox("Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])

# Convert dropdown to numbers
mainroad = 1 if mainroad == "yes" else 0
guestroom = 1 if guestroom == "yes" else 0
basement = 1 if basement == "yes" else 0
airconditioning = 1 if airconditioning == "yes" else 0
prefarea = 1 if prefarea == "yes" else 0

st.markdown("---")

# Button
if st.button("💰 Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking,
                                mainroad, guestroom, basement, airconditioning, prefarea]],
                             columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                                      'mainroad', 'guestroom', 'basement', 'airconditioning', 'prefarea'])

    price = model.predict(input_data)

    st.success(f"🏷️ Estimated Price: ₹ {int(price[0]):,}")