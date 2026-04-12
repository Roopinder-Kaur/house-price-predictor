import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Title
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")

# Load data
data = pd.read_csv("Housing.csv")

# Convert yes/no to 1/0
cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in cols:
    data[col] = data[col].map({'yes':1, 'no':0})

# Features
X = data[['area','bedrooms','bathrooms','stories','parking',
          'mainroad','guestroom','basement','airconditioning','prefarea']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏡 House Details")
    area = st.number_input("Area (sq ft)", min_value=0)
    bedrooms = st.number_input("Bedrooms", min_value=0)
    bathrooms = st.number_input("Bathrooms", min_value=0)
    stories = st.number_input("Stories", min_value=0)
    parking = st.number_input("Parking", min_value=0)

with col2:
    st.subheader("✨ Additional Features")
    mainroad = st.selectbox("Main Road", ["yes","no"])
    guestroom = st.selectbox("Guest Room", ["yes","no"])
    basement = st.selectbox("Basement", ["yes","no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes","no"])
    prefarea = st.selectbox("Preferred Area", ["yes","no"])

# Convert dropdown
mainroad = 1 if mainroad == "yes" else 0
guestroom = 1 if guestroom == "yes" else 0
basement = 1 if basement == "yes" else 0
airconditioning = 1 if airconditioning == "yes" else 0
prefarea = 1 if prefarea == "yes" else 0

st.markdown("---")

# Prediction
if st.button("💰 Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking,
                                mainroad, guestroom, basement, airconditioning, prefarea]],
                             columns=X.columns)

    price = model.predict(input_data)[0]

    # Price category
    if price < 5000000:
        category = "💵 Budget House"
    elif price < 10000000:
        category = "🏠 Mid-Range House"
    else:
        category = "🏡 Luxury House"

    st.success(f"💰 Estimated Price: ₹ {int(price):,}")
    st.info(f"Category: {category}")

# 📊 Graph section
st.markdown("---")
st.subheader("📊 Area vs Price Trend")

fig, ax = plt.subplots()
ax.scatter(data['area'], data['price'])
ax.set_xlabel("Area")
ax.set_ylabel("Price")

st.pyplot(fig)