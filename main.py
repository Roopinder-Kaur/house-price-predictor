import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load real dataset
data = pd.read_csv("Housing.csv")

print(data.head())

# Convert yes/no to 1/0
data['mainroad'] = data['mainroad'].map({'yes':1, 'no':0})
data['guestroom'] = data['guestroom'].map({'yes':1, 'no':0})
data['basement'] = data['basement'].map({'yes':1, 'no':0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes':1, 'no':0})
data['airconditioning'] = data['airconditioning'].map({'yes':1, 'no':0})
data['prefarea'] = data['prefarea'].map({'yes':1, 'no':0})

# Features
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

# Target
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nPredicted:", y_pred[:5])
print("Actual:", list(y_test[:5]))

# Error
error = mean_squared_error(y_test, y_pred)
print("\nError:", error)

# Custom prediction
new_house = [[3000, 3, 2, 2, 1]]
price = model.predict(new_house)

print("\nPredicted price:", price)