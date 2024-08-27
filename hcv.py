import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

# Load and preprocess the data
data = pd.read_csv("hcvdat0.csv")
data = data.drop(columns=['Unnamed: 0'])

numerical_cols = data.select_dtypes(include=[np.number]).columns
imputer_mean = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_mean.fit_transform(data[numerical_cols])

categorical_cols = data.select_dtypes(include=[object]).columns
imputer_mode = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_mode.fit_transform(data[categorical_cols])
data['ClassName'] = data['Category'].str.extract('=(.*)')

class_mapping = {
    'Blood Donor': 0,
    'suspect Blood Donor': 1,
    'Hepatitis': 2,
    'Fibrosis': 3,
    'Cirrhosis': 4
}
data['Class'] = data['ClassName'].map(class_mapping).astype(int)
data = data.drop(columns=['Category'])
data['Sex'] = data['Sex'].map({'m': 0, 'f': 1})

X = data.drop(columns=['Class', 'ClassName'])
y = data['Class']
class_names = list(class_mapping.keys())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Define prediction function
def predict_class(input_data):
    input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)
    return class_names[class_index]

# Streamlit UI
st.title("Liver Disease Classification")

st.write("Enter the input features for classification:")

# Collect user input
inputs = []
for feature in X.columns:
    value = st.number_input(f"Enter value for {feature}:")
    inputs.append(value)

# Predict button
if st.button("Predict"):
    predicted_class = predict_class(inputs)
    st.write(f"The predicted class is: {predicted_class}")

