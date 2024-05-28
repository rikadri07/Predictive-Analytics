# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('Predictive Analytics Dashboard')

data = pd.read_csv('data.csv')
st.write(data)

X = data[['feature1', 'feature2']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
st.pyplot(fig)
