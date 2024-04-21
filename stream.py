import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('cricket.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_Runs_Scored(Highest_Score,Balls_Faced):
    features = np.array([Highest_Score,Balls_Faced])
    features = features.reshape(1,-1)
    emission = model.predict(features)
    return emission[0]

# Streamlit UI
st.title('Runs_Scored PREDICTION')
st.write("""
## Input Features
ENTER THE VALUES FOR THE INPUT FEATURES TO PREDICT Runs_Scored.
""")

# Input fields for user
Highest_Score = st.number_input('Highest_Score')
Balls_Faced = st.number_input('Balls_Faced')

# Prediction button
if st.button('Predict'):
    # Predict EMISSION
    Runs_Scored_prediction = predict_Runs_Scored(Highest_Score,Balls_Faced)
    st.write(f"PREDICTED Runs_Scored: {Runs_Scored_prediction}")