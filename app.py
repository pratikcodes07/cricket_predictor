import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the dataset (Ensure 'dataset.csv' exists)
df = pd.read_csv("ipl_data.csv")

# Load the trained model (Ensure you have a saved model file)
model_filename = "cricket_score_model.pkl"
with open("cricket_score_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Cricket Score Predictor", layout="centered", initial_sidebar_state="collapsed")
st.title("🏏 Cricket Score Predictor")

# Extract unique values from dataset
venue = st.selectbox("Select Venue", df['venue'].unique().tolist())
batting_team = st.selectbox("Select Batting Team", df['bat_team'].unique().tolist())
bowling_team = st.selectbox("Select Bowling Team", df['bowl_team'].unique().tolist())
striker = st.selectbox("Select Striker", df['batsman'].unique().tolist())
bowler = st.selectbox("Select Bowler", df['bowler'].unique().tolist())

runs_last_5 = st.number_input("Runs scored in last 5 overs", min_value=0, value=0)
wickets_last_5 = st.number_input("Wickets lost in last 5 overs", min_value=0, value=0)
overs = st.number_input("Overs completed", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
runs = st.number_input("Current Score", min_value=0, value=0)
wickets = st.number_input("Wickets fallen", min_value=0, max_value=10, value=0)

# Prediction button
if st.button("Predict Score"):
    # Convert inputs to model format (ensure correct preprocessing as per trained model)
    input_data = np.array([[runs_last_5, wickets_last_5, overs, runs, wickets]])
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.subheader(f"Predicted Score: {int(prediction)}")
