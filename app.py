import streamlit as st
from src.predict import predict
import csv
import os

st.title("Fake News Classifier")
user_input = st.text_area("Enter a news article or headline:", key="user_input")

# Store prediction in session state
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if st.button("Classify"):
    result = predict(user_input)
    st.session_state['prediction'] = result
    st.session_state['last_input'] = user_input

if st.session_state['prediction'] is not None:
    result = st.session_state['prediction']
    st.write(f"### ❗ This news is **{result}**")
    if result == "FAKE":
        st.warning("This news is likely fake. Please verify with trusted sources.")
    else:
        st.success("This news is likely real. However, always verify with trusted sources.")
        st.balloons()

    # Feedback section
    st.markdown("---")
    st.write("#### Was this prediction correct?")
    feedback = st.radio("Select the correct label:", ("REAL", "FAKE"), index=0 if result=="REAL" else 1, key="feedback_radio")
    if st.button("Submit Feedback"):
        feedback_file = "user_feedback.csv"
        write_header = not os.path.exists(feedback_file)
        with open(feedback_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["text", "model_prediction", "user_label"])
            writer.writerow([st.session_state['last_input'], result, feedback])
        st.success("Thank you for your feedback! It will help improve the model.")
        st.session_state['prediction'] = None  # Reset after feedback