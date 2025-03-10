import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# 🔹 Download stopwords only when needed
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

# Load English stopwords
stop_words = set(stopwords.words("english"))

# ============================
# 🔹 1. Load Pretrained Medical Q&A Model
# ============================
# qa_model_name = "deepset/roberta-base-squad2"  # Better model for medical Q&A
# tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
model_name = "dmis-lab/biobert-large-cased-v1.1-squad"  # ✅ Updated Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# ============================
# 🔹 2. Load Symptom Checker Model
# ============================
model = xgb.XGBClassifier()
model.load_model("symptom_disease_model.json")  # Load trained model
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))  # Load label encoder
X_train = pd.read_csv("X_train.csv")  # Load symptoms
symptom_list = X_train.columns.tolist()

# ============================
# 🔹 3. Load Precaution Data
# ============================
precaution_df = pd.read_csv("Disease precaution.csv")
precaution_dict = {
    row["Disease"].strip().lower(): [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
    for _, row in precaution_df.iterrows()
}

# ============================
# 🔹 4. Load Medical Context
# ============================
def load_medical_context():
    with open("medical_context.txt", "r", encoding="utf-8") as file:
        return file.read()

medical_context = load_medical_context()

# ============================
# 🔹 5. Doctor Database
# ============================
doctor_database = {
    "malaria": [{"name": "Dr. Rajesh Kumar", "specialty": "Infectious Diseases", "location": "Apollo Hospital", "contact": "9876543210"}],
    "diabetes": [{"name": "Dr. Anil Mehta", "specialty": "Endocrinologist", "location": "AIIMS Delhi", "contact": "9876543233"}],
    "heart attack": [{"name": "Dr. Vikram Singh", "specialty": "Cardiologist", "location": "Medanta Hospital", "contact": "9876543255"}],
}

# ============================
# 🔹 6. Predict Disease from Symptoms
# ============================
def predict_disease(user_symptoms):
    """Predicts disease based on user symptoms using the trained XGBoost model."""
    input_vector = np.zeros(len(symptom_list))

    for symptom in user_symptoms:
        if symptom in symptom_list:
            input_vector[symptom_list.index(symptom)] = 1

    input_vector = input_vector.reshape(1, -1)  # Reshape for model input
    predicted_class = model.predict(input_vector)[0]  # Predict disease
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_disease

# ============================
# 🔹 7. Get Precautions for a Disease
# ============================
def get_precautions(disease):
    """Returns the precautions for a given disease."""
    return precaution_dict.get(disease.lower(), ["No precautions available"])

# ============================
# 🔹 8. Answer Medical Questions (Q&A Model)
# ============================
def get_medical_answer(question):
    """Uses the pre-trained Q&A model to answer general medical questions."""
    inputs = tokenizer(question, medical_context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    if answer.strip() in ["", "[CLS]", "<s>"]:
        return "I'm not sure. Please consult a medical professional."
    
    return answer
# ============================
# 🔹 9. Book a Doctor's Appointment
# ============================
def book_appointment(disease):
    """Finds a doctor for the given disease and returns appointment details."""
    disease = disease.lower().strip()
    doctors = doctor_database.get(disease, [])
    if not doctors:
        return f"Sorry, no available doctors found for {disease}."

    doctor = doctors[0]
    return f"Appointment booked with **{doctor['name']}** ({doctor['specialty']}) at **{doctor['location']}**.\nContact: {doctor['contact']}"

# ============================
# 🔹 10. Handle User Queries
# ============================
def handle_user_query(user_query):
    """Handles user queries related to symptoms, diseases, and doctor appointments."""
    user_query = user_query.lower().strip()

    # Check if query is about symptoms
    if "symptoms" in user_query or "signs" in user_query:
        disease = user_query.replace("symptoms", "").replace("signs", "").strip()
        return get_medical_answer(f"What are the symptoms of {disease}?")

    # Check if query is about treatment
    elif "treatment" in user_query or "treat" in user_query:
        disease = user_query.replace("treatment", "").replace("treat", "").strip()
        return get_medical_answer(f"What is the treatment for {disease}?")

    # Check for doctor recommendation
    elif "who should i see" in user_query:
        disease = user_query.replace("who should i see for", "").strip()
        return book_appointment(disease)

    # Check for appointment booking
    elif "book appointment" in user_query:
        disease = user_query.replace("book appointment for", "").strip()
        return book_appointment(disease)

    # Default case: general medical question
    else:
        return get_medical_answer(user_query)
