---
title: AI Health Assistant Basic
emoji: 🌍
colorFrom: gray
colorTo: pink
sdk: streamlit
sdk_version: 1.43.1
app_file: app.py
pinned: false
license: mit
short_description: Used pretrained models, a basic one
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
what is the project does?
A Virtual Health Assistant is an AI-powered tool designed to assist users with basic health-related queries and medical appointments. It provides instant responses to common health concerns, offering general advice based on symptoms. The assistant does not replace doctors but helps users understand their conditions better. It can suggest possible causes for symptoms and recommend whether medical attention is needed. Additionally, it helps users find and book doctor appointments based on availability and specialization. The system ensures convenience by reducing wait times and providing accessible healthcare support. With natural language processing, it engages in user-friendly conversations. The assistant is available 24/7, making healthcare assistance more accessible. It enhances the patient experience by bridging the gap between self-care and professional consultation.

Tech stack we used:
streamlit,
pandas,
numpy,
pytorch,
transformers,
scikit-learn,
nltk

Advantages of vitual health assitant :
Instant Health Information:Provides quick, AI-driven answers to common health queries.
24/7 Availability:Users can get medical guidance anytime, anywhere.
Reduces Doctor Workload :Filters non-urgent cases, allowing doctors to focus on critical patients.
Smart Symptom Screening: Detects serious symptoms and suggests urgent medical attention.
Easy Appointment Booking: Helps users find the right doctor and book available slots instantly.
Cost-Effective:Reduces unnecessary clinic visits, saving time and money.
Scalable Solution: Can handle multiple patient queries simultaneously.
Personalized Healthcare:Tracks symptoms and offers tailored recommendations.
Integrates with Health Records:Syncs with hospital databases for seamless experience.
Multilingual & Accessible: Supports different languages for a wider reach.
 
 Input:
 we have tested some sample testcase ,when approcah with that testcases you can get the relevant output
 The test cases we tested are:
 basic queries like:
    "what is diabetes",
    "what are the symptoms of flu",
    "how to reduce fever",
    "what is hypertension"
symptom advice:
    "i have a fever",
    "i feel feverish",
    "i have a cough",
    "i have a headache",
    "i feel dizzy",
    "i have stomach pain"
critical_symptom:
    "I feel heart pain",
    "I have chest pain and shortness of breath",
    "I am experiencing severe headache and dizziness",
    "I have slurred speech and numbness on one side",
    "I have severe abdominal pain and vomiting blood",
doctor_specialties :
     "General Physician",
    "Cardiologist",
    "Dermatologist",
    "Neurologist",
    "Pediatrician",
    "Dentist"
