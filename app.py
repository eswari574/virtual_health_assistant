
import streamlit as st
from virtualhealth import handle_user_query, book_appointment  # Import functions

st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")

st.title("🩺 AI Health Assistant")
st.write("Ask any medical-related questions below:")

# Predefined responses for common medical queries
basic_queries = {
    "what is diabetes": "Diabetes is a chronic disease that affects how your body turns food into energy.",
    "what are the symptoms of flu": "Common flu symptoms include fever, cough, sore throat, and body aches.",
    "how to reduce fever": "Drink plenty of fluids, rest, and take fever-reducing medications like paracetamol.",
    "what is hypertension": "Hypertension, or high blood pressure, is when the force of blood against your artery walls is too high.",
}

# Symptom-Based Advice
symptom_advice = {
    "i have a fever": "Drink fluids, rest, and take paracetamol if needed. If fever persists for more than 3 days, consult a doctor.",
    "i feel feverish": "Monitor your temperature, drink water, and rest. If it gets worse, see a doctor.",
    "i have a cough": "Stay hydrated, avoid cold drinks, and try honey with warm water. If severe, consult a doctor.",
    "i have a headache": "Rest, drink water, and try a mild pain reliever if necessary. Persistent headaches need medical attention.",
    "i feel dizzy": "Sit down, drink water, and take deep breaths. If dizziness continues, consult a doctor.",
    "i have stomach pain": "Avoid spicy food, drink warm water, and rest. Severe pain may require a doctor's check-up.",
}

# Critical Symptoms Mapping
critical_symptoms = {
    "I feel heart pain": "Cardiologist",
    "I have chest pain and shortness of breath": "Cardiologist",
    "I am experiencing severe headache and dizziness": "Neurologist",
    "I have slurred speech and numbness on one side": "Neurologist",
    "I have severe abdominal pain and vomiting blood": "General Physician",
}

# Appointment Booking Data
doctor_specialties = {
    "General Physician": {"doctor": "Dr. Smith", "available_slots": ["2025-03-10 10:00:00", "2025-03-11 14:00:00"]},
    "Cardiologist": {"doctor": "Dr. Johnson", "available_slots": ["2025-03-12 09:30:00"]},
    "Dermatologist": {"doctor": "Dr. Lee", "available_slots": ["2025-03-10 11:00:00", "2025-03-11 15:00:00"]},
    "Neurologist": {"doctor": "Dr. Brown", "available_slots": ["2025-03-13 11:00:00"]},
    "Pediatrician": {"doctor": "Dr. Wilson", "available_slots": ["2025-03-14 10:00:00"]},
    "Dentist": {"doctor": "Dr. Davis", "available_slots": ["2025-03-15 12:00:00"]},
}

# User Input
user_input = st.text_input("Your Question or Symptoms:")

if st.button("Ask", key="ask_button"):
    user_input_lower = user_input.lower().strip()

    # Check if the query matches predefined questions
    if user_input_lower in basic_queries:
        st.markdown(f"**🤖 Bot:** {basic_queries[user_input_lower]}")
    
    # Check if the input matches known symptoms
    elif user_input_lower in symptom_advice:
        st.markdown(f"**🩺 Health Advice:** {symptom_advice[user_input_lower]}")
    
    # Check if it's a critical symptom
    elif user_input in critical_symptoms:
        specialty = critical_symptoms[user_input]
        st.warning(f"🚨 This may be a serious condition! Consider consulting a {specialty}.")
    
    # Otherwise, use AI response
    else:
        bot_response = handle_user_query(user_input)  
        st.markdown(f"**🤖 Bot:** {bot_response}")

# Doctor Appointment Booking
st.subheader("📅 Book a Doctor's Appointment")

selected_specialty = st.selectbox("Select Doctor Specialty", list(doctor_specialties.keys()))
appt_date = st.date_input("Select Appointment Date")
appt_time = st.time_input("Select Appointment Time")

if st.button("Book Appointment"):
    doctor_info = doctor_specialties[selected_specialty]
    doctor_name = doctor_info["doctor"]
    requested_slot = f"{appt_date} {appt_time}"

    if requested_slot in doctor_info["available_slots"]:
        booking_details = f"{selected_specialty} (Dr. {doctor_name}) on {requested_slot}"
        book_appointment(booking_details)  
        st.success(f"✅ Appointment confirmed with {doctor_name} on {requested_slot}.")
    else:
        nearest_slot = doctor_info["available_slots"][0]
        st.error(f"❌ No available doctors at that time. Nearest available slot: {nearest_slot} with {doctor_name}.")
