import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model from Hugging Face model hub
model_path = hf_hub_download(repo_id="SrikanthKontham/tourism_project", filename="toursim_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts whether a customer will **purchase a travel package**
based on their demographics and interaction details.
Please enter the customer details below to get a prediction.
""")

# Categorical inputs
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Standard"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10, step=1)
num_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=20, value=2, step=1)
passport = st.selectbox("Has Passport", [0, 1])
pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns a Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=1, step=1)
monthly_income = st.number_input("Monthly Income (USD)", min_value=0.0, max_value=100000.0, value=20000.0, step=500.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': type_of_contact,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_persons_visiting,
    'NumberOfFollowups': num_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("The customer is **likely to purchase** the travel package.")
    else:
        st.warning("The customer is **unlikely to purchase** the travel package.")
