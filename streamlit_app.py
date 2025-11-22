import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("random_forest_model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer information to predict whether they are likely to churn.")

# --- Features & Mode Values ---
features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'Partner_Yes', 'Partner_No', 'Dependents_Yes', 'Dependents_No',
            'PhoneService_Yes', 'PhoneService_No', 'MultipleLines_Yes', 'MultipleLines_No', 'MultipleLines_No phone service',
            'InternetService_Fiber optic', 'InternetService_No', 'InternetService_DSL',
            'OnlineSecurity_Yes', 'OnlineSecurity_No', 'OnlineSecurity_No internet service',
            'OnlineBackup_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',
            'DeviceProtection_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service',
            'TechSupport_Yes', 'TechSupport_No', 'TechSupport_No internet service',
            'StreamingTV_Yes', 'StreamingTV_No', 'StreamingTV_No internet service',
            'StreamingMovies_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service',
            'Contract_One year', 'Contract_Two year', 'Contract_Month-to-month',
            'PaperlessBilling_Yes', 'PaperlessBilling_No',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check', 'PaymentMethod_Bank transfer (automatic)']

mode_values = [0.0]*len(features)  # replace with actual mode values
feature_dict = dict(zip(features, mode_values))

# --- Grouped binary features mapping ---
group_map = {
    "Partner": ("Partner_Yes", "Partner_No"),
    "Dependents": ("Dependents_Yes", "Dependents_No"),
    "Phone Service": ("PhoneService_Yes", "PhoneService_No"),
    "Multiple Lines": ("MultipleLines_Yes", "MultipleLines_No"),
    "Online Security": ("OnlineSecurity_Yes", "OnlineSecurity_No"),
    "Online Backup": ("OnlineBackup_Yes", "OnlineBackup_No"),
    "Device Protection": ("DeviceProtection_Yes", "DeviceProtection_No"),
    "Tech Support": ("TechSupport_Yes", "TechSupport_No"),
    "Streaming TV": ("StreamingTV_Yes", "StreamingTV_No"),
    "Streaming Movies": ("StreamingMovies_Yes", "StreamingMovies_No"),
    "Paperless Billing": ("PaperlessBilling_Yes", "PaperlessBilling_No"),
}

numeric_features = ["Tenure", "Monthly Charges", "Total Charges"]

# --- User chooses number of features to input ---
num_features = st.number_input("How many features do you want to manually input?", min_value=1, max_value=15, value=5)

user_inputs = {}
available_features = list(group_map.keys()) + numeric_features + ["Contract"]

for i in range(num_features):
    feature = st.selectbox(f"Select Feature {i+1}", options=available_features, key=f"feature_{i}")
    
    # Remove selected feature from remaining options
    available_features.remove(feature)
    
    if feature in group_map:
        value = st.selectbox(f"Enter value for {feature}:", ["Yes", "No"], key=f"value_{i}")
        yes_col, no_col = group_map[feature]
        user_inputs[yes_col] = 1.0 if value.lower() == "yes" else 0.0
        user_inputs[no_col] = 1.0 if value.lower() == "no" else 0.0
    
    elif feature == "Contract":
        value = st.selectbox("Select Contract Type:", ["Month-to-month", "One year", "Two year"], key=f"value_{i}")
        user_inputs["Contract_Month-to-month"] = 1.0 if value == "Month-to-month" else 0.0
        user_inputs["Contract_One year"] = 1.0 if value == "One year" else 0.0
        user_inputs["Contract_Two year"] = 1.0 if value == "Two year" else 0.0
    
    else:  # numeric
        val = st.number_input(f"Enter value for {feature}:", key=f"value_{i}")
        f_key = feature.replace(" ", "")  # convert display name to feature column
        user_inputs[f_key] = val

# --- Prepare final input array ---
input_array = []
for f in features:
    if f in user_inputs:
        input_array.append(user_inputs[f])
    else:
        input_array.append(feature_dict[f])

input_array = np.array(input_array).reshape(1, -1)

# --- Prediction ---
if st.button("Predict Churn"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk: {probability:.2f}%")
        st.write("Recommended Action: Offer discount or retention call.")
    else:
        st.success(f"✅ Customer Likely to Stay ({probability:.2f}% chance of churn)")
        st.write("Status: **Stable customer**")
