# """
# Employee Attrition Prediction - Streamlit Dashboard
# 
# Run with:  streamlit run app.py

# Loads the pipeline saved by analysis.ipynb (attrition_model.pkl) and uses it to:
#   1. Predict attrition risk for a single employee entered via a form
#   2. Score a whole CSV of employees at once (bulk upload)
#   3. Show the top 10 feature importances behind the model
#   4. Give a short, plain-language recommendation per employee



import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# ----------------------------------------------------------------------------
# Load the saved pipeline (preprocessing + model bundled together).
# Cached so it only loads once per session, not on every interaction.
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("attrition_model.pkl")

model = load_model()

# -
# Same prediction function as in the notebook - one employee in, prediction out.
#
def predict_employee(employee: dict) -> dict:
    employee_df = pd.DataFrame([employee])
    probability_leave = model.predict_proba(employee_df)[0, 1]
    prediction = model.predict(employee_df)[0]
    return {
        "prediction": "Likely to Leave" if prediction == 1 else "Likely to Stay",
        "probability_of_leaving": round(float(probability_leave), 3),
    }

# A short, plain-language nudge based on the predicted probability band.
# (Simple if/elif on purpose - no need for a lookup table for 3 bands.)
def recommendation_for(probability):
    if probability >= 0.5:
        return "High risk - recommend a retention conversation this month."
    elif probability >= 0.3:
        return "Moderate risk - worth a check-in on workload and career growth."
    else:
        return "Low risk - no action needed right now."

st.title("Employee Attrition Predictor")
st.caption("Built on the IBM HR Analytics Attrition dataset - "
           "for HR use as a screening aid, not a final decision-maker.")

tab1, tab2, tab3 = st.tabs(["Single Employee", "Bulk Upload (CSV)", "What Drives Attrition"])

# 
# TAB 1 - Single employee form
# 
with tab1:
    st.subheader("Enter employee details")
    st.write("Fill in the fields below and click Predict.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=35)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
        ])
        education = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)

    with col2:
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=5)
        job_level = st.slider("Job Level", 1, 5, 2)
        job_involvement = st.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)

    with col3:
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
        years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3)
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)
        work_life_balance = st.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)

    # A few lower-impact fields with sensible fixed defaults, kept out of the
    # form so it doesn't become a 30-field wall - the model still needs them,
    # so we fill them in here rather than asking the user for every single one.
    extra_defaults = {
        "DailyRate": 800, "HourlyRate": 65, "MonthlyRate": 14000,
        "NumCompaniesWorked": 2, "PercentSalaryHike": 14, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TrainingTimesLastYear": 3,
    }

    if st.button("Predict", type="primary"):
        employee = {
            "Age": age, "BusinessTravel": business_travel, "Department": department,
            "DistanceFromHome": distance_from_home, "Education": education,
            "EducationField": education_field, "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender, "JobInvolvement": job_involvement, "JobLevel": job_level,
            "JobRole": job_role, "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
            "OverTime": overtime, "TotalWorkingYears": total_working_years,
            "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
            **extra_defaults,
        }
        result = predict_employee(employee)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Prediction", result["prediction"])
        with c2:
            st.metric("Probability of Leaving", f"{result['probability_of_leaving'] * 100:.1f}%")
        st.info(recommendation_for(result["probability_of_leaving"]))

# 
# TAB 2 - Bulk CSV upload
# 
with tab2:
    st.subheader("Score a batch of employees")
    st.write("Upload a CSV with the same columns as `HR_Attrition.csv` "
             "(raw, un-encoded - the model handles preprocessing internally).")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        # Drop the same constant/ID columns the notebook drops, if they're present
        cols_to_drop = ["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount", "Attrition"]
        batch_features = batch_df.drop(columns=[c for c in cols_to_drop if c in batch_df.columns])

        probabilities = model.predict_proba(batch_features)[:, 1]
        predictions = model.predict(batch_features)

        results_df = batch_df.copy()
        results_df["Probability_of_Leaving"] = np.round(probabilities, 3)
        results_df["Prediction"] = np.where(predictions == 1, "Likely to Leave", "Likely to Stay")
        results_df["Recommendation"] = [recommendation_for(p) for p in probabilities]
        results_df = results_df.sort_values("Probability_of_Leaving", ascending=False)

        st.write(f"Scored {len(results_df)} employees. Highest-risk employees shown first.")
        st.dataframe(results_df, use_container_width=True)

        csv_download = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored results as CSV", csv_download,
                            file_name="attrition_predictions.csv", mime="text/csv")

# 
# TAB 3 - Feature importance (same chart as the notebook's Task 6 / Chart 4)
# 
with tab3:
    st.subheader("What drives the model's predictions")

    classifier_step = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [f.split("__", 1)[1] for f in feature_names]

    if hasattr(classifier_step, "feature_importances_"):
        importances = pd.Series(classifier_step.feature_importances_, index=feature_names)
    else:
        importances = pd.Series(np.abs(classifier_step.coef_[0]), index=feature_names)

    top_10 = importances.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_10.sort_values().plot(kind="barh", color="#55A868", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)

    st.caption("This mirrors Chart 4 from the analysis notebook - the same model, "
               "the same ranking, just embedded in the dashboard for quick reference.")
