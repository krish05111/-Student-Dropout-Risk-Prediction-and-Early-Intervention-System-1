import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("models/dropout_model.pkl")

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🎓 Student Dropout Risk Prediction and Early Intervention System")

st.write(
    "This system predicts the dropout risk of a student "
    "and provides early intervention recommendations."
)

st.divider()

# -----------------------------------
# STUDENT INPUT
# -----------------------------------

st.header("📝 Student Details")

col1, col2 = st.columns(2)

with col1:

    attendance = st.slider(
        "Attendance Percentage",
        min_value=0,
        max_value=100,
        value=75
    )

    marks = st.slider(
        "Average Marks",
        min_value=0,
        max_value=100,
        value=60
    )

    study_hours = st.slider(
        "Daily Study Hours",
        min_value=0,
        max_value=12,
        value=3
    )


with col2:

    failed_subjects = st.slider(
        "Failed Subjects",
        min_value=0,
        max_value=10,
        value=0
    )

    assignments_completed = st.slider(
        "Assignments Completed",
        min_value=0,
        max_value=10,
        value=5
    )


st.divider()

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("🔍 Predict Dropout Risk", use_container_width=True):

    # Create DataFrame
    student_data = pd.DataFrame(
        [[
            attendance,
            marks,
            study_hours,
            failed_subjects,
            assignments_completed
        ]],
        columns=[
            "attendance",
            "marks",
            "study_hours",
            "failed_subjects",
            "assignments_completed"
        ]
    )

    # -----------------------------------
    # PREDICT DROPOUT PROBABILITY
    # -----------------------------------

    probability = model.predict_proba(student_data)[0][1]

    probability_percent = probability * 100

    # -----------------------------------
    # RISK CATEGORY
    # -----------------------------------

    if probability_percent < 30:

        risk_level = "LOW"

    elif probability_percent < 60:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # -----------------------------------
    # DISPLAY RESULT
    # -----------------------------------

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if risk_level == "LOW":

            st.success("🟢 LOW RISK")

        elif risk_level == "MEDIUM":

            st.warning("🟡 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")


    with result_col2:

        st.metric(
            "Dropout Risk Probability",
            f"{probability_percent:.2f}%"
        )


    st.divider()

    # -----------------------------------
    # RISK FACTORS
    # -----------------------------------

    st.header("🔍 Identified Risk Factors")

    risk_factors = []

    if attendance < 60:
        risk_factors.append("Low attendance")

    if marks < 50:
        risk_factors.append("Low academic marks")

    if study_hours < 2:
        risk_factors.append("Low study hours")

    if failed_subjects >= 2:
        risk_factors.append("Multiple failed subjects")

    if assignments_completed < 5:
        risk_factors.append("Low assignment completion")


    if len(risk_factors) == 0:

        st.success("✅ No major risk factors detected.")

    else:

        for factor in risk_factors:

            st.write(f"⚠️ {factor}")


    st.divider()

    # -----------------------------------
    # EARLY INTERVENTION
    # -----------------------------------

    st.header("💡 Recommended Early Intervention")

    recommendations = []

    if attendance < 60:

        recommendations.append(
            "Schedule attendance counselling with the student."
        )


    if marks < 50:

        recommendations.append(
            "Provide additional academic tutoring."
        )


    if study_hours < 2:

        recommendations.append(
            "Help the student create a regular study schedule."
        )


    if failed_subjects >= 2:

        recommendations.append(
            "Arrange academic mentoring and subject support."
        )


    if assignments_completed < 5:

        recommendations.append(
            "Follow up with the student regarding pending assignments."
        )


    if risk_level == "HIGH":

        recommendations.append(
            "Schedule an immediate meeting with an academic advisor."
        )


    if len(recommendations) == 0:

        recommendations.append(
            "Continue regular monitoring."
        )


    for recommendation in recommendations:

        st.write(f"👉 {recommendation}")


    st.divider()

    # -----------------------------------
    # ADVISOR REVIEW
    # -----------------------------------

    st.header("👨‍🏫 Advisor Review")

    advisor_decision = st.radio(
        "Advisor Decision",
        [
            "Accept Recommendation",
            "Override Recommendation"
        ]
    )


    advisor_notes = st.text_area(
        "Advisor Notes",
        placeholder="Enter advisor comments here..."
    )


    # -----------------------------------
    # SAVE ADVISOR REVIEW
    # -----------------------------------

    if st.button("💾 Save Advisor Review"):

        review_data = pd.DataFrame(
            [{
                "attendance": attendance,
                "marks": marks,
                "study_hours": study_hours,
                "failed_subjects": failed_subjects,
                "assignments_completed": assignments_completed,
                "risk_level": risk_level,
                "risk_probability": round(
                    probability_percent,
                    2
                ),
                "advisor_decision": advisor_decision,
                "advisor_notes": advisor_notes
            }]
        )


        review_file = "data/advisor_reviews.csv"


        # Check if file already exists
        if os.path.exists(review_file):

            review_data.to_csv(
                review_file,
                mode="a",
                header=False,
                index=False
            )

        else:

            review_data.to_csv(
                review_file,
                index=False
            )


        st.success(
            "✅ Advisor review saved successfully!"
        )


# -----------------------------------
# MODEL COMPARISON
# -----------------------------------

st.divider()

st.header("📈 Model Performance Comparison")

comparison_file = "models/model_comparison.csv"


if os.path.exists(comparison_file):

    comparison_data = pd.read_csv(
        comparison_file
    )

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

else:

    st.info(
        "Run train_model.py to generate model comparison results."
    )


# -----------------------------------
# PROJECT INFORMATION
# -----------------------------------

st.divider()

st.header("ℹ️ About the Project")

st.write(
    """
    This is a basic AI-assisted prototype designed to identify
    students who may be at risk of dropping out.

    The system uses Machine Learning to analyze student-related
    factors and predict dropout risk.

    The system considers:

    • Attendance

    • Academic Marks

    • Daily Study Hours

    • Failed Subjects

    • Assignment Completion

    Based on the identified risk factors, the system provides
    early intervention recommendations.

    The final decision remains with the academic advisor,
    who can accept or override the AI recommendation.
    """
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Student Dropout Risk Prediction and Early Intervention System | "
    "AI Project Prototype"
)