import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fc;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102a56, #183f70);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(120deg, #102a56, #2563a6);
    padding: 42px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
}

.hero-tag {
    color: #b9dcff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero h1 {
    color: white;
    font-size: 39px;
    font-weight: 800;
    line-height: 1.2;
    margin-top: 18px;
}

.hero p {
    color: #e3efff;
    font-size: 16px;
    line-height: 1.7;
}

.info-card {
    background: white;
    border: 1px solid #e1e9f5;
    padding: 25px;
    border-radius: 17px;
    min-height: 170px;
    box-shadow: 0 5px 18px rgba(30, 60, 100, 0.04);
}

.info-card h3 {
    color: #17365f;
    font-size: 21px;
}

.info-card p {
    color: #65748b;
    font-size: 15px;
    line-height: 1.6;
}

.section-heading {
    color: #17365f;
    font-size: 25px;
    font-weight: 800;
    margin: 25px 0 18px 0;
}

.result-approved {
    background: #e8f8ef;
    border: 1px solid #9cdeb6;
    padding: 28px;
    border-radius: 17px;
    text-align: center;
}

.result-rejected {
    background: #fff0f0;
    border: 1px solid #f2b5b5;
    padding: 28px;
    border-radius: 17px;
    text-align: center;
}

.result-approved h2 {
    color: #167343;
}

.result-rejected h2 {
    color: #bd3434;
}

div.stButton > button {
    background: linear-gradient(90deg, #174b87, #2875bd);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 750;
    min-height: 48px;
    width: 100%;
}

div.stButton > button:hover {
    background: #102a56;
    color: white;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #174b87, #2875bd);
    color: white;
    border-radius: 10px;
    min-height: 48px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #718096;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. PROJECT FILE PATHS
# =====================================================

MODEL_PATH = "model/best_model.pkl"

FEATURE_PATH = "model/feature_columns.pkl"

DATA_PATH = "data/feature_engineered_loan_data.csv"

TRAIN_PATH = "data/X_train.csv"


# =====================================================
# 4. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Load exact training feature columns
    if os.path.exists(FEATURE_PATH):

        feature_columns = joblib.load(FEATURE_PATH)

    elif os.path.exists(TRAIN_PATH):

        train_data = pd.read_csv(TRAIN_PATH)

        feature_columns = train_data.columns.tolist()

    else:

        raise FileNotFoundError(
            "Feature columns file or X_train.csv not found."
        )

    return model, list(feature_columns)


# =====================================================
# 5. LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:

    model, feature_columns = load_model()

    data = load_dataset()

except Exception as e:

    st.error("Unable to load project files.")

    st.code(str(e))

    st.stop()


# =====================================================
# 6. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("# 🏦 Loan Predictor")

    st.caption("AI POWERED FINANCIAL SYSTEM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "📊 Prediction",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.success("Random Forest Model")

    st.caption("Model: best_model.pkl")

    st.caption("Status: Loaded Successfully")

    st.divider()

    st.markdown(
        "### Smart Loan Prediction"
    )

    st.caption(
        "Machine Learning Based Academic Project"
    )


# =====================================================
# 7. REUSABLE HERO SECTION
# =====================================================

def show_hero(tag, heading, description):

    st.html(f"""
    <div class="hero">

        <div class="hero-tag">
            {tag}
        </div>

        <h1>
            {heading}
        </h1>

        <p>
            {description}
        </p>

    </div>
    """)


# =====================================================
# 8. HOME PAGE
# =====================================================

if page == "🏠 Home":

    show_hero(
        "AI POWERED FINANCIAL ASSISTANT",
        "Loan Approval<br>Prediction System",
        "Smart loan prediction using Machine Learning. "
        "Enter applicant information and discover your "
        "predicted loan eligibility instantly."
    )

    st.markdown(
        '<div class="section-heading">Welcome to Loan Predictor</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.html("""
        <div class="info-card">

            <h3>🎯 Smart Prediction</h3>

            <p>
            Predict loan approval using a trained
            Random Forest classification model.
            </p>

        </div>
        """)

    with col2:

        st.html("""
        <div class="info-card">

            <h3>⚡ Instant Results</h3>

            <p>
            Enter applicant details and receive
            prediction results within seconds.
            </p>

        </div>
        """)

    st.markdown(
        '<div class="section-heading">Project Highlights</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best Algorithm",
            "Random Forest"
        )

    with c2:
        st.metric(
            "Test Accuracy",
            "92.83%"
        )

    with c3:
        st.metric(
            "Models Compared",
            "2"
        )

    st.info(
        "Select Prediction from the sidebar "
        "to check loan eligibility."
    )


# =====================================================
# 9. PREDICTION PAGE
# =====================================================

elif page == "📊 Prediction":

    show_hero(
        "SMART LOAN ELIGIBILITY CHECK",
        "Check Your Loan Eligibility",
        "Fill in the applicant details below to generate "
        "a prediction using our trained machine learning model."
    )

    st.markdown(
        '<div class="section-heading">👤 Applicant Information</div>',
        unsafe_allow_html=True
    )

    # Remove target column from input data
    input_data = data.drop(
        columns=["loan_status"],
        errors="ignore"
    )

    categorical_cols = input_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_cols = input_data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    user_input = {}

    with st.form("loan_prediction_form"):

        # ---------------------------------------------
        # NUMERICAL DETAILS
        # ---------------------------------------------

        st.markdown("#### 📈 Numerical Details")

        num_columns = st.columns(2)

        for i, col in enumerate(numerical_cols):

            median_value = input_data[col].median()

            if pd.isna(median_value):
                median_value = 0.0

            with num_columns[i % 2]:

                user_input[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(median_value),
                    min_value=0.0,
                    step=1.0,
                    key=f"num_{col}"
                )

        st.divider()

        # ---------------------------------------------
        # CATEGORICAL DETAILS
        # ---------------------------------------------

        st.markdown("#### 📋 Personal & Loan Details")

        cat_columns = st.columns(2)

        for i, col in enumerate(categorical_cols):

            options = sorted(
                input_data[col].dropna().unique().tolist(),
                key=lambda x: str(x)
            )

            if len(options) == 0:
                continue

            with cat_columns[i % 2]:

                user_input[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options=options,
                    key=f"cat_{col}"
                )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Loan Approval"
        )

    # ---------------------------------------------
    # MAKE PREDICTION
    # ---------------------------------------------

    if submitted:

        try:

            user_df = pd.DataFrame([user_input])

            # Encode categorical values
            user_encoded = pd.get_dummies(
                user_df,
                drop_first=True
            )

            # Match exact training feature columns
            user_encoded = user_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            user_encoded = user_encoded.astype(float)

            # Model prediction
            prediction = model.predict(
                user_encoded
            )[0]

            # Approval probability
            approval_probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    user_encoded
                )[0]

                classes = list(model.classes_)

                approved_class = next(
                    (
                        cls for cls in classes
                        if str(cls).strip() == "1"
                    ),
                    None
                )

                if approved_class is not None:

                    approval_probability = (
                        probabilities[
                            classes.index(approved_class)
                        ] * 100
                    )

            st.divider()

            st.markdown(
                '<div class="section-heading">Prediction Result</div>',
                unsafe_allow_html=True
            )

            # -----------------------------------------
            # RESULT DISPLAY
            # -----------------------------------------

            if str(prediction).strip() == "1":

                st.html("""
                <div class="result-approved">

                    <h2>🎉 Loan Approved!</h2>

                    <p>
                    The machine learning model predicts
                    that this application belongs to
                    the approved class.
                    </p>

                </div>
                """)

            else:

                st.html("""
                <div class="result-rejected">

                    <h2>Loan Not Approved</h2>

                    <p>
                    The machine learning model predicts
                    that this application belongs to
                    the not-approved class.
                    </p>

                </div>
                """)

            if approval_probability is not None:

                st.markdown("### Approval Probability")

                st.metric(
                    "Predicted Approval Probability",
                    f"{approval_probability:.2f}%"
                )

                st.progress(
                    int(
                        np.clip(
                            approval_probability,
                            0,
                            100
                        )
                    )
                )

            st.caption(
                "This is an academic machine learning prediction. "
                "It is not an actual bank approval or financial decision."
            )

        except Exception as e:

            st.error("Prediction could not be generated.")

            st.code(str(e))


# =====================================================
# 10. ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ️ About Project":

    show_hero(
        "PROJECT INFORMATION",
        "About Our Project",
        "Understanding loan approval through data analytics "
        "and machine learning."
    )

    st.markdown(
        '<div class="section-heading">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write("""
    The Loan Approval Prediction System is a machine
    learning based application designed to predict
    loan approval outcomes using applicant information.

    It uses data cleaning, exploratory data analysis,
    feature engineering, preprocessing and classification
    algorithms to generate predictions.
    """)

    st.markdown(
        '<div class="section-heading">Technologies Used</div>',
        unsafe_allow_html=True
    )

    tech1, tech2 = st.columns(2)

    with tech1:

        st.markdown("""
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        """)

    with tech2:

        st.markdown("""
        - Streamlit
        - Matplotlib
        - Seaborn
        - Joblib
        """)

    st.markdown(
        '<div class="section-heading">Project Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    1. Data Collection
    2. Data Cleaning
    3. Exploratory Data Analysis
    4. Feature Engineering
    5. Data Preprocessing
    6. Model Training
    7. Model Evaluation
    8. Loan Approval Prediction
    """)

    st.info(
        "This project demonstrates the practical "
        "application of machine learning in financial analytics."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("---")

st.html("""
<div class="footer">

    🏦 Loan Approval Prediction System

    <br>

    Machine Learning Academic Project

</div>
""")