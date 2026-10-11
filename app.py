import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AutoValue | Used Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #101827 0%, #172554 100%);
        color: #F8FAFC;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 32px;
        border-radius: 22px;
        background: linear-gradient(120deg, #1D4ED8, #4338CA, #6D28D9);
        margin-bottom: 25px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.20);
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E0E7FF;
        font-size: 17px;
        margin-bottom: 0;
    }

    .section-title {
        color: #C4B5FD;
        font-size: 24px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 18px;
    }

    .price-card {
        background: linear-gradient(120deg, #065F46, #047857);
        border: 1px solid #34D399;
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        margin-top: 24px;
        margin-bottom: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .price-label {
        color: #D1FAE5;
        font-size: 17px;
    }

    .price-value {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin: 8px 0;
    }

    .small-note {
        color: #CBD5E1;
        font-size: 13px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 25px;
        font-size: 17px;
        font-weight: 700;
        width: 100%;
        transition: 0.2s;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        border: 1px solid #C4B5FD;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #E2E8F0;
        font-weight: 600;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🚘 AutoValue</h1>
    <p>Used Car Price Prediction System</p>
    <p>Discover an estimated car value using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND DATA ----------------

try:
    model = joblib.load("models/car_price_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    df = pd.read_csv("car_data.csv")
    df.columns = df.columns.str.strip()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error.filename}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ---------------- CAR DETAILS FORM ----------------

st.markdown(
    '<div class="section-title">📝 Enter Car Details</div>',
    unsafe_allow_html=True
)

inputs = {}

with st.form("car_prediction_form"):

    left_column, right_column = st.columns(2)

    for index, column in enumerate(feature_columns):

        if column not in df.columns:
            continue

        current_column = (
            left_column if index % 2 == 0 else right_column
        )

        with current_column:

            label = column.replace("_", " ").title()

            # Text and categorical features
            if (
                pd.api.types.is_object_dtype(df[column])
                or pd.api.types.is_string_dtype(df[column])
                or isinstance(df[column].dtype, pd.CategoricalDtype)
            ):

                options = sorted(
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if options:
                    inputs[column] = st.selectbox(
                        label,
                        options,
                        key=f"input_{column}"
                    )
                else:
                    inputs[column] = st.text_input(
                        label,
                        key=f"input_{column}"
                    )

            # Numerical features
            else:

                numeric_values = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).dropna()

                if numeric_values.empty:

                    inputs[column] = st.number_input(
                        label,
                        value=0.0,
                        key=f"input_{column}"
                    )

                else:

                    minimum = float(numeric_values.min())
                    maximum = float(numeric_values.max())
                    median = float(numeric_values.median())

                    if minimum == maximum:

                        inputs[column] = minimum

                        st.text_input(
                            label,
                            value=str(minimum),
                            disabled=True,
                            key=f"display_{column}"
                        )

                    else:

                        inputs[column] = st.number_input(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=median,
                            key=f"input_{column}"
                        )

    submitted = st.form_submit_button(
        "✨ Predict Car Price"
    )


# ---------------- PREDICTION ----------------

if submitted:

    try:
        input_df = pd.DataFrame(
            [inputs],
            columns=feature_columns
        )

        prediction = float(model.predict(input_df)[0])

        if pd.isna(prediction):
            st.error("The model returned an invalid prediction.")

        elif prediction < 0:
            st.warning(
                "The model predicted a negative price. "
                "Please check the input data and model."
            )

        else:

            st.markdown(f"""
            <div class="price-card">
                <div class="price-label">
                    💰 Estimated Selling Price
                </div>
                <div class="price-value">
                    ₹{prediction:,.0f}
                </div>
                <div class="small-note">
                    Price estimated using Machine Learning
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------

st.markdown("""
<hr>
<p style="text-align:center; color:#CBD5E1; font-size:13px;">
    🚘 AutoValue | Used Car Price Prediction
</p>
""", unsafe_allow_html=True)