# ==============================================================================
# PRO-ACTIVE CHURN MANAGEMENT SYSTEM - Main Application File
# ==============================================================================
#
# Author: Bharath Kumar.V
# Version: 2.0.0
# Last Updated: 2025-09-08
#
# Description:
# This Streamlit application serves as a comprehensive suite for customer churn
# prediction and analysis. It provides four key modules:
#   1. Live Prediction Dashboard: For real-time analysis of a single customer.
#   2. Batch Prediction Center: For processing multiple customers from a CSV file.
#   3. Model Performance Deep Dive: For understanding the model's evaluation metrics.
#   4. Feature Explorer: For interactive exploration of the training data.
#
# The application is designed with a modern, professional UI and a modular
# codebase for maintainability and scalability.
# ==============================================================================

# --- 1. CORE IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# --- 2. PROJECT-SPECIFIC IMPORTS ---
# These imports rely on the structure of our project (the 'src' folder).
try:
    from src.utils import load_object, logging
    from src.feature_engineering import FeatureEngineering
except ImportError:
    # A fallback for users who might run the app without the full project structure.
    st.error("CRITICAL: The 'src' module was not found. Please ensure you are running this app from the root directory of the project and that the 'src' folder is present.")
    st.stop()


# --- 3. PAGE CONFIGURATION ---
# Sets the page title, icon, layout, and initial sidebar state.
# This must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Pro-Active Churn Management System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 4. MASSIVE CUSTOM CSS STYLING ---
# This extensive CSS block defines the entire visual theme of the application.
# It includes styles for layout, typography, widgets, animations, and custom classes.
def load_css():
    """
    Injects a large block of custom CSS for a modern, professional, and readable UI.
    This theme uses a dark palette with high-contrast elements and animations.
    """
    st.markdown("""
        <style>
            /* --- Font Import --- */
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

            /* --- Main App Styling & Background --- */
            .stApp {
                background: linear-gradient(to right top, #0b0f19, #12192c, #1a2341, #242d57, #30376e);
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                font-family: 'Roboto', sans-serif;
            }

            /* --- Glassmorphism Containers for a modern look --- */
            .glass-container {
                background: rgba(30, 35, 50, 0.7);
                border-radius: 16px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.18);
                padding: 30px;
                margin-bottom: 25px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .glass-container:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px 0 rgba(0, 191, 255, 0.3);
            }

            /* --- Typography for Readability --- */
            h1, h2, h3 {
                color: #FFFFFF;
                font-weight: 700;
                text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
            }

            .stMarkdown, p, .st-emotion-cache-16idsys p {
                color: #E0E0E0; /* Light gray for body text */
                font-size: 16px;
                line-height: 1.6;
            }
            
            /* --- Horizontal Rule Styling --- */
            hr {
                border-top: 1px solid rgba(0, 191, 255, 0.3);
            }

            /* --- Sidebar Styling --- */
            .st-emotion-cache-16txtl3 {
                background: rgba(15, 20, 30, 0.9);
                backdrop-filter: blur(15px);
                -webkit-backdrop-filter: blur(15px);
                border-right: 1px solid rgba(0, 191, 255, 0.2);
            }
            
            .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16txtl3 .stMarkdown p {
                color: #FFFFFF;
            }
            
            /* --- Customizing Streamlit Widgets for a cohesive look --- */
            
            /* Slider Styling */
            .st-emotion-cache-134i79s { /* Slider Thumb */
                background-color: #00BFFF; /* Deep sky blue */
                border: 2px solid white;
            }
            .st-emotion-cache-wnm525 { /* Slider Track */
                background-color: rgba(255,255,255,0.2);
            }

            /* Expander Header */
            .st-emotion-cache-1y4p8pa { 
                background-color: rgba(0, 191, 255, 0.1);
                border-radius: 8px;
                border: 1px solid rgba(0, 191, 255, 0.3);
                transition: background-color 0.3s ease;
            }
            .st-emotion-cache-1y4p8pa:hover {
                background-color: rgba(0, 191, 255, 0.2);
            }
            .st-emotion-cache-1y4p8pa p {
                color: #FFFFFF;
                font-weight: bold;
            }
            
            /* File Uploader */
            .st-emotion-cache-7dmedh {
                border-color: rgba(0, 191, 255, 0.5);
            }

            /* Download Button */
            .stDownloadButton>button {
                background-color: #39FF14;
                color: #0b0f19;
                font-weight: bold;
                border-radius: 8px;
                border: 1px solid #39FF14;
                transition: all 0.3s ease-in-out;
            }
            .stDownloadButton>button:hover {
                background-color: #FFFFFF;
                border-color: #FFFFFF;
                box-shadow: 0 0 15px #39FF14;
            }

            /* Metric Display */
            .st-emotion-cache-ocqkz7 {
                background-color: transparent !important;
                border: none !important;
                padding: 0 !important;
            }

            /* --- Custom classes for status and highlights --- */
            .highlight-text {
                color: #00BFFF;
                font-weight: bold;
            }
            .success-text {
                color: #39FF14; /* Neon green */
            }
            .error-text {
                color: #FF3131; /* Neon red */
            }

            /* --- Custom Scrollbar --- */
            ::-webkit-scrollbar {
                width: 10px;
            }
            ::-webkit-scrollbar-track {
                background: #12192c; 
            }
            ::-webkit-scrollbar-thumb {
                background: #242d57; 
                border-radius: 5px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #30376e; 
            }
            
            /* --- Animation for containers --- */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .main-content {
                animation: fadeIn 0.8s ease-out;
            }

            /* --- Hiding Streamlit's default branding --- */
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}

        </style>
    """, unsafe_allow_html=True)

load_css()


# --- 5. ASSET & DATA LOADING ---
# Caching functions ensure that data and models are loaded only once,
# which significantly improves the app's performance and responsiveness.

@st.cache_resource
def load_model_artifacts():
    """Loads the model and preprocessor, caching them for performance."""
    logging.info("Attempting to load model artifacts...")
    try:
        model = load_object(os.path.join("saved_models", "best_model.pkl"))
        preprocessor = load_object(os.path.join("saved_models", "preprocessor.pkl"))
        if model is None or preprocessor is None:
            st.error("FATAL: Model or Preprocessor not found. Please run the training pipeline first.", icon="🚨")
            return None, None
        logging.info("Model artifacts loaded successfully.")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}", icon="🔥")
        return None, None

@st.cache_data
def load_raw_data():
    """Loads the raw dataset for the Feature Explorer page."""
    logging.info("Attempting to load raw data...")
    try:
        raw_df = pd.read_csv(os.path.join("data", "raw", "ecommerce_churn_data.csv"))
        # Basic cleaning for visualization
        raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
        raw_df.dropna(inplace=True)
        return raw_df
    except FileNotFoundError:
        st.error("Raw data file not found. Please ensure 'ecommerce_churn_data.csv' is in the 'data/raw' folder.", icon="📁")
        return None
    except Exception as e:
        st.error(f"Error loading raw data: {e}", icon="🔥")
        return None

@st.cache_data
def get_image_as_base64(path):
    """Encodes an image to a base64 string for embedding in Markdown."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

# --- Initializing the cached objects ---
model, preprocessor = load_model_artifacts()
raw_df = load_raw_data()


# --- 6. UI COMPONENT HELPER FUNCTIONS ---
# These functions create reusable and styled UI components, keeping the page
# rendering logic clean and organized.

def create_gauge_chart(probability):
    """Creates a futuristic gauge chart for churn probability using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 40, 'color': "white"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24, 'color': "#00BFFF"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "cyan"},
            'bar': {'color': "#30376e"},
            'bgcolor': "rgba(0,0,0,0.2)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#39FF14'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': '#FF3131'}],
        }))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "sans-serif"})
    return fig

def display_key_drivers(input_data):
    """Analyzes and displays top features influencing a high churn prediction."""
    st.subheader("🔑 Key Prediction Drivers")
    
    # This is a rule-based explanation method. For more accuracy, libraries like SHAP or LIME would be used.
    # We assign weights to common high-risk factors.
    drivers = {}
    if input_data['Contract'].iloc[0] == 'Month-to-month':
        drivers["📄 Contract Type"] = ("Month-to-month contracts have the highest churn rate.", 3)
    if input_data['tenure'].iloc[0] <= 12:
        drivers["⏳ Low Tenure"] = ("Customers with a tenure of one year or less are more likely to churn.", 3)
    if input_data['InternetService'].iloc[0] == 'Fiber optic':
        drivers["🚀 Internet Service"] = ("Fiber optic customers, while valuable, have a higher churn rate in this dataset.", 2)
    if input_data['OnlineSecurity'].iloc[0] == 'No':
        drivers["🛡️ No Online Security"] = ("Lacking security services correlates with higher churn.", 2)
    if input_data['PaymentMethod'].iloc[0] == 'Electronic check':
         drivers["💳 Payment Method"] = ("Customers using electronic checks tend to churn more often.", 1)

    # Sort drivers by their impact weight (descending)
    sorted_drivers = sorted(drivers.items(), key=lambda item: item[1][1], reverse=True)

    if not sorted_drivers:
        st.info("This customer profile does not exhibit the most common high-risk churn factors.")
        return
        
    # Display top 3 drivers
    for i, (feature, (reason, _)) in enumerate(sorted_drivers[:3]):
        st.markdown(f"**{i+1}. {feature}**")
        st.markdown(f"> {reason}")

def create_download_link(df, filename):
    """Creates a link to download a DataFrame as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Processed CSV File</a>'


# --- 7. SIDEBAR & NAVIGATION ---
# The sidebar serves as the main navigation hub for the application.
st.sidebar.markdown("<h1>Pro-Active Churn Management System</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select a Module",
    ["Live Prediction Dashboard", "Batch Prediction Center", "Model Performance Deep Dive", "Feature Explorer"],
    key="app_mode"
)
st.sidebar.markdown("---")

# Conditional sidebar for user input, only shows on the Live Prediction page.
if app_mode == "Live Prediction Dashboard":
    st.sidebar.header("👤 Customer Profile Input")
    
    def get_user_input():
        with st.sidebar.expander("🔑 Account Information", expanded=True):
            tenure = st.slider("Tenure (Months)", 0, 72, 12, key="tenure")
            contract = st.select_slider("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 55.0, key="monthly")
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure), key="total")
            paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], key="billing", horizontal=True)

        with st.sidebar.expander("👤 Demographics"):
            gender = st.radio("Gender", ["Male", "Female"], key="gender", horizontal=True)
            senior_citizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key="senior", horizontal=True)
            partner = st.radio("Has Partner", ["Yes", "No"], key="partner", horizontal=True)
            dependents = st.radio("Has Dependents", ["Yes", "No"], key="dependents", horizontal=True)

        with st.sidebar.expander("📡 Services Subscribed"):
            phone_service = st.radio("Phone Service", ["Yes", "No"], key="phone", horizontal=True)
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="multiline")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security")
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="protection")
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="payment")

        data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        return pd.DataFrame([data])
    input_df = get_user_input()


# --- 8. PAGE RENDERING LOGIC ---
# Each page is encapsulated in its own function for clarity and modularity.

# ============================
# PAGE 1: LIVE PREDICTION DASHBOARD
# ============================
def render_dashboard_page():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h1>Live Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("### Real-time predictive insights into customer loyalty.")
    st.markdown("---")

    if model is not None and preprocessor is not None:
        try:
            featured_df = FeatureEngineering(input_df).transform_data()
            processed_data = preprocessor.transform(featured_df)
            prediction_value = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0][1]

            with st.container():
                st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                st.subheader("Customer Churn Risk Analysis")
                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.plotly_chart(create_gauge_chart(prediction_proba), use_container_width=True)
                with col2:
                    if prediction_value == 1:
                        st.markdown("<h3>Status: <span class='error-text'>HIGH CHURN RISK</span> 🚨</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3>Status: <span class='success-text'>LOW CHURN RISK</span> ✅</h3>", unsafe_allow_html=True)
                    st.markdown(f"The model predicts a <span class='highlight-text'>{prediction_proba:.2%}</span> probability that this customer will churn.", unsafe_allow_html=True)
                    st.markdown("---")
                    if prediction_value == 1:
                        display_key_drivers(input_df)
                    else:
                        st.info("This customer profile does not exhibit the most common high-risk churn factors.")

                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                st.subheader("💡 Suggested Retention Strategies")
                if prediction_value == 1:
                    st.warning("Immediate retention action is recommended.")
                    st.markdown("""
                        - **Offer Proactive Discount:** Provide a limited-time discount (e.g., 15% off for 3 months) to show appreciation.
                        - **Initiate Support Call:** A customer success agent should call to address potential issues and gather feedback.
                        - **Propose Contract Upgrade:** Offer a move to a 1-year contract with a bonus feature to increase stability.
                    """)
                else:
                    st.info("Customer is likely loyal. Focus on engagement and upselling.")
                    st.markdown("""
                        - **Promote Loyalty Program:** Ensure the customer is aware of the benefits of your loyalty program.
                        - **Introduce New Features:** Inform them about new, relevant services or features they might find valuable.
                        - **Offer Referral Bonus:** Encourage them to refer new customers in exchange for a credit or bonus.
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}", icon="🔥")
    else:
        st.warning("Model artifacts are not loaded. Cannot make a prediction.", icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# PAGE 2: BATCH PREDICTION CENTER
# ============================
def render_batch_prediction_page():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h1>Batch Prediction Center</h1>", unsafe_allow_html=True)
    st.markdown("### Process multiple customers by uploading a CSV file.")
    st.markdown("---")

    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("1. Upload Customer Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.write("Preview of your data:")
                st.dataframe(batch_df.head())

                if st.button("🚀 Run Batch Prediction"):
                    with st.spinner("Processing customers... This may take a moment."):
                        # Process the batch data
                        featured_batch_df = FeatureEngineering(batch_df.copy()).transform_data()
                        processed_batch_data = preprocessor.transform(featured_batch_df)
                        
                        predictions = model.predict(processed_batch_data)
                        probabilities = model.predict_proba(processed_batch_data)[:, 1]
                        
                        # Add results to the dataframe
                        results_df = batch_df.copy()
                        results_df['Predicted Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                        results_df['Churn Probability'] = probabilities
                        
                        st.subheader("2. Prediction Results")
                        st.dataframe(results_df)

                        st.subheader("3. Download Results")
                        st.markdown(create_download_link(results_df, "churn_predictions.csv"), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}", icon="🔥")
        else:
            st.info("Please upload a CSV file with the same columns as the training data to begin.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# PAGE 3: MODEL PERFORMANCE DEEP DIVE
# ============================
def render_model_deep_dive_page():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h1>Model Performance Deep Dive</h1>", unsafe_allow_html=True)
    st.markdown("### Understanding the performance and logic of the trained model.")
    st.markdown("---")

    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("📊 Key Performance Metrics")
        st.markdown("These metrics are based on the model's performance on the unseen test dataset.")
        
        # In a real-world scenario, these would be loaded from a JSON/YAML file saved by the pipeline.
        # For this example, we'll use the values from our last run.
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "79.8%")
        col2.metric("Precision", "63.9%")
        col3.metric("Recall", "55.3%")
        col4.metric("F1-Score", "59.3%")

        st.markdown("---")
        st.subheader("🖼️ Evaluation Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Confusion Matrix**")
            st.image("reports/confusion_matrix.png", use_column_width=True, caption="Shows the counts of correct and incorrect predictions.")
        with col2:
            st.markdown("**ROC Curve**")
            st.image("reports/roc_curve.png", use_column_width=True, caption="Illustrates the diagnostic ability of the model.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# PAGE 4: FEATURE EXPLORER
# ============================
def render_feature_explorer_page():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h1>Feature Explorer</h1>", unsafe_allow_html=True)
    st.markdown("### Interactively explore the raw dataset to find patterns.")
    st.markdown("---")

    if raw_df is not None:
        with st.container():
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.subheader("Explore Feature Distributions")
            
            # Select feature to plot
            all_cols = raw_df.columns.tolist()
            feature_to_plot = st.selectbox("Select a feature to visualize", all_cols, index=all_cols.index('tenure'))
            
            if pd.api.types.is_numeric_dtype(raw_df[feature_to_plot]):
                # Histogram for numerical features
                fig = px.histogram(raw_df, x=feature_to_plot, color="Churn", 
                                   title=f"Distribution of {feature_to_plot} by Churn",
                                   color_discrete_map={"Yes": "#FF3131", "No": "#39FF14"})
            else:
                # Bar chart for categorical features
                df_grouped = raw_df.groupby([feature_to_plot, 'Churn']).size().reset_index(name='count')
                fig = px.bar(df_grouped, x=feature_to_plot, y='count', color="Churn",
                             title=f"Distribution of {feature_to_plot} by Churn",
                             color_discrete_map={"Yes": "#FF3131", "No": "#39FF14"})
            
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Raw data is not available for exploration.", icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)


# --- 9. MAIN APPLICATION LOGIC ---
# This block controls which page is displayed based on the sidebar navigation.
if __name__ == "__main__":
    if app_mode == "Live Prediction Dashboard":
        render_dashboard_page()
    elif app_mode == "Batch Prediction Center":
        render_batch_prediction_page()
    elif app_mode == "Model Performance Deep Dive":
        render_model_deep_dive_page()
    elif app_mode == "Feature Explorer":
        render_feature_explorer_page()

# --- End of File ---
###
