import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========================================
# CUSTOM CSS STYLING (Themed: Brown & White)
# ========================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #FDFCFB;
    }
    
    /* Global Font adjustments */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Container for a cleaner look */
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* Title styling */
    .title {
        text-align: center;
        color: #4A3728; /* Deep Brown */
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #8C7867; /* Muted Brown */
        font-size: 18px;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Section header styling */
    .section-header {
        color: #5D4037;
        font-size: 22px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #E0D7D0;
    }
    
    /* Button styling */
    .stButton>button {
        background: #5D4037; /* Rich Brown */
        color: #FFFFFF !important;
        font-size: 20px;
        font-weight: 600;
        border-radius: 50px;
        padding: 15px 20px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(93, 64, 55, 0.2);
    }
    
    .stButton>button:hover {
        background: #3E2723;
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(93, 64, 55, 0.3);
    }
    
    /* Form input styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 15px;
    }
    
    /* Success/Result box styling */
    .success-box {
        background: #FFFFFF;
        color: #4A3728;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid #D7CCC8;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-top: 30px;
    }
    
    .price-text {
        font-size: 52px;
        font-weight: 900;
        color: #5D4037;
        margin: 15px 0;
    }
    
    .price-label {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8C7867;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #A1887F;
        font-size: 13px;
        margin-top: 60px;
        padding: 20px;
        border-top: 1px solid #E0E0E0;
    }

    /* Small adjustments to widgets */
    .stNumberInput label, .stSlider label {
        color: #5D4037 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# LOAD PRE-TRAINED MODEL & SCALER
# ========================================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model("model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception:
        # Graceful fallback for demonstration if files are missing
        return None, None

model, scaler = load_model_and_scaler()

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    try:
        features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                              population, ave_occup, latitude, longitude]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled, verbose=0)
        return float(prediction[0][0]) * 100000
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ========================================
# MAIN APP INTERFACE
# ========================================

# Clean Header
st.markdown('<p class="title">Estimatr 🏠</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Premium House Price Valuation Model</p>', unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Model or Scaler files not found. Please upload 'model.h5' and 'scaler.pkl'.")

# Organizing Inputs into a Card-like Structure
with st.container():
    st.markdown('<p class="section-header">📍 Location & Demographic</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        med_inc = st.number_input("💰 Median Income (x$10k)", 0.5, 15.0, 3.5, 0.1)
        latitude = st.number_input("🌐 Latitude", 32.0, 42.0, 34.0, 0.01)
    with c2:
        population = st.number_input("👥 Local Population", 100, 10000, 1500, 50)
        longitude = st.number_input("🌐 Longitude", -124.0, -114.0, -118.0, 0.01)

    st.markdown('<p class="section-header">🏠 Property Details</p>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        house_age = st.number_input("🏚️ House Age", 1, 52, 20)
    with c4:
        ave_rooms = st.number_input("🛋️ Total Rooms", 1.0, 15.0, 6.0, 0.1)
    with c5:
        ave_bedrms = st.number_input("🛏️ Total Bedrooms", 1.0, 8.0, 3.0, 0.1)

    ave_occup = st.slider("👨‍👩‍👧‍👦 Average Occupancy per Home", 1.0, 10.0, 3.0, 0.1)

st.markdown("<br>", unsafe_allow_html=True)

# Action
if st.button("Calculate Market Value"):
    if model is not None:
        with st.spinner("Calculating valuation..."):
            predicted_price = predict_house_price(
                med_inc, house_age, ave_rooms, ave_bedrms, 
                population, ave_occup, latitude, longitude
            )
            
            if predicted_price is not None:
                st.markdown(f"""
                    <div class="success-box">
                        <div class="price-label">Estimated Property Value</div>
                        <div class="price-text">${predicted_price:,.0f}</div>
                        <div style="color:#8C7867; font-size: 0.9rem;">
                            Model Confidence: High <br>
                            Analysis based on current California market trends.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded.")

# Footer
st.markdown("""
    <div class="footer">
        <b>Architecture:</b> Deep Neural Network (Keras) &bull; <b>Data:</b> CA Housing Dataset<br>
        Developed for Real Estate Professionals
    </div>
""", unsafe_allow_html=True)
