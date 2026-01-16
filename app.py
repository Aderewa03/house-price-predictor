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
# CUSTOM CSS STYLING
# ========================================
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #eef2f7;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Title styling */
    .title {
        text-align: center;
        color: #1b3a57;
        font-size: 44px;
        font-weight: 700;
        margin-bottom: 5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #4b5c72;
        font-size: 20px;
        margin-bottom: 25px;
    }
    
    /* Section header styling */
    .section-header {
        color: #1b3a57;
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
        padding-left: 12px;
        border-left: 5px solid #00b894;
        background: #f0f4f8;
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 6px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #00b894, #019875);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 40px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #019875, #007f5f);
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,184,148,0.3);
    }
    
    /* Input container styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stVerticalBlock"] > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #d1d5db;
        padding: 8px;
        transition: border 0.3s ease;
    }
    .stNumberInput > div > div > input:focus {
        border: 2px solid #00b894;
        outline: none;
        box-shadow: 0 0 8px rgba(0,184,148,0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #00b894;
    }
    
    /* Success box styling */
    .success-box {
        background: linear-gradient(135deg, #6fcf97, #2d9c6f);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(47, 128, 80, 0.3);
        margin-top: 25px;
    }
    
    .price-text {
        font-size: 50px;
        font-weight: 700;
        margin: 10px 0;
        letter-spacing: 1px;
    }
    
    .price-label {
        font-size: 17px;
        opacity: 0.95;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #7a869a;
        font-size: 14px;
        margin-top: 40px;
        padding: 20px;
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
    except FileNotFoundError as e:
        st.error("⚠️ Model files not found. Please ensure 'model.h5' and 'scaler.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

model, scaler = load_model_and_scaler()

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    try:
        features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled, verbose=0)
        price = float(prediction[0][0]) * 100000
        return price
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
        return None

# ========================================
# MAIN APP INTERFACE
# ========================================
st.markdown('<p class="title">🏠 House Price Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered California Real Estate Valuation</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### 📋 Enter Property Details")
st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<p class="section-header">📊 Demographics & Economics</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    med_inc = st.number_input("💰 Median Income", min_value=0.5, max_value=15.0, value=3.5, step=0.1, help="Median income in block group (in tens of thousands of dollars)")
with col2:
    population = st.number_input("👥 Population", min_value=100, max_value=10000, value=1500, step=50, help="Total population in the block group")

st.markdown('<p class="section-header">🏡 Property Characteristics</p>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    house_age = st.number_input("🏚️ House Age", min_value=1, max_value=52, value=20, step=1, help="Median age of houses in block group (years)")
with col4:
    ave_rooms = st.number_input("🛋️ Avg Rooms", min_value=1.0, max_value=15.0, value=6.0, step=0.1, help="Average number of rooms per household")
with col5:
    ave_bedrms = st.number_input("🛏️ Avg Bedrooms", min_value=1.0, max_value=8.0, value=3.0, step=0.1, help="Average number of bedrooms per household")

st.markdown('<p class="section-header">👨‍👩‍👧‍👦 Occupancy</p>', unsafe_allow_html=True)
ave_occup = st.number_input("🏠 Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1, help="Average number of household members")

st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    latitude = st.number_input("🌐 Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01, help="Geographic latitude of block group")
with col7:
    longitude = st.number_input("🌐 Longitude", min_value=-124.0, max_value=-114.0, value=-118.0, step=0.01, help="Geographic longitude of block group")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Price"):
    with st.spinner("Analyzing property features with neural network..."):
        predicted_price = predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude)
        if predicted_price is not None:
            st.markdown(f"""
                <div class="success-box">
                    <div class="price-label">💰 Estimated Market Value</div>
                    <div class="price-text">${predicted_price:,.2f}</div>
                    <div class="price-label">Based on Neural Network analysis of California housing data</div>
                </div>
            """, unsafe_allow_html=True)
            st.info(f"""
            **Property Summary:**
            - 💰 Median Income: ${med_inc * 10000:,.0f}
            - 🏚️ House Age: {house_age} years
            - 🛋️ Average Rooms: {ave_rooms:.1f}
            - 🛏️ Average Bedrooms: {ave_bedrms:.1f}
            - 👥 Population: {population:,}
            - 🏠 Average Occupancy: {ave_occup:.1f}
            - 📍 Location: ({latitude:.2f}, {longitude:.2f})
            """)

st.divider()
st.markdown("""
    <div class="footer">
        Built with Streamlit & TensorFlow 🤖<br>
        Powered by Deep Learning Neural Network
    </div>
""", unsafe_allow_html=True)
