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
    
    /* Global Font adjustments - INCREASED BASE SIZE */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 18px; 
    }

    /* Target all markdown text for better readability */
    [data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
    }

    /* Container for a cleaner look */
    .main .block-container {
        padding-top: 2rem;
        max-width: 850px;
    }

    /* Title styling - BIGGER */
    .title {
        text-align: center;
        color: #4A3728;
        font-size: 54px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #8C7867;
        font-size: 24px;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Section header styling - BIGGER */
    .section-header {
        color: #5D4037;
        font-size: 28px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 3px solid #E0D7D0;
    }
    
    /* Button styling - BIGGER */
    .stButton>button {
        background: #5D4037;
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 600;
        border-radius: 50px;
        padding: 20px 25px;
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
    
    /* Input Labels - BIGGER */
    .stNumberInput label, .stSlider label {
        font-size: 20px !important;
        color: #5D4037 !important;
        font-weight: 700 !important;
    }

    /* Actual input text inside the box - BIGGER */
    input {
        font-size: 20px !important;
    }
    
    /* Success/Result box styling */
    .success-box {
        background: #FFFFFF;
        color: #4A3728;
        padding: 50px;
        border-radius: 25px;
        text-align: center;
        border: 2px solid #D7CCC8;
        box-shadow: 0 15px 35px rgba(0,0,0,0.07);
        margin-top: 30px;
    }
    
    .price-text {
        font-size: 64px;
        font-weight: 900;
        color: #5D4037;
        margin: 20px 0;
    }
    
    .price-label {
        font-size: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8C7867;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #A1887F;
        font-size: 16px;
        margin-top: 70px;
        padding: 30px;
        border-top: 1px solid #E0E0E0;
    }

    /* Styling for info box property summary */
    .stAlert {
        font-size: 18px !important;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# LOAD PRE-TRAINED MODEL & SCALER
# ========================================
@st.cache_resource
def load_model_and_scaler():
    """
    Load the pre-trained Keras model and StandardScaler.
    """
    try:
        model = load_model("model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Ensure 'model.h5' and 'scaler.pkl' are in the directory.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    """
    Predict house price using the pre-trained Keras model.
    """
    try:
        # Prepare features array
        features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                              population, ave_occup, latitude, longitude]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)
        
        # Convert to actual price (assuming model outputs in units of $100,000)
        price = float(prediction[0][0]) * 100000
        
        return price
    
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
        return None

# ========================================
# MAIN APP INTERFACE
# ========================================

# Header Section
st.markdown('<p class="title">Estimate house price 🏠</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Premium House Valuation</p>', unsafe_allow_html=True)

if model is None:
    st.stop()

# Input section
st.markdown("<br>", unsafe_allow_html=True)

# Organizing Inputs into a Card-like Structure
with st.container():
    # SECTION 1: LOCATION & DEMOGRAPHICS
    st.markdown('<p class="section-header">📍 Location & Demographic</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        med_inc = st.number_input(
            "💰 Median Income (x$10k)",
            min_value=0.5, max_value=15.0, value=3.5, step=0.1,
            help="Median income in tens of thousands"
        )
        latitude = st.number_input(
            "🌐 Latitude",
            min_value=32.0, max_value=42.0, value=34.0, step=0.01
        )
    with c2:
        population = st.number_input(
            "👥 Local Population",
            min_value=100, max_value=10000, value=1500, step=50
        )
        longitude = st.number_input(
            "🌐 Longitude",
            min_value=-124.0, max_value=-114.0, value=-118.0, step=0.01
        )

    # SECTION 2: PROPERTY DETAILS
    st.markdown('<p class="section-header">🏡 Property Characteristics</p>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        house_age = st.number_input("🏚️ House Age", 1, 52, 20)
    with c4:
        ave_rooms = st.number_input("🛋️ Total Rooms", 1.0, 15.0, 6.0, 0.1)
    with c5:
        ave_bedrms = st.number_input("🛏️ Total Bedrooms", 1.0, 8.0, 3.0, 0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    ave_occup = st.slider("👨‍👩‍👧‍👦 Average Occupancy per Home", 1.0, 10.0, 3.0, 0.1)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Logic
if st.button("Calculate Market Value"):
    with st.spinner("Analyzing property with Neural Network..."):
        predicted_price = predict_house_price(
            med_inc, house_age, ave_rooms, ave_bedrms, 
            population, ave_occup, latitude, longitude
        )
        
        if predicted_price is not None:
            # Main Result Card
            st.markdown(f"""
                <div class="success-box">
                    <div class="price-label">Estimated Property Value</div>
                    <div class="price-text">${predicted_price:,.0f}</div>
                    <div style="color:#8C7867; font-size: 1.2rem; font-weight: 400;">
                        Analysis complete. This estimate is based on deep learning <br>
                        regression against the California housing dataset.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Secondary Summary
            st.info(f"""
            **Property Summary:**
            - 🏚️ Age: {house_age} years | 🛋️ Rooms: {ave_rooms:.1f} | 🛏️ Bedrooms: {ave_bedrms:.1f}
            - 👥 Population: {population:,} | 👨‍👩‍👧‍👦 Occupancy: {ave_occup:.1f}
            """)

# Footer
st.markdown("""
    <div class="footer">
        <b>Architecture:</b> Deep Neural Network (Keras) &bull; <b>Data:</b> CA Housing Dataset<br>
        Developed for Real Estate Professionals | 2026
    </div>
""", unsafe_allow_html=True)

