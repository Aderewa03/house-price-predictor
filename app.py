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
        background-color: #f5f7fa;
    }
    
    /* Title styling */
    .title {
        text-align: center;
        color: #1e3a5f;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    /* Section header styling */
    .section-header {
        color: #1e3a5f;
        font-size: 20px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-left: 10px;
        border-left: 4px solid #4a90e2;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4a90e2, #357abd);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 40px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #357abd, #2563eb);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }
    
    /* Input container styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #4a90e2;
    }
    
    /* Success box styling */
    .success-box {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
        margin-top: 20px;
    }
    
    .price-text {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .price-label {
        font-size: 16px;
        opacity: 0.9;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #94a3b8;
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
    """
    Load the pre-trained Keras model and StandardScaler.
    Uses caching to avoid reloading on every interaction.
    """
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
    """
    Predict house price using the pre-trained Keras model.
    
    Args:
        med_inc (float): Median income in block group (tens of thousands)
        house_age (float): Median house age in block group
        ave_rooms (float): Average number of rooms per household
        ave_bedrms (float): Average number of bedrooms per household
        population (float): Block group population
        ave_occup (float): Average number of household members
        latitude (float): Block group latitude
        longitude (float): Block group longitude
    
    Returns:
        float: Predicted price in dollars
    """
    try:
        # Prepare features array
        features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                              population, ave_occup, latitude, longitude]])
        
        # CRITICAL: Scale the features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)
        
        # Convert to actual price (model outputs in units of $100,000)
        price = float(prediction[0][0]) * 100000
        
        return price
    
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
        return None

# ========================================
# MAIN APP INTERFACE
# ========================================

# Title and subtitle
st.markdown('<p class="title">🏠 House Price Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered California Real Estate Valuation</p>', unsafe_allow_html=True)

# Spacing
st.markdown("<br>", unsafe_allow_html=True)

# Input section
st.markdown("### 📋 Enter Property Details")
st.markdown("<br>", unsafe_allow_html=True)

# ========================================
# SECTION 1: DEMOGRAPHICS & ECONOMICS
# ========================================
st.markdown('<p class="section-header">📊 Demographics & Economics</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input(
        "💰 Median Income",
        min_value=0.5,
        max_value=15.0,
        value=3.5,
        step=0.1,
        help="Median income in block group (in tens of thousands of dollars)"
    )

with col2:
    population = st.number_input(
        "👥 Population",
        min_value=100,
        max_value=10000,
        value=1500,
        step=50,
        help="Total population in the block group"
    )

# ========================================
# SECTION 2: PROPERTY CHARACTERISTICS
# ========================================
st.markdown('<p class="section-header">🏡 Property Characteristics</p>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    house_age = st.number_input(
        "🏚️ House Age",
        min_value=1,
        max_value=52,
        value=20,
        step=1,
        help="Median age of houses in block group (years)"
    )

with col4:
    ave_rooms = st.number_input(
        "🛋️ Avg Rooms",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.1,
        help="Average number of rooms per household"
    )

with col5:
    ave_bedrms = st.number_input(
        "🛏️ Avg Bedrooms",
        min_value=1.0,
        max_value=8.0,
        value=3.0,
        step=0.1,
        help="Average number of bedrooms per household"
    )

# ========================================
# SECTION 3: OCCUPANCY
# ========================================
st.markdown('<p class="section-header">👨‍👩‍👧‍👦 Occupancy</p>', unsafe_allow_html=True)

ave_occup = st.number_input(
    "🏠 Average Occupancy",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
    help="Average number of household members"
)

# ========================================
# SECTION 4: LOCATION
# ========================================
st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)

col6, col7 = st.columns(2)

with col6:
    latitude = st.number_input(
        "🌐 Latitude",
        min_value=32.0,
        max_value=42.0,
        value=34.0,
        step=0.01,
        help="Geographic latitude of block group"
    )

with col7:
    longitude = st.number_input(
        "🌐 Longitude",
        min_value=-124.0,
        max_value=-114.0,
        value=-118.0,
        step=0.01,
        help="Geographic longitude of block group"
    )

# Spacing
st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button("🔮 Predict Price"):
    # Show loading spinner
    with st.spinner("Analyzing property features with neural network..."):
        # Make prediction
        predicted_price = predict_house_price(
            med_inc, house_age, ave_rooms, ave_bedrms, 
            population, ave_occup, latitude, longitude
        )
        
        if predicted_price is not None:
            # Display result in styled box
            st.markdown(f"""
                <div class="success-box">
                    <div class="price-label">💰 Estimated Market Value</div>
                    <div class="price-text">${predicted_price:,.2f}</div>
                    <div class="price-label">Based on Neural Network analysis of California housing data</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional info
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

# Divider
st.divider()

# Footer
st.markdown("""
    <div class="footer">
        Built with Streamlit & TensorFlow 🤖<br>
        Powered by Deep Learning Neural Network
    </div>
""", unsafe_allow_html=True)