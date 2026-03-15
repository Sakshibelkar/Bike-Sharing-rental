import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Dynamically get exact feature names from training process
    df = pd.read_csv('Dataset.csv')
    df.replace('?', np.nan, inplace=True)
    
    # Same preprocessing as training
    num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'yr', 'mnth', 'hr']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    df['is_peak_hour'] = df['hr'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    
    cat_cols = ['season', 'weathersit', 'weekday', 'workingday', 'holiday']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df_encoded.drop(['cnt', 'dteday', 'instant'], axis=1)
    
    # These are EXACTLY the features scaler was trained on
    feature_names = X.columns.tolist()
    
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

st.title("🚲 Bike Rental Demand Prediction")
st.markdown(f"✅ Model loaded with **{len(feature_names)} features**")

# Input collection
st.sidebar.header("📊 Input Features")
col1, col2 = st.sidebar.columns(2)
with col1:
    yr = st.sidebar.selectbox("Year", [2011, 2012], index=0)
with col2:
    mnth = st.sidebar.selectbox("Month", list(range(1,13)), index=0)

hr = st.sidebar.slider("Hour (0-23)", 0, 23, 12)
temp = st.sidebar.slider("Temperature (0-1)", 0.0, 1.0, 0.5, 0.01)
atemp = st.sidebar.slider("Apparent Temperature (0-1)", 0.0, 1.0, 0.5, 0.01)
hum = st.sidebar.slider("Humidity (0-1)", 0.0, 1.0, 0.6, 0.01)
windspeed = st.sidebar.slider("Wind Speed (0-1)", 0.0, 1.0, 0.2, 0.01)

col1, col2 = st.sidebar.columns(2)
casual = st.sidebar.slider("Casual Users", 0, 1000, 50)
registered = st.sidebar.slider("Registered Users", 0, 4000, 200)

is_peak_hour = st.sidebar.checkbox("Peak Hour (7-9, 17-19)", value=False)

# Categorical inputs
st.sidebar.header("🏷️ Categorical Features")
season_map = {0: 'springer', 1: 'summer', 2: 'winter', 3: 'fall'}
season = st.sidebar.selectbox("Season", list(season_map.keys()), format_func=lambda x: list(season_map.values())[x])
weathersit = st.sidebar.selectbox("Weather", ['Clear', 'Mist', 'Light Snow', 'Heavy Rain Snow Fog'])
weekday = st.sidebar.selectbox("Weekday (0=Sun)", list(range(7)))
workingday = st.sidebar.selectbox("Working Day", ['No work', 'Working Day'])
holiday = st.sidebar.selectbox("Holiday", ['No', 'Yes'])

if st.sidebar.button("🔮 Predict Demand"):
    # Create feature vector matching EXACT training order
    feature_vector = np.zeros(len(feature_names))
    
    # Set numeric features
    feature_vector[feature_names.index('yr')] = yr
    feature_vector[feature_names.index('mnth')] = mnth  
    feature_vector[feature_names.index('hr')] = hr
    feature_vector[feature_names.index('temp')] = temp
    feature_vector[feature_names.index('atemp')] = atemp
    feature_vector[feature_names.index('hum')] = hum
    feature_vector[feature_names.index('windspeed')] = windspeed
    feature_vector[feature_names.index('casual')] = casual
    feature_vector[feature_names.index('registered')] = registered
    feature_vector[feature_names.index('is_peak_hour')] = 1 if is_peak_hour else 0
    
    # Set categorical features using exact column names from training
    season_cols = [col for col in feature_names if 'season_' in col]
    weather_cols = [col for col in feature_names if 'weathersit_' in col]
    weekday_cols = [col for col in feature_names if 'weekday_' in col]
    
    # Season encoding (springer dropped)
    season_name = season_map[season]
    if season_name != 'springer' and f'season_{season_name}' in feature_names:
        feature_vector[feature_names.index(f'season_{season_name}')] = 1
        
    # Weather encoding (Clear dropped)
    if weathersit != 'Clear' and f'weathersit_{weathersit}' in feature_names:
        feature_vector[feature_names.index(f'weathersit_{weathersit}')] = 1
        
    # Weekday encoding (0 dropped)
    if f'weekday_{weekday}' in feature_names:
        feature_vector[feature_names.index(f'weekday_{weekday}')] = 1
        
    # Workingday and holiday (first category dropped)
    if 'workingday_No work' in feature_names and workingday == 'No work':
        feature_vector[feature_names.index('workingday_No work')] = 1
    if 'holiday_No' in feature_names and holiday == 'No':
        feature_vector[feature_names.index('holiday_No')] = 1
    
    # Scale and predict
    feature_scaled = scaler.transform([feature_vector])
    prediction = model.predict(feature_scaled)[0]
    
    st.success(f"### 🎯 Predicted Bike Rentals: **{prediction:.0f}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Casual Users", casual)
    with col2:
        st.metric("Registered Users", registered)
    with col3:
        st.metric("Total Demand", f"{prediction:.0f}", delta=f"{prediction-casual-registered:+.0f}")

# Show feature names for debugging
with st.expander("🔍 Debug: Feature Names (first 10)"):
    st.write(feature_names[:10])
    st.write(f"Total features: {len(feature_names)}")

st.markdown("---")
st.caption("Built with Streamlit | Model trained on Capital Bikeshare hourly dataset")
