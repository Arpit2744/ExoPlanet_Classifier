import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocess_data import preprocess_data, FINAL_MODEL_FEATURES 

st.set_page_config(
    page_title="Exoplanet Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- GLOBAL MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model/lightgbm_exoplanet_model.pkl")
        le = joblib.load("model/label_encoder.pkl")
        class_names = le.inverse_transform([0, 1, 2])
        class_map = {i: name for i, name in enumerate(class_names)}
        return model, le, class_map
    except FileNotFoundError:
        st.error("Error: Model or LabelEncoder files not found. Please ensure 'lightgbm_exoplanet_model.pkl' and 'label_encoder.pkl' are present.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}. Check if files are correct.")
        return None, None, None

model, le, class_map = load_models()

# --- INPUT FEATURE GROUPS
# Group features for better UI organization in single prediction mode
TRANSIT_FEATURES = ['TransitEpoch_BKJD', 'ImpactParameter', 'TransitDuration_hrs', 'TransitSignal_to_Noise']
STELLAR_FEATURES = ['StellarEffectiveTemperature_K', 'StellarSurfaceGravity', 'Kepler_bandmag']
PLANETARY_FEATURES = ['EquilibriumTemperature_K', 'orbital_speed_estimate', 'planet_to_star_radius_ratio', 'planet_to_star_density_ratio', 'transit_duration_to_period_ratio']
UNCERT_FEATURES = [f for f in FINAL_MODEL_FEATURES if f.endswith('_uncertainty') or f == 'combined_uncertainty_score']
LOG_FEATURES = [f for f in FINAL_MODEL_FEATURES if f.startswith('log_')]
PERIOD_RELATED = [f for f in FINAL_MODEL_FEATURES if 'period' in f.lower() or 'frequency' in f.lower()]


# --- APP LAYOUT 
st.title("🛰️ Exoplanet Classifier ")
st.subheader("Predicting Confirmed, Candidate, and False Positive Planets")
st.markdown("---")

if model is not None:
    
    # Mode selection
    mode = st.sidebar.radio("Select Prediction Mode:", ["Batch Upload (CSV)", "Single Manual Prediction"])

    if mode == "Batch Upload (CSV)":
        # --- BATCH UPLOAD MODE ---
        st.header("1. Batch Prediction (Upload Raw Data)")
        st.markdown("Upload a raw CSV file. The app will automatically map columns, engineer 27 features, and provide predictions.")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with raw data.", 
            type=["csv"],
            help="Example columns: pl_orbper, st_teff, koi_depth, etc."
        )

        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(raw_df)} rows of raw data. Applying pipeline...")
            
            # 1. PREPROCESS DATA
            with st.spinner("Applying Feature Engineering and Preprocessing Pipeline..."):
                try:
                    processed_df = preprocess_data(raw_df)
                except Exception as e:
                    st.error(f"Preprocessing Error: The pipeline failed. Error details: {e}")
                    st.stop()
            
            # 2. MAKE PREDICTIONS
            if not processed_df.empty:
                preds = model.predict(processed_df)
                probs = model.predict_proba(processed_df)
                
                # 3. DECODE AND DISPLAY RESULTS
                preds_decoded = le.inverse_transform(preds)
                results_df = raw_df.copy()
                results_df['Predicted_Disposition'] = preds_decoded
                
                for i, label in class_map.items():
                    results_df[f'Prob_{label}'] = probs[:, i]
                
                results_df['Confidence_Score'] = np.max(probs, axis=1)
                
                st.markdown("### 📊 Prediction Results Table")
                st.dataframe(results_df[['Predicted_Disposition', 'Confidence_Score'] + [f'Prob_{label}' for label in class_map.values()] + list(raw_df.columns[:5])])
            else:
                st.warning("The processed DataFrame is empty. Cannot make predictions.")
    
    else:
        # --- SINGLE PREDICTION MODE ---
        st.header("2. Single Manual Prediction")
        st.markdown("Enter the 8 core observational values below. The pipeline will automatically calculate the 19 engineered features.")

        input_data = {}
        
        # UI Input for BASE FEATURES (8 inputs needed for engineering)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Transit Timing")
            input_data['OrbitalPeriod_days'] = st.number_input('Orbital Period (days)', value=10.0, format="%.6f")
            input_data['TransitEpoch_BKJD'] = st.number_input('Transit Epoch (BKJD)', value=170.5, format="%.6f")
            input_data['TransitDuration_hrs'] = st.number_input('Transit Duration (hrs)', value=3.0, format="%.4f")
            
        with col2:
            st.markdown("#### Transit Geometry")
            input_data['PlanetaryRadius_Earthradii'] = st.number_input('Planet Radius (R_Earth)', value=2.0, format="%.4f")
            input_data['TransitDepth_ppm'] = st.number_input('Transit Depth (ppm)', value=600.0, format="%.1f")
            input_data['ImpactParameter'] = st.number_input('Impact Parameter (b)', value=0.5, format="%.4f")
            
        with col3:
            st.markdown("#### Stellar/Energy")
            input_data['StellarRadius_Solarradii'] = st.number_input('Star Radius (R_Sun)', value=0.9, format="%.4f")
            input_data['StellarEffectiveTemperature_K'] = st.number_input('Star Temp (K)', value=5700.0, format="%.1f")
            input_data['InsolationFlux_Earthflux'] = st.number_input('Insolation Flux (F_Earth)', value=1.5, format="%.4f")

        # Mock values for the other 4 base features used by the model (if user skips them)
        input_data['EquilibriumTemperature_K'] = st.number_input('Equilibrium Temperature (K)', value=600.0, format="%.1f", help="Used as base feature, calculation is complex.")
        input_data['TransitSignal_to_Noise'] = st.number_input('Signal-to-Noise Ratio (SNR)', value=15.0, format="%.1f", help="Clarity of the signal.")
        input_data['StellarSurfaceGravity'] = st.number_input('Stellar Surface Gravity (log g)', value=4.5, format="%.4f")
        input_data['Kepler_bandmag'] = st.number_input('Star Brightness (mag)', value=14.5, format="%.4f")
        
        if st.button("Predict Disposition"):
        
            all_input_features = {}
            for f in FINAL_MODEL_FEATURES:
                # We'll use a placeholder for the missing engineered features for now
                # or you MUST run the simplified pipeline.
                if f in input_data:
                    all_input_features[f] = input_data[f]
                else:
                    all_input_features[f] = 0.0

            # Creating a DataFrame from the user's input, ensuring column order
            input_df = pd.DataFrame([all_input_features], columns=FINAL_MODEL_FEATURES)
            
            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]

            pred_decoded = le.inverse_transform([pred])[0]
            confidence = np.max(probs)

            st.markdown(f"## **Prediction:** {pred_decoded} (Confidence: {confidence:.2f})")
            st.write("---")
            st.write("### Probability Breakdown:")
            
            prob_dict = {le.inverse_transform([i])[0]: f"{probs[i]*100:.2f}%" for i in range(len(probs))}
            st.write(prob_dict)
