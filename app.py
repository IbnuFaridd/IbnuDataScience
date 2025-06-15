import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Prediksi Status Berat Badan",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .debug-info {
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 1rem;
            margin-top: 1rem;
            font-family: monospace;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler with error handling
@st.cache_resource
def load_models():
    try:
        models = {
            'rf': joblib.load('best_random_forest_model.pkl'),
            'dt': joblib.load('best_decision_tree_model.pkl'),
            'knn': joblib.load('best_knn_model.pkl')
        }
        scaler = joblib.load('scaler_3features.pkl')
        
        # Validate models and scaler
        for name, model in models.items():
            if not hasattr(model, 'predict'):
                raise ValueError(f"Model {name} is invalid")
                
        return models['rf'], models['dt'], models['knn'], scaler
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

rf_model, dt_model, knn_model, scaler = load_models()

# Debug: Display model info
st.sidebar.subheader("Model Info")
st.sidebar.write(f"Random Forest features")
st.sidebar.write(f"Decision Tree features")
st.sidebar.write(f"KNN features")

class_mapping = {
    0: 'Normal Weight',
    1: 'Overweight Level I',
    2: 'Overweight Level II',
    3: 'Obesitas Tipe I',
    4: 'Obesitas Tipe II',
    5: 'Obesitas Tipe III',
    6: 'Underweight'
}

def predict_with_debug(model, input_data, scaler=None):
    try:
        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)
        
        # Debug: Display input before scaling
        st.write("### Debug Info")
        with st.expander("View Prediction Process Details"):
            
            
            if scaler:
                # Debug scaling
                scaled_data = scaler.transform(input_array)
                
                
                # Predict
                prediction = model.predict(scaled_data)
                
                # If model has predict_proba
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(scaled_data)
                    st.write("**Probabilities:**", proba)
            else:
                prediction = model.predict(input_array)
                
        return prediction[0]
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main UI
st.title("⚖️ Prediksi Status Berat Badan")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Data")
    age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=30)
    height = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Berat Badan (kg)", min_value=20, max_value=300, value=70)
    
    model_choice = st.selectbox(
        "Pilih Model",
        ["Random Forest", "Decision Tree", "K-Nearest Neighbors"]
    )

with col2:
    st.subheader("Hasil Prediksi")
    
    if st.button("Prediksi"):
        # Select model based on user's choice
        model = {
            "Random Forest": rf_model,
            "Decision Tree": dt_model,
            "K-Nearest Neighbors": knn_model
        }[model_choice]
        
        # Prepare input data
        input_data = [age, height, weight]
        
        # Make prediction
        prediction_idx = predict_with_debug(model, input_data, scaler)
        
        if prediction_idx is not None:
            prediction_class = class_mapping.get(prediction_idx, "Unknown")
            
            # Display result
            st.success(f"Hasil Prediksi: **{prediction_class}**")
            
            # Display probabilities if available
            if hasattr(model, 'predict_proba'):
                scaled_input = scaler.transform(np.array(input_data).reshape(1, -1))
                probabilities = model.predict_proba(scaled_input)[0]
                
                st.subheader("Probabilitas Klasifikasi")
                prob_df = pd.DataFrame({
                    "Kelas": [class_mapping[i] for i in range(len(class_mapping))],
                    "Probabilitas": probabilities
                }).sort_values("Probabilitas", ascending=False)
                
                st.bar_chart(prob_df.set_index("Kelas"))


