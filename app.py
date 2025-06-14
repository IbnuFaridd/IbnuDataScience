import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Weight Status Prediction",
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
        .stSelectbox>div>div>select {
            padding: 0.5rem;
            border-radius: 5px;
        }
        .stNumberInput>div>div>input {
            padding: 0.5rem;
            border-radius: 5px;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-card {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('best_random_forest_model.pkl')
        dt_model = joblib.load('best_decision_tree_model.pkl')
        knn_model = joblib.load('best_knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return rf_model, dt_model, knn_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading files: {str(e)}")
        st.stop()

rf_model, dt_model, knn_model, scaler = load_models()

# Get expected number of features from the model
n_features_rf = rf_model.n_features_in_
n_features_dt = dt_model.n_features_in_
n_features_knn = knn_model.n_features_in_

class_labels = {
    0: 'Normal Weight',
    1: 'Overweight Level I',
    2: 'Overweight Level II',
    3: 'Obesitas Tipe I',
    4: 'Obesitas Tipe II',
    5: 'Obesitas Tipe III',
    6: 'Underweight'
}

class_colors = {
    'Normal Weight': '#4CAF50',
    'Overweight Level I': '#FFC107',
    'Overweight Level II': '#FF9800',
    'Obesitas Tipe I': '#F44336',
    'Obesitas Tipe II': '#E91E63',
    'Obesitas Tipe III': '#9C27B0',
    'Underweight': '#2196F3'
}

def predict(model, features, scaler=None):
    features = np.array(features).reshape(1, -1)
    if scaler:
        features = scaler.transform(features)
    return model.predict(features)

# App layout
st.title("⚖️ Prediksi Status Berat Badan")
st.markdown("Aplikasi ini memprediksi kategori berat badan Anda berdasarkan usia, tinggi badan, dan berat badan.")

# Sidebar
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan model machine learning untuk memprediksi status berat badan Anda berdasarkan:
    - Usia
    - Tinggi badan
    - Berat badan
    
    Pilih model yang ingin digunakan dan masukkan data Anda untuk mendapatkan prediksi.
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data Diri Anda")
    with st.container():
        age = st.number_input('Usia (tahun)', min_value=0, max_value=100, value=25)
        height = st.number_input('Tinggi Badan (cm)', min_value=50, max_value=250, value=170)
        weight = st.number_input('Berat Badan (kg)', min_value=30, max_value=200, value=70)

    model_choice = st.selectbox('Pilih Model Prediksi', 
                              ['Random Forest', 'Decision Tree', 'K-Nearest Neighbors'],
                              help="Pilih algoritma machine learning untuk prediksi")

with col2:
    st.subheader("Hasil Prediksi")
    if st.button('Prediksi Status Berat Badan', use_container_width=True):
        # Select model and get expected feature count
        if model_choice == 'Random Forest':
            model = rf_model
            expected_features = n_features_rf
        elif model_choice == 'Decision Tree':
            model = dt_model
            expected_features = n_features_dt
        else:
            model = knn_model
            expected_features = n_features_knn
        
        # Create features array
        if expected_features == 3:
            features = [age, height, weight]
        else:
            st.error(f"Model expects {expected_features} features. Please update the code.")
            st.stop()
        
        try:
            prediction = predict(model, features, scaler)
            predicted_class = class_labels[prediction[0]]
            class_color = class_colors[predicted_class]
            
            # Display prediction card
            st.markdown(f"""
                <div class="prediction-card" style="background-color: {class_color}">
                    <h3 style="color: white; margin: 0;">Hasil Prediksi</h3>
                    <h2 style="color: white; margin: 0;">{predicted_class}</h2>
                    <p style="color: white; margin: 0;">Menggunakan model: {model_choice}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaler.transform(np.array(features).reshape(1, -1)))[0]
                
                st.subheader("Probabilitas Kategori")
                for i, p in enumerate(proba):
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        st.markdown(f"<div style='width: 20px; height: 20px; background-color: {class_colors[class_labels[i]]}; border-radius: 5px;'></div>", 
                                   unsafe_allow_html=True)
                    with cols[1]:
                        st.write(class_labels[i])
                    with cols[2]:
                        st.write(f"{p*100:.1f}%")
                
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")