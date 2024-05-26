# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:34:42 2024

@author: 91702
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the machine learning models
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.sav", "rb"))

# Set the page configuration
st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction", "Breast Cancer Prediction"],
                           icons=["clipboard2-pulse", "activity", "person-walking", "lungs"],
                           default_index=0,
                           styles={
                               "container": {"padding": "5!important", "background-color": "#fafafa"},
                               "icon": {"color": "black", "font-size": "25px"},
                               "nav-link": {"font-weight": "bold", "color": "#363636", "text-align": "left", "margin": "0px"},
                               "nav-link-selected": {"background-color": "#F63366"},
                           }
                           )

# Breast Cancer Prediction Functions
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv("breastcancer.csv")
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    
    
    with open("style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    
    with st.expander("Enter Patient Details"):
    # Use columns for a better layout
        col1, col2 = st.columns(2)
        
        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
            Glucose = st.number_input('Glucose Level', min_value=0, step=1)
            BloodPressure = st.number_input('Blood Pressure value', min_value=0, step=1)
            SkinThickness = st.number_input('Skin Thickness value', min_value=0, step=1)
        
        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0, step=1)
            BMI = st.number_input('BMI value', min_value=0.0, step=0.1)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value',min_value=0.0, step=0.1)
            Age = st.number_input('Age of the Person', min_value=0, step=1)
    
    diabetes_prediction = ''
    
    # Creating a button with a loading spinner
    if st.button('Diabetes Test Result', use_container_width=True):
        with st.spinner('Predicting...'):
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diab_prediction = diabetes_model.predict([user_input])
    
            if diab_prediction[0] == 1:
                diabetes_prediction = 'The person is diabetic'
            else:
                diabetes_prediction = 'The person is not diabetic'
    
        # Display the prediction result with styling
    st.success(diabetes_prediction, icon="🩺")

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction Using ML")

    # Use an expander to organize the input fields
    with st.expander("Enter Patient Details"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=0, step=1)
            sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: '0' if x == 0 else '1')
            cp = st.selectbox('Chest Pain types', options=[0, 1, 2, 3])
            trestbps = st.number_input('Resting Blood Pressure', min_value=0, step=1)
            chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, step=1)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
            restecg = st.selectbox('Resting Electrocardiographic results',  options=[0, 1])

        with col2:

            thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, step=1)
            exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
            oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, step=0.1)
            slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, step=1)
            ca = st.selectbox('Major vessels colored by flourosopy', options=[0, 1, 2, 3])
            thal = st.selectbox('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', options=[0, 1, 2,3])

    heart_prediction = ''

    # Creating a button for Prediction with a loading spinner
    if st.button('Heart Disease Test Result', use_container_width=True):
        with st.spinner('Predicting...'):
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            heart_data_prediction = heart_disease_model.predict([user_input])

            if heart_data_prediction[0] == 1:
                heart_prediction = 'The person is having heart disease'
            else:
                heart_prediction = 'The person does not have any heart disease'

    # Display the prediction result with styling
    st.success(heart_prediction, icon="❤")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # Use tabs to organize the input fields
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Information", "Voice Measurements", "Vocal Measurements", "Other Measurements"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Input fields for basic information
            pass

        with col2:
            # Input fields for basic information
            pass

    with tab2:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, step=0.01)

        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, step=0.01)

        with col3:
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, step=0.01)

        with col4:
            Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, step=0.01)

        with col5:
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, step=0.000001)

    with tab3:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            RAP = st.number_input('MDVP:RAP', min_value=0.0, step=0.01)

        with col2:
            PPQ = st.number_input('MDVP:PPQ', min_value=0.0, step=0.01)

        with col3:
            DDP = st.number_input('Jitter:DDP', min_value=0.0, step=0.01)

        with col4:
            Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, step=0.01)

        with col5:
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, step=0.01)

        with col1:
            APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, step=0.01)

        with col2:
            APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, step=0.01)

        with col3:
            APQ = st.number_input('MDVP:APQ', min_value=0.0, step=0.01)

        with col4:
            DDA = st.number_input('Shimmer:DDA', min_value=0.0, step=0.01)

        with col5:
            NHR = st.number_input('NHR', min_value=0.0, step=0.01)

    with tab4:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            HNR = st.number_input('HNR', min_value=0.0, step=0.01)

        with col2:
            RPDE = st.number_input('RPDE', min_value=0.0, step=0.01)

        with col3:
            DFA = st.number_input('DFA', min_value=0.0, step=0.01)

        with col4:
            spread1 = st.number_input('spread1', min_value=0.0, step=0.01)

        with col5:
            spread2 = st.number_input('spread2', min_value=0.0, step=0.01)

        with col1:
            D2 = st.number_input('D2', min_value=0.0, step=0.01)

        with col2:
            PPE = st.number_input('PPE', min_value=0.0, step=0.01)

    # Code for Prediction
    parkinsons_diagnosis = ''

    # Creating a button for Prediction with a loading spinner
    if st.button("Parkinson's Test Result", use_container_width=True):
        with st.spinner('Predicting...'):
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"

    # Display the prediction result with styling
    st.success(parkinsons_diagnosis, icon="🧠")

# Breast Cancer Prediction Page
if selected == "Breast Cancer Prediction":
    # Call the main function from the second code block
    main()