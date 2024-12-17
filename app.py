import streamlit as st
import pickle
import plotly.graph_objects as go
import numpy as np
from Model.model import *

def add_sidebar():
    st.sidebar.header("🔬 Cell Nuclei Measurements")


    data = get_clean_data()

    slider_labels = [
        ("🧬 Radius (mean)", "radius_mean"),
        ("🌀 Texture (mean)", "texture_mean"),
        ("📏 Perimeter (mean)", "perimeter_mean"),
        ("📐 Area (mean)", "area_mean"),
        ("✨ Smoothness (mean)", "smoothness_mean"),
        ("📉 Compactness (mean)", "compactness_mean"),
        ("⬇️ Concavity (mean)", "concavity_mean"),
        ("📍 Concave points (mean)", "concave points_mean"),
        ("🔄 Symmetry (mean)", "symmetry_mean"),
        ("🌌 Fractal dimension (mean)", "fractal_dimension_mean"),
        ("🧬 Radius (se)", "radius_se"),
        ("🌀 Texture (se)", "texture_se"),
        ("📏 Perimeter (se)", "perimeter_se"),
        ("📐 Area (se)", "area_se"),
        ("✨ Smoothness (se)", "smoothness_se"),
        ("📉 Compactness (se)", "compactness_se"),
        ("⬇️ Concavity (se)", "concavity_se"),
        ("📍 Concave points (se)", "concave points_se"),
        ("🔄 Symmetry (se)", "symmetry_se"),
        ("🌌 Fractal dimension (se)", "fractal_dimension_se"),
        ("🧬 Radius (worst)", "radius_worst"),
        ("🌀 Texture (worst)", "texture_worst"),
        ("📏 Perimeter (worst)", "perimeter_worst"),
        ("📐 Area (worst)", "area_worst"),
        ("✨ Smoothness (worst)", "smoothness_worst"),
        ("📉 Compactness (worst)", "compactness_worst"),
        ("⬇️ Concavity (worst)", "concavity_worst"),
        ("📍 Concave points (worst)", "concave points_worst"),
        ("🔄 Symmetry (worst)", "symmetry_worst"),
        ("🌌 Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    # Setting the min,mean,max value

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

# Get vlaue 0 to 1
# Manual
def get_scaled_values(input_dict):
    # Fetch the clean data
    data = get_clean_data()

    # Drop the 'diagnosis' column to focus on the features only
    X = data.drop(['diagnosis'], axis=1)

    # Create a dictionary to hold the scaled values
    scaled_dict = {}

    # Loop through the input dictionary to scale each value(Side bar)
    for key, value in input_dict.items():
        # Calculate the min and max for each feature in the dataset
        max_val = X[key].max()
        min_val = X[key].min()

        # Apply Min-Max scaling: (value - min) / (max - min)
        scaled_value = (value - min_val) / (max_val - min_val)

        # Add the scaled value to the dictionary
        scaled_dict[key] = scaled_value

    return scaled_dict




# input the dictionary value(side bar)
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = [
        'Radius', 'Texture', 'Perimeter', 'Area', 
        'Smoothness', 'Compactness', 
        'Concavity', 'Concave Points',
        'Symmetry', 'Fractal Dimension'
    ]

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
        name='Mean Value 🌟' #These are the average values for each of these features across all samples in the dataset.
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error 📊' # The standard error is a measure of the variability or precision of the mean value.
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
        name='Worst Value 🚨' # (most extreme) values
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


# Add Prediction
def add_predictions(input_data):
    model = pickle.load(open("Model/model.pkl", "rb"))
    scaler = pickle.load(open("Model/scaler.pkl", "rb"))

    # Convert Single array with the values
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Scaling the values
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("🔍 Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.success("🟢 Benign ✅")
    else:
        st.error("🔴 Malignant ⚠️")

    # Probability
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])

    st.info("🧑‍⚕️ This app can assist medical professionals in making a diagnosis but should not replace professional medical advice.")

def main():
    st.set_page_config(
        page_title="Breast Cancer AI 🚑",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

 
    input_data = add_sidebar() # Value comes from side bar

    with st.container():
        st.title("Breast Cancer AI 🎗️")
        st.write("🌟 Connect this app to your cytology lab for breast cancer diagnosis using AI.")
        st.info("Adjust the sliders in the sidebar to provide measurement data.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()
