import streamlit as st
import numpy as np
import joblib


def load_trained_model(filepath):
    """Load a trained model from the specified filepath."""
    return joblib.load(filepath)


def get_prediction(model, features):
    """Predict the player's performance based on input features."""
    return model.predict(features)


def main():
    st.title("Player Performance Predictor")

    # Sidebar inputs
    st.sidebar.header("Input Player Attributes")
    
    features = {
        'Movement Reactions': st.sidebar.slider('Movement Reactions', 0, 100, 50),
        'Potential': st.sidebar.slider('Potential', 0, 100, 50),
        'Passing': st.sidebar.slider('Passing', 0, 100, 50),
        'RCM Skill': st.sidebar.slider('Right Center Midfielder (RCM) Skill', 0, 100, 50),
        'CM Skill': st.sidebar.slider('Center Midfielder (CM) Skill', 0, 100, 50),
        'LCM Skill': st.sidebar.slider('Left Center Midfielder (LCM) Skill', 0, 100, 50),
        'Wage in Euros': st.sidebar.slider("Player's Wage in Euros", 0.0, 600000.0, 30000.0),
        'Mentality Composure': st.sidebar.slider('Mentality Composure', 0, 100, 50),
        'RF Skill': st.sidebar.slider('Right Forward (RF) Skill', 0, 100, 50),
        'CF Skill': st.sidebar.slider('Center Forward (CF) Skill', 0, 100, 50),
    }
    
    # Convert features to numpy array
    input_features = np.array([list(features.values())])
    
    # Load model
    model_path = 'trained_model.pkl' 
    model = load_trained_model(model_path)
    
    # Predict performance
    predicted_performance = get_prediction(model, input_features)[0]
    
    # Display the results
    st.subheader("Predicted Player Performance")
    st.markdown(f"<div style='font-size:36px; color:green;'>{predicted_performance}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
