import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set custom CSS for background color
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f8ff;
    }
    .sidebar .sidebar-content {
        background: #f0f8ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Species descriptions
species_descriptions = {
    0: "Setosa: Small flowers with relatively small petals and sepals. Colorful, but not as tall.",
    1: "Versicolor: Medium-sized flowers with larger petals and sepals. Not as colorful as Setosa.",
    2: "Virginica: Larger flowers with broad petals and sepals. Tall with vivid colors."
}

# App title and description
st.title("Iris Flower Classification")
st.write("""
This app classifies Iris flower species based on sepal and petal measurements using a Random Forest Classifier.
Adjust the sliders on the left to input the measurements and click "Predict" to see the results.
""")

# Sidebar inputs
st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Prediction
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)
    predicted_species = target_names[prediction[0]]
    
    st.write("### Prediction")
    st.write(f"The predicted species is: **{predicted_species}**")

    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame(prediction_prob, columns=target_names, index=['Probability'])
    st.write(prob_df)
    
    st.write("### About the Predicted Species")
    st.write(species_descriptions[prediction[0]])

# Plot the training data
st.write("### Training Data Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='Set1', ax=ax)
ax.set_title('Sepal Length vs Sepal Width')
st.pyplot(fig)

# Instructions
st.write("""
**Instructions:**
1. Use the sliders on the left to adjust the sepal and petal measurements.
2. Click the "Predict" button to see the predicted Iris species and the prediction probabilities.
3. Below, you will see a description and an image of the predicted species, as well as a visualization of the training data.
""")
