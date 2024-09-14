import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.3, random_state=42)

# Sidebar inputs for Random Forest hyperparameters
st.sidebar.title("Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

# RandomForest model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Species descriptions and images
species_descriptions = {
    0: ("Setosa: Small flowers with relatively small petals and sepals. Colorful, but not as tall.",
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flax_Flowers.jpg"),
    1: ("Versicolor: Medium-sized flowers with larger petals and sepals. Not as colorful as Setosa.",
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"),
    2: ("Virginica: Larger flowers with broad petals and sepals. Tall with vivid colors.",
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica_2006.jpg")
}

# App title and description
st.title("Iris Flower Classification")
st.write("""
This app classifies Iris flower species based on sepal and petal measurements using a Random Forest Classifier.
Adjust the sliders on the left to input the measurements and click "Predict" to see the results.
""")

# Sidebar inputs for feature values
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
    st.write(species_descriptions[prediction[0]][0])

    # Display species image
    st.image(species_descriptions[prediction[0]][1], caption=predicted_species)

# Model accuracy display
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Correlation heatmap of the features
st.write("### Feature Correlation Heatmap")
corr_matrix = df.iloc[:, :-1].corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

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
3. Below, you will see a description and an image of the predicted species, as well as a visualization of the training data and a feature correlation heatmap.
""")
