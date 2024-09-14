import streamlit as st
import pandas as pd

st.title("Streamlit text input")
name=st.text_input("Enter your name:")

age=st.slider("select your age:",0,100,25)
st.write(f"Your age is:{age}.")

options=["Python","Java","C","R","C++"]
choice=st.selectbox("Choose your fav programming language:",options)
st.write(f"You selected {choice}.")
if name:
    st.write(f"Hello,{name}")

data={
    "Name":['John','Jane','Jake','Jill'],
    "Age":[28,33,44,22],
    "City":["INdia","USA","UK","NY"]
}
df=pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)

uploaded_file=st.file_uploader("choose a csv file",type="csv")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)