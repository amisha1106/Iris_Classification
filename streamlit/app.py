import streamlit as st
import pandas as pd
import numpy as np 

##TITLE of the application
st.title("Hello Streamlit!!!")

## DISPLAY A SIMPLE TEXT
st.write("This is a text")

##DATAFRAME
df=pd.DataFrame({
    'col1':[1,2,3,4,5],
    'col2':[10,20,30,40,50]
})

##Display the dataframe
st.write("Here is the dataframe")
st.write(df)

##create a line chart
chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)