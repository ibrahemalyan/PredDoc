import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from io import StringIO

st.title("Heart Attack Data Visualizer")
uploaded_file = st.file_uploader("Choose a file")
data_load_state = st.text('Loading data...')

def plot_all_samples(data):
    df = pd.DataFrame(data[:300], columns=[
        "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall",
        "output"
    ])
    df.hist()
    plt.show()
    st.pyplot(plt)

def plot_sample(data):
    """
        ADD CODE HERE
        Operator function - operates on an individual sample and displays an interesting property of the sample.
    """
    df = pd.DataFrame(data[:300], columns=[
        "age","trtbps"
    ])
    df.hist()
    plt.show()
    # st.pyplot(plt)
    st.line_chart(df)


@st.cache
def load_data(nrows,file):
    data = pd.read_csv(file, nrows=nrows)
    return data


if uploaded_file is not None:
    heart_data = load_data(1000,uploaded_file)
    if st.button('Show All Data'):
        st.subheader("Heart Attack data")
        st.write(heart_data)
        plot_all_samples(heart_data)
    else:
        st.write('')
    text_input = st.text_input(
        "Enter some Data (age,sex,cp, trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall"
""

    )

    if text_input:
        st.write(": ", text_input)

