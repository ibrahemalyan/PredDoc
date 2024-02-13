import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from io import StringIO
import ml_models


def plot_all_samples(data):
    pass


def plot_sample(data):
    """
        ADD CODE HERE
        Operator function - operates on an individual sample and displays an interesting property of the sample.
    """
    df = pd.DataFrame(data[:300], columns=[
        "age", "trtbps"
    ])
    df.hist()
    plt.show()
    # st.pyplot(plt)
    st.line_chart(df)


@st.cache(allow_output_mutation=True)
def load_data(nrows, file):
    data = pd.read_csv(file, nrows=nrows)
    return data


def show_table(heart_data_to_display):
    st.subheader("Heart Attack data")
    heart_data_to_display.columns = ["AGE", "SEX", "Chest Pain Type", "blood pressure", "Cholestoral",
                                     "fasting blood sugar", "resting", "max heart rate", "Exercise induced angina",
                                     "old peak", "slope",
                                     "Num of major vessels", "Thalium Stress Test", "result"]
    #
    heart_data_to_display["SEX"] = heart_data_to_display["SEX"].replace({0: "male", 1: "female"})
    heart_data_to_display["Chest Pain Type"] = heart_data_to_display["Chest Pain Type"].replace(
        {0: "Typical Angina", 1: " Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"})
    heart_data_to_display["resting"] = heart_data_to_display["resting"].replace(
        {0: "Normal", 2: "Left ventricular hypertrophy", 1: "ST-T wave normality"})
    heart_data_to_display["Exercise induced angina"] = heart_data_to_display["Exercise induced angina"].replace(
        {0: "No", 1: "Yes"})
    st.write(heart_data_to_display)


st.title("PreDoc - Heart Attack Predictor")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    heart_data = load_data(1000, uploaded_file)

    if st.button('Show All Data'):
        show_table(heart_data)

    else:
        st.write('')
age = st.slider('How old are you?', 0, 120, 22)
sex = st.selectbox(
    'Choose Your Sex?',
    ('male', 'female'))
chest_pain = st.selectbox(
    'Choose Your chest pain type?',
    ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))

blood_pressure = st.number_input('Enter Your Blood Pressure')
chol = st.number_input('Enter Your Cholesterol level')
sugar = st.selectbox(
    "fasting blood sugar > 120 mg/dl?",
    ('Yes', 'No'))

restecg = st.selectbox(
    " Resting electrocardiographic results",
    ("Normal", "ST-T wave normality", "Left ventricular hypertrophy"))

maxheart_rete = st.slider('Enter your max heart rate', 0, 200, 80)
peak = st.slider('previous peak', 0, 10, 1)
slop = st.slider('slop', 0, 10, 1)

veseles = st.number_input('Number of major vessels')
thall = st.number_input("Thalium Stress Test result ~ (0,3)")
exng = st.selectbox("Exercise induced angina", ("YES", "NO"))
model_used = st.selectbox(
    'Choose a ML model:?',
    ("SVM Model", "Logistic regression Model", "GBT Model"))

if st.button('Predict'):
    single_data_vector = []

    single_data_vector.append(int(age))  # 1
    if sex == "male":  # 2
        single_data_vector.append(0)
    else:
        single_data_vector.append(1)
    if chest_pain == "Typical Angina":  # 3
        single_data_vector.append(0)
    elif chest_pain == "Atypical Angina":
        single_data_vector.append(1)
    elif chest_pain == "Non-anginal Pain":
        single_data_vector.append(2)
    elif chest_pain == "Asymptomatic":
        single_data_vector.append(3)
    single_data_vector.append(int(blood_pressure))  # 4
    single_data_vector.append(int(chol))  # 5
    if sugar == "Yes":  # 6
        single_data_vector.append(0)
    else:
        single_data_vector.append(1)
    if restecg == "Normal":  # 7
        single_data_vector.append(0)
    elif restecg == "ST-T wave normality":
        single_data_vector.append(1)
    elif restecg == "Left ventricular hypertrophy":
        single_data_vector.append(2)
    single_data_vector.append(int(maxheart_rete))  # 8
    single_data_vector.append(float(peak))  # 9
    single_data_vector.append(float(slop))  # 10
    single_data_vector.append(int(veseles))  # 11
    single_data_vector.append(int(thall))  # 12
    if sugar == "exng":  # 13
        single_data_vector.append(0)
    else:
        single_data_vector.append(1)
    data = np.array(single_data_vector)
    data = data.reshape(1, 13)
    result = ml_models.predict(uploaded_file, data, model_used)
    if result == 1:
        st.write(f'According to the {model_used}, You would be Diagnoesd with Heart Attack 💔')
    else:
        st.write(f'According to the {model_used}, You would not be Diagnosed with Heart Attack ❤️')

if st.button('See Your State according to Others'):
    pass
if st.button('See '):
    pass
