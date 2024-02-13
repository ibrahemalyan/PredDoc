import random
from typing import Tuple, NoReturn
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import math


def explore_data(df: pd.DataFrame, cat_col, con_col, targ_col) -> NoReturn:
    print("The shape of the data: ", df.shape)

    print("preview of the dataset, first 5 rows")
    print(df.head(n=5))
    print("The categorical cols are : ", cat_col)
    print("The continuous cols are : ", con_col)
    print("The target variable is :  ", targ_col)


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = X.sample(frac=train_proportion)
    test = X.loc[X.index.difference(train.index)]
    return train, y.loc[train.index], test, y.loc[test.index]


def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    y = y.dropna()
    X = X.loc[y.index]
    X.dropna().drop_duplicates()
    X = X.drop("output", axis=1)
    y = y.loc[X.index]
    # X = X.get_dummies(X, columns=categorial_col, drop_First=True)
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, con_cols, output_path: str = ".") -> NoReturn:
    for i in con_cols:
        rho = np.cov(X[i], y)[0, 1] / (np.std(X[i]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[i], 'y': y}), x="x", y="y", trendline="ols",
                         color_discrete_sequence=["black"],
                         title=f"Pearson Correlation Between {i} Values and Response <br> {rho}",
                         labels={"x": f"{i} Values", "y": "Response"})
        fig.write_image(output_path + f"/pearson_correlation_{i}_feature.png")

        df_corr = X[con_cols].corr()
        fig = px.imshow(df_corr[con_cols])
        fig.update_layout(width=800, height=800, title='Correlation Matrix Heatmap',
                          xaxis_title='Features', yaxis_title='Features')
        fig.write_image(output_path + '/correlation_matrix.png')


def svm_model(train_X, train_y, test_X, test_y, test):
    clf = SVC(kernel='linear', C=1, random_state=42).fit(train_X, train_y)

    # predicting the values
    pred_y = clf.predict(test)
    # printing the test accuracy
    # random_test_y = random.randint(0, 9)
    # print("The test accuracy score of SVM is ", accuracy_score(test_y[random_test_y:random_test_y + 1], pred_y))
    return pred_y.item(0)


def log_reg_model(X_train, y_train, X_test, y_test, test):
    # instantiating the object
    logreg = LogisticRegression(max_iter=1000)

    # fitting the object
    logreg.fit(X_train, y_train)

    # calculating the probabilities
    y_pred_proba = logreg.predict(test)

    return y_pred_proba.item(0)

    # finding the predicted valued
    # y_pred = np.argmax(y_pred_proba, axis=1)
    # printing the test accuracy
    # print("The test accuracy score of Logistric Regression is ", accuracy_score(y_test, y_pred))


def gbd_model(X_train, y_train, X_test, y_test, test):
    gbt = GradientBoostingClassifier(n_estimators=300, max_depth=1, subsample=0.8, max_features=0.2, random_state=42)

    # fitting the model
    gbt.fit(X_train, y_train)

    # predicting values
    y_pred = gbt.predict(test)
    return y_pred.item(0)
    # print("The test accuracy score of Gradient Boosting Classifier is ", accuracy_score(y_test, y_pred))


def predict(file, test, model):
    df = pd.read_csv(file)
    categories_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
    continuous_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
    target_col = ["output"]
    explore_data(df, categories_cols, continuous_cols, target_col)

    train_X, train_y, test_X, test_y = split_train_test(df, df.output, train_proportion=0.8)
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X, test_y = preprocess_data(test_X, test_y)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    feature_evaluation(train_X, train_y, continuous_cols, output_path="./analysis-graphs")  # TODO graphs prob
    if model == "SVM Model":
        return svm_model(train_X, train_y, test_X, test_y, test)
    elif model == "Logistic regression Model":
        return log_reg_model(train_X, train_y, test_X, test_y, test)
    else:
        return gbd_model(train_X, train_y, test_X, test_y,test)
    # return svm_model(train_X, train_y, test_X, test_y, test)

# age - Age of the patient
#
# sex - Sex of the patient
#
# cp - Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
#
# trtbps - Resting blood pressure (in mm Hg)
#
# chol - Cholestoral in mg/dl fetched via BMI sensor
#
# fbs - (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False
#
# restecg - Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy
#
# thalachh - Maximum heart rate achieved
#
# oldpeak - Previous peak
#
# slp - Slope
#
# caa - Number of major vessels
#
# thall - Thalium Stress Test result ~ (0,3)
#
# exng - Exercise induced angina ~ 1 = Yes, 0 = No
#
# output - Target variable
