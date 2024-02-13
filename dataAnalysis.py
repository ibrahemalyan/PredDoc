import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple,NoReturn, Optional
import plotly.express as px


def data_explore(df: pd.DataFrame) -> NoReturn:
    print("The shape of the data: ", df.shape())

    print("preview of the dataset")
    df.head()

    categorial_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
    continious_cols = ["age","trtbps","chol","thalachh","oldpeak"]
    target_col = ["output"]
    print("The categorial cols are : ", categorial_cols)
    print("The continuous cols are : ", continious_cols)
    print("The target variable is :  ", target_col)

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float) \
     -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = X.sample(frac=train_proportion)
    test = X.loc[X.index.difference(train.index)]
    return train, y.loc[train.index], test, y.loc[test.index]


def preprocess_data(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    output_col = X.drop("output",axis=1)
    categorial_col = ["sex", "exng", "caa", "cp", "fps", "restecg", "slp", "thall"]
    X = X.get_dummies(X,columns=categorial_col,drop_First=True)
    return X,output_col

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    for i in X:
        rho = np.cov(X[i], y)[0, 1] / (np.std(X[i]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[i], 'y': y}), x="x", y="y", trendline="ols",
                         color_discrete_sequence=["black"],
                         title=f"Pearson Correlation Between {i} Values and Response <br> {rho}",
                         labels={"x": f"{i} Values", "y": "Response"})
        fig.write_image(output_path + f"/pearson_correlation_{i}_feature.png")

        corr_matrix = X.corr()

        corr_df = corr_matrix.stack().reset_index()
        corr_df.columns = ['var1', 'var2', 'corr']

        fig = px.imshow(corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                labels=dict(x="Variable", y="Variable", color="Correlation"),color_continuous_scale='RdBu')

        fig.update_layout(width=800, height=800, title='Correlation Matrix Heatmap')

        fig.update_layout(width=800, height=800, title='Correlation Matrix Heatmap',
                  xaxis=dict(side='top'), yaxis=dict(side='left'),
                  margin=dict(l=200, r=200, t=100, b=100),
                  coloraxis_colorbar=dict(title='Correlation'),
                  layout_coloraxis_showscale=True,
                  layout_coloraxis_colorbar_len=0.75, layout_coloraxis_colorbar_y=0.45,
                  layout_coloraxis_colorbar_x=1.1,
                  layout_coloraxis_colorbar_bgcolor='rgba(0,0,0,0)',
                  layout_coloraxis_colorbar_tickfont_size=10,
                  layout_coloraxis_colorbar_title_font_size=12,
                  xgap=2, ygap=2)
        fig.write_image(output_path + "/correlation_matrix.png")

if __name__ == '__main__':
    df = pd.read_csv("./Dataset/heart.csv.xls")

