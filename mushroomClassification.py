#%%
import streamlit as st
import pandas as pd
import numpy as np


#%%
import sklearn
from sklearn.svm import SVC   ##Support Vector Modifier是一种支持向量机的分类器模型。它有许多可调节的参数，
from sklearn.linear_model import LogisticRegression ##罗辑回归
from sklearn.ensemble import RandomForestClassifier ##随机森林
from sklearn.preprocessing import LabelEncoder ##对数据进行编码标签eg.小狗为01小猫02 so on
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

##remember to run pip freeze > requirements.txt in terminal to generate requirements.txt!!!
#%%
def main():
    st.title('binary classification web') ##