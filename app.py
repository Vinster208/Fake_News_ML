import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df_fake["class"]=0
df_true["class"]=1

df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")
df_marge = pd.concat([df_fake, df_true], axis =0 )
df=df_marge.drop(["title", "subject", "date"], axis=1)

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(word_drop)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
def output_lable(n):
    if n == 0:
        st.write("Fake News")
    elif n == 1:
        st.write("Not A Fake News")


def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    if pred_LR[0] == 0:
        st.write("\n\nLR Prediction: Fake News")
    elif pred_LR[0] == 1:
        st.write("\n\nLR Prediction: Not A Fake News")

    if pred_DT[0] == 0:
        st.write("\n\nDT Prediction: Fake News")
    elif pred_DT[0] == 1:
        st.write("\n\nDT Prediction: Not A Fake News")

    if pred_GBC[0] == 0:
        st.write("\n\nGBC Prediction: Fake News")
    elif pred_GBC[0] == 1:
        st.write("\n\nGBC Prediction: Not A Fake News")

    if pred_RFC[0] == 0:
        st.write("\n\nRFC Prediction: Fake News")
    elif pred_RFC[0] == 1:
        st.write("\n\nRFC Prediction: Not A Fake News")





st.title('Fake News Detection')
input_text = st.text_input('Enter news Article')
if st.button('Predict News'):
    manual_testing(input_text)