import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import string
from nltk.text import Text


df = pd.read_csv("TwitterHateSpeech.csv", usecols = ['label', 'tweet'])
# print(df.tail())
# print(f'Number of Observations: {df.shape[0]}')
# print(f'Number of Features: {df.shape[1]}')
# print(df.columns)
# print(df.dtypes)
# print(df.info())
# print(df.size)
#week 8? 
# Predictor Attribute
text = df.iloc[:, 1:] #strips of label, just id and tweet
# print(text.tail())

# target Attribute
label = df.iloc[:, 0:1] #just id and label
# label.tail()

# print(df.isnull().sum()) #gives sum of vals that are null

#get number of hate vs normal speech
hateSpeech=df[df['label']==1].shape[0] #w/o .shape[0] is just all hate speech, shape0 gets cols, 1 rows
normSpeech=df[df['label']==0].shape[0]

typesSpeech=[hateSpeech,normSpeech]
labels=['Hate Speech','Non-hate speech']
plt.pie(typesSpeech,explode=[0,.1],labels=labels,autopct='%1.1f%%',startangle=-50)
plt.show()
#week 9 data cleaning?

#text cleaning

#pre processing

#feature extraction