import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
plt.style.use('seaborn-v0_8-pastel')

import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#Data Ingestion
df = pd.read_csv("TwitterHateSpeech.csv", usecols = ['label', 'tweet'])
data_stats={'Observations':df.shape[0],'Features':df.shape[1],'File_size':df.size,
            'Columns':df.columns,'Data_types':df.dtypes,'Null_vals':df.isnull().sum()}

#Data understanding
#Predictor Attributes
text = df.iloc[:, 1:]
#Target Attributes
label = df.iloc[:, 0:1]

#Data Cleaning
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'ð','',text)
    text = re.sub(r'â','',text)
    text = re.sub(r'user','',text)
    text = re.sub(r'amp','',text)
    text=re.sub(r'(@[A-Za-z0-9]+)',"",text)
    text=text.translate(str.maketrans('','',string.punctuation)) 
    text=" ".join(e for e in text.split() if e.isalnum())

    return text
df['tweet'] = df['tweet'].apply(data_processing)


#Pre-Processing
#Tokenizing
from nltk.tokenize import word_tokenize
import nltk
def tokenize(text):
    text=word_tokenize(text)
    return text
df['tweet']=df['tweet'].apply(tokenize)

#Remove StopWords
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
def StopWords(tokens):
    filtered_tokens = [token for token in tokens if token.lower() not in stop]
    return " ".join(filtered_tokens)
df['tweet']=df['tweet'].apply(StopWords)

#Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatized(data): 
    tweet = [lemmatizer.lemmatize(word) for word in data]
    return data
df['tweet'] = df['tweet'].apply(lambda x: lemmatized(x))

#Pie Chart
fig = plt.figure(figsize=(7,7))
hateSpeech=df[df['label']==1].shape[0]
normSpeech=df[df['label']==0].shape[0]
typesSpeech=[hateSpeech,normSpeech]
labels=['Hate Speech','Non-hate speech']
plt.pie(typesSpeech,explode=[0,.1],labels=labels,autopct='%1.0f%%',startangle=-50)
plt.show()

#WordCloud
fig,axs=plt.subplots(1,2,figsize=(16,8))
normSpeech = df[df.label == 0]
text = ' '.join([word for word in normSpeech['tweet']])
train_cloud_pos = WordCloud(max_words=500, collocations = False, background_color = 'white').generate(text)
hate_speech = df[df.label == 1]
text = ' '.join([word for word in hate_speech['tweet']])
train_cloud_neg= WordCloud(max_words=500, collocations = False, background_color = 'black').generate(text)
axs[0].imshow(train_cloud_pos, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Non-Hate Comments')
axs[1].imshow(train_cloud_neg, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Hate Comments')
plt.show()

#Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000,ngram_range=(1,2)).fit(df['tweet'])

x= df['tweet']
y= df['label']
x= vect.transform(x) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
y_pred_test = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

#Confusion Matrix:
confusion_matrix = confusion_matrix(y_test, y_pred_test)
TP = confusion_matrix[1, 1]        
TN = confusion_matrix[0, 0]           
FP = confusion_matrix[0, 1]           
FN = confusion_matrix[1, 0]
group_names = ['TN','FP','FN','TP']
group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Greens')
plt.figure(figsize = (12, 5))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, roc_auc_score
#Accuracy Score
Accuracy = accuracy_score(y_test, y_pred_test)
print('Accuracy Score:', Accuracy) 

# Precision Score
Precision = precision_score(y_test, y_pred_test)
print('Precision Score:', Precision)   

# True positive Rate (TPR)/Sensitivity/Recall
TPR = recall_score(y_test, y_pred_test)
print('True positive Rate:', TPR)             

# False positive Rate (FPR)
FPR = FP/float(TN+FP)
print('False positive Rate', FPR)                       

#F1 Score or F-Measure or F-Score
F1 = f1_score(y_test, y_pred_test)
print('F1 Score:', F1)                 

#Specificity
Specificity = TN/(TN+FP)
print('Specificity:', Specificity )                    

#Mean Absolute Error
Error = mean_absolute_error(y_test, y_pred_test)
print('Mean Absolute Error:', Error)   

#ROC Area
Roc = roc_auc_score(y_test, y_pred_test)
print('ROC Area:', Roc) 

result = [Accuracy, Precision, TPR, FPR, F1, Specificity, Error, Roc]
label = ["Accuracy", "Precision", "TPR", "FPR", "F-Score", "Specificity", "Error", "Roc Area"]
colors=[ 'red', 'green', 'blue', 'darkgoldenrod', 'orange', 'purple', 'brown', 'darkcyan']
plt.bar(label, result, color = colors, edgecolor='black')
plt.show()