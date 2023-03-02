import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import string
from nltk.text import Text

plt.style.use('seaborn-v0_8-pastel')
# warnings.filterwarnings("ignore")

#Data Ingestion

df = pd.read_csv("TwitterHateSpeech.csv", usecols = ['label', 'tweet'])
data_stats={'Observations':df.shape[0],'Features':df.shape[1],'File_size':df.size,
            'Columns':df.columns,'Data_types':df.dtypes,'Null_vals':df.isnull().sum()}
# print(data_stats)
# print('info:',df.info()) 

#Data understanding
# Predictor Attribute
text = df.iloc[:, 1:] #strips of label, just id and tweet
# print(text.tail())

#Target Attributes
label = df.iloc[:, 0:1] #just id and label
# label.tail()

#get number of hate vs normal speech
hateSpeech=df[df['label']==1].shape[0] #w/o .shape[0] is just all hate speech, shape0 gets cols, 1 gets rows
normSpeech=df[df['label']==0].shape[0]

typesSpeech=[hateSpeech,normSpeech]
labels=['Hate Speech','Non-hate speech']
plt.pie(typesSpeech,explode=[0,.1],labels=labels,autopct='%1.1f%%',startangle=-50)
plt.show()

#Data cleaning
#text cleaning

def clean(text):
    """Makes all input text lowercase, cleans of @users, 
    removes punctuation and special characters"""
    newtext=''
    newtext=text.lower()
    newtext=re.sub(r'(@[A-Za-z0-9]+)',"",text)
    newtext=text.translate(str.maketrans('','',string.punctuation))
    newtext=" ".join(e for e in text.split() if e.isalnum())
    return newtext


df['tweet']=df['tweet'].apply(clean)


#pre processing

#tokenizing
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
# nltk.download('punkt')
def tokenize(text): #tweet to list of words
    text=word_tokenize(text)
    return text
# print(df['tweet'])
df['tweet']=df['tweet'].apply(tokenize)
# print('post tokenize')
# print(df['tweet'])


#removing StopWords (words that are useless to task)
from nltk.corpus import stopwords
stop=stopwords.words('english')
#rename function to be better
def StopWords(text): #
    """"removes stop words from text"""
    return ' '.join([word for word in text if word not in (stop)])


df['tweet']=df['tweet'].apply(StopWords)

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

def lemmatize(text): #breaks stuff, fix
    return [lemmatizer.lemmatize(token) for token in text]

# df['tweet']=df['tweet'].apply(lemmatize)

  

#WordCloud
from wordcloud import WordCloud
from wordcloud import STOPWORDS
hate_speech = df[df['label'] == 1]   
comment_words = ''
stopwords = set(STOPWORDS)
for val in hate_speech.tweet:
    #typecaste each val to string
    val = str(val)
    #split the value
    tokens = val.split()
    #Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 12).generate(comment_words)
 
# plot the WordCloud hate speech image                      
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
# plt.show()


normSpeech = df[df['label'] == 0]   
comment_words = ''
stopwords = set(STOPWORDS)
for val in normSpeech.tweet:
    #typecaste each val to string
    val = str(val)
    #split the value
    tokens = val.split()
    #Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 11).generate(comment_words)
 
# plot the WordCloud norm speech image                      
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.title('Norm Speech WordCloud')
# plt.tight_layout(pad = 0)
# plt.show()

#Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=5000)
# list_to_str = [] # remove the list inside tweet cols which was create due to lemm??
#breaks stuff
# for lists in df['tweet']:
#     list_to_str.append(' '.join(map(str, lists)))
# df['tweet'] = list_to_str

corpus=df['tweet']
tfidf_matrix=vectorizer.fit_transform(corpus)
text=tfidf_matrix.toarray()
# print(text.shape)

#Split dataset
from sklearn.model_selection import train_test_split
labels=df.iloc[:,0]
x_train, x_test, y_train, y_test=train_test_split(text,labels,test_size=.3,random_state=0)

print('Training data:',x_train.shape)
print('Testing data:',x_test.shape)

#Build model

