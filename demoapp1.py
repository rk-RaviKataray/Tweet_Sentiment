import streamlit as st 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
import pickle
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

with open('tf_idf', 'rb') as training_model:
    tf = pickle.load(training_model)
    
tweets = pd.read_csv('sentiment_tweets3.csv')
tweets.head(20)

tweets.drop(['Unnamed: 0'], axis = 1, inplace = True)

totalTweets = 8000 + 2314
trainIndex, testIndex = list(), list()
for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.98:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = tweets.iloc[trainIndex]
testData = tweets.iloc[testIndex]


st.title(' Welcome to Tweet Classifier')

activity_select = st.sidebar.selectbox(
    'Select Action to be Performed',
    ('Classify Statement', 'Word Cloud Analysis', 'Performance Report', 'Data Report')
)
  
if activity_select == 'Classify Statement':
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Is the person going through depression </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    input_text = st.text_input("input_text","Type Here")

    predicted=""
    if st.button("Predict"):

        input_text1 = [input_text]
        corpus1 = []
        for i in range(0, len(input_text1)):
            review = re.sub('[^a-zA-Z]', ' ', input_text1[i])
            review = re.sub(r'\s+[a-zA-Z]\s+', ' ', input_text1[i])
            review = review.lower()
            review = review.split()

            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus1.append(review)
       
        
        
        M = tf.transform(corpus1)
        predicted = model.predict(M)
        
        if (predicted[0]==0):
            st.write('The person is not going through depression')
        else:
            st.write('The person is going through depression')
              

elif activity_select == 'Word Cloud Analysis':
    html_temp1 = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Word Cloud  </h2>
    </div>
    """
    st.markdown(html_temp1,unsafe_allow_html=True)
    if st.button("Positive words"):
        positive_words = ' '.join(list(tweets[tweets['label'] == 0]['message']))
        positive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(positive_words)
        fig=plt.figure(figsize = (10, 8), facecolor = 'k')
        plt.imshow(positive_wc)
        plt.axis('off'), 
        plt.tight_layout(pad = 0)
        st.pyplot(fig)
    elif st.button("Depressive words"): 
        depressive_words = ' '.join(list(tweets[tweets['label'] == 1]['message']))
        depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Blues").generate(depressive_words)
        fig1=plt.figure(figsize = (10, 8), facecolor = 'k')
        plt.imshow(depressive_wc)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        st.pyplot(fig1)

elif activity_select == 'Performance Report':
    html_temp2 = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Word Cloud  </h2>
    </div>
    """
    st.markdown(html_temp2,unsafe_allow_html=True)    
    if st.button("Generate Model Performance Report"):  
        st.write("Precision:0.961 ")
        st.write("Recall:0.5 ")
        st.write("F-score:0.657 ")
        st.write("Accuracy:0.86 ")



elif activity_select =='Data Report':
    html_temp3 = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Total Data Report  </h2>
    </div>
    """
    st.markdown(html_temp3,unsafe_allow_html=True) 
    st.write(tweets['label'].value_counts())


    html_temp4 = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Training Data Report  </h2>
    </div>
    """
    st.markdown(html_temp4,unsafe_allow_html=True) 
    st.write(trainData['label'].value_counts())   

    html_temp5 = """
    <div style="background-color:tomato;padding:5px">
    <h2 style="color:white;text-align:center;">Testing Data Report  </h2>
    </div>
    """
    st.markdown(html_temp5,unsafe_allow_html=True) 
    st.write(testData['label'].value_counts())      

        
        
  
        
        
        