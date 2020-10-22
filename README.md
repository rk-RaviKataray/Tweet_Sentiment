# Tweet_Sentiment
This is an NLP problem statement which is used to detect if a person is going through depression by analyzing the tweets they posts.


As there are a few basic steps to need to be formed on every machine learning problem statement:
 1. Data Loading - Loading the data from
 2. Data Cleaning
 3. Feature engineering
 4. Splitting the data set
 5. Training the model
 6. Testing the Model created using the test data
 
 
 ## Fearure enginnering for NLP 
This being an NLP problem feature engineering is done in this format:
 1. Lowering all the statements in the dataset.
 2. removing STOPWORDS(The words which are of no use in analyzing the statement eg- That, for, in ,is, this, a, am etc )
 3. Stemming  - Reducing the words to its root word. eg- imagination to imagin
    Lemmatization - Reducing the word to its root word which makes more sense eg- imagination to imagine
 4. Vectorisation - As the systems are not intelligent enough to understand the words we need to convert them into numericles
 
While developing this particular app we used Stemming to reduce the words to rootwords and TF_IDF method to vectorize the words.

Front-end of this app was created using streamlit, which is a Library in python which makes developin the frontend easier.

This app was deployed in Heroku which is a PAAS

## Procedure to deploy app in heroku

setup.sh - Which is used to setup the environment required
Prockfile - This file contains the line need to be triggered to run the app
