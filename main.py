import streamlit as st
import joblib
import os
from utils import process_new
import warnings
warnings.filterwarnings('ignore')

#load the model
model_path_NB = os.path.join(os.getcwd(), r"D:\Data Science Diploma\finalTest\test\NaiveBayes_clf_model.pkl")
model_NB= joblib.load(model_path_NB)
model_path_LR = os.path.join(os.getcwd(), r"D:\Data Science Diploma\finalTest\test\LogisticRegression_clf_model.pkl")
model_LR = joblib.load(model_path_LR)

def tweet_classification():
    #Title
    st.title("Tweeter Classification Prediction - Trump or Trudeau!")
    st.markdown("""
                ***Description***
                + This is a natural language processing project that takes a tweet
                and tells you after cleanin and lemmatizing the input tweet whether 
                it belongs to Donald Trump or Justin Trudeau.
                Enter you tweet below and choose its language and it will tell you who
                the tweet belongs to.""")
    st.markdown('<hr>', unsafe_allow_html=True)

    #Choose model
    model_type = st.selectbox("Choose the Model:", options = ["NaiveBayes", "LogisticRegression"] )
    
    #Input fields
    input_text = st.text_input(label= "Enter your tweet here:")
    input_language = st.selectbox('Langugage', options=['English', 'French'])
    if input_language == "English":
        input_language = 0
    if input_language == "French":
        input_language= 1
 
    st.markdown('<hr>', unsafe_allow_html=True)

    if st.button('Predict the tweeter:'):
        ## Call the function from utils.py to apply the pipeline
        X_processed = process_new(new_text=input_text, language=input_language)

    # Making predictions 
        if model_type=="NaiveBayes":
            prediction = model_NB.predict(X_processed)[0]
        elif model_type=="LogisticRegression":
            prediction = model_LR.predict(X_processed)[0]

        output_mapping = {0: 'Trump tweeted it', 1: 'Trudeau tweeted it'}
        relevant_output = output_mapping.get (prediction, "unknown")

    ## Display Results
        st.success(f'Tweet Prediction is ... {prediction}')
        st.write(prediction, relevant_output)

if __name__ == '__main__':
    ## Call the function
    tweet_classification()
