import streamlit as st
import joblib
import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (required for deployment)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. LOAD MODELS (Use st.cache_resource so they only load once)
@st.cache_resource
def load_models():
    # Downsize to MiniLM for free deployment (uses less RAM than mpnet)
    model_name = 'sentence-transformers/all-MiniLM-L6-v2' 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sbert_model = AutoModel.from_pretrained(model_name)
    
    fraud_model = joblib.load("model/Numerical BERT_fraud_flag_histgb_model.pkl")
    txn_model = joblib.load("model/Numerical BERT_transaction_type_histgb_model.pkl")
    
    # Load the label encoders here so they are cached!
    label_encoders = joblib.load("model/label_encoders.pkl")
    
    return tokenizer, sbert_model, fraud_model, txn_model, label_encoders

# Unpack the returned models and encoders
tokenizer, sbert_model, fraud_model, txn_model, label_encoders = load_models()

# 2. HELPER FUNCTIONS
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def get_sbert_embedding(text):
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = sbert_model(**encoded_input)
    # Mean pooling
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return (sum_embeddings / sum_mask).numpy()

# 3. BUILD THE UI
st.title("UPI Transaction Fraud & Type Classifier")

# Get user input
txn_text = st.text_input("Transaction Description / Notes")
amount = st.number_input("Transaction Amount", min_value=0.0)
sender_age = st.number_input("Sender Age", min_value=18, max_value=100)
receiver_age = st.number_input("Receiver Age", min_value=18, max_value=100)

if st.button("Predict"):
    with st.spinner("Analyzing transaction..."):
        # Process input
        cleaned_text = clean_text(txn_text)
        text_features = get_sbert_embedding(cleaned_text)
        
        # Combine text features with numerical features
        numeric_features = np.array([[amount, sender_age, receiver_age]]) 
        final_features = np.hstack((text_features, numeric_features))
        
        # Predict (Returns the encoded numerical class)
        fraud_pred = fraud_model.predict(final_features)[0]
        txn_pred = txn_model.predict(final_features)[0]
        
        # Inverse transform to get original string labels using your loaded encoders
        fraud_encoder = label_encoders['fraud_flag']
        txn_encoder = label_encoders['transaction_type']
        
        fraud_result_string = fraud_encoder.inverse_transform([fraud_pred])[0]
        txn_result_string = txn_encoder.inverse_transform([txn_pred])[0]
        
        # Display Results using the dynamically decoded strings
        st.subheader("Results:")
        
        # Optional: Add a little color coding for Fraud!
        if fraud_result_string == "Fraud":
            st.error(f"**Fraud Status:** {fraud_result_string} 🚨")
        else:
            st.success(f"**Fraud Status:** {fraud_result_string} ✅")
            
        st.info(f"**Transaction Type:** {txn_result_string}")