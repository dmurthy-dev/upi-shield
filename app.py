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
    # Using the original model to match the 768 feature dimensions
    model_name = 'sentence-transformers/all-mpnet-base-v2'
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
st.write("Enter the transaction description below to analyze it for potential fraud and categorize the transaction type.")

# Get user input
txn_text = st.text_input("Transaction Description / Notes")

if st.button("Predict"):
    if not txn_text.strip():
        st.warning("Please enter a transaction description first.")
    else:
        with st.spinner("Analyzing transaction..."):
            # Process input
            cleaned_text = clean_text(txn_text)
            
            # SBERT outputs a 2D array of shape (1, 768)
            text_features = get_sbert_embedding(cleaned_text)
            
            # Use ONLY the text features to match the 768 requirement
            final_features = text_features
            
            # Predict (Returns the encoded numerical class)
            fraud_pred = fraud_model.predict(final_features)[0]
            txn_pred = txn_model.predict(final_features)[0]
            
            # Inverse transform to get original string labels using your loaded encoders
            fraud_encoder = label_encoders['fraud_flag']
            txn_encoder = label_encoders['transaction_type']
            
            # Convert to string to safely handle both text ("Normal") and numeric ("0") label variants
            fraud_result_string = str(fraud_encoder.inverse_transform([fraud_pred])[0])
            txn_result_string = str(txn_encoder.inverse_transform([txn_pred])[0])
            
            # Display Results
            st.subheader("Results:")
            
            # Catch the '0' or '1' and force it to display the correct word
            if fraud_result_string == "0" or fraud_result_string.lower() == "normal":
                st.success("**Fraud Status:** Normal ✅")
            else:
                st.error("**Fraud Status:** Fraud 🚨")
                
            st.info(f"**Transaction Type:** {txn_result_string}")
