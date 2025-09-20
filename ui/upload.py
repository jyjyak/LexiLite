import streamlit as st
import pdfplumber

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    return ""
