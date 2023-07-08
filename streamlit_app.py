import streamlit as st
import torch
from transformers import pipeline

st.title('🤗 Text Summarizer: https://www.linkedin.com/in/ibrahim-sobh-phd-8681757/')

summarizer = pipeline(
    "summarization",
    "pszemraj/led-base-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)

st.write('summarizer pipeline is loaded')


result = []
with st.form('summarize_form', clear_on_submit=True):
    txt_input = st.text_area('Enter text:', '', height=200)
    submitted = st.form_submit_button('Submit')
    if submitted: 
        with st.spinner('Calculating...'):
            summary_txt = summarizer(
                txt_input,
                min_length=8,
                max_length=64,
                no_repeat_ngram_size=2,
                encoder_no_repeat_ngram_size=2,
                #repetition_penalty=3.3,
                repetition_penalty=1.3,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
            )
            result.append(summary_txt)

if len(result):
    st.info(result[0][0]["summary_text"])            
