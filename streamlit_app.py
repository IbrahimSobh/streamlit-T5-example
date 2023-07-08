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

initial_text = "Egypt is a country located in the north of Africa. Its heartland, the Nile River valley and delta. \
Egypt has thousands of years of recorded history. Ancient Egypt was among the earliest civilizations in the world and \
was the site of one of the world’s earliest urban and literate societies.\
Ancient Egypt was the preeminent civilization in the Mediterranean world for almost 30 centuries. \
Egypt has a rich history & culture that dates back thousands of years ago starting with the Pharaohnic culture the Christianity & Islam. \
Egypt is a popular tourist destination with many attractions. Some of the popular tourist destinations are Cairo, Sharm El Sheikh, Hurghada, Luxor, Aswan"  

result = []
with st.form('summarize_form', clear_on_submit=True):
    txt_input = st.text_area('Enter text:', initial_text, height=200)
    submitted = st.form_submit_button('Submit')
    if submitted: 
        with st.spinner('Calculating...'):
            summary_txt = summarizer(
                txt_input,
                min_length=8,
                max_length=32,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,
                num_beams=2,
                do_sample=False,
                early_stopping=True,
            )
            result.append(summary_txt)

if len(result):
    st.info(result[0][0]["summary_text"])            
