import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.title('🤗 Text Summarizer: https://www.linkedin.com/in/ibrahim-sobh-phd-8681757/')


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
st.write(tokenizer.decode(outputs[0]))


# summarizer = pipeline(
#     "summarization",
#     "pszemraj/led-base-book-summary",
#     device=0 if torch.cuda.is_available() else -1,
# )

# st.write('summarizer pipeline is loaded')


# result = []
# with st.form('summarize_form', clear_on_submit=False):
#     txt_input = st.text_area('Enter text:', '', height=200)
#     submitted = st.form_submit_button('Submit')
#     if submitted: 
#         with st.spinner('Calculating...'):
#             summary_txt = summarizer(
#                 txt_input,
#                 min_length=8,
#                 max_length=64,
#                 no_repeat_ngram_size=2,
#                 encoder_no_repeat_ngram_size=2,
#                 #repetition_penalty=3.3,
#                 repetition_penalty=1.3,
#                 num_beams=5,
#                 do_sample=False,
#                 early_stopping=True,
#             )
#             result.append(summary_txt)

# if len(result):
#     st.info(result[0][0]["summary_text"])            
