import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

"""
# Welcome to Streamlit! 🤗 Generaive AI tutorial with Faln-T5 model 

[T5](https://huggingface.co/t5-base) model reframes all NLP tasks into a unified text-to-text-format where the input and output are always text strings.
The same model, loss function, and hyperparameters can be used on any NLP task. [Flan-T5](https://huggingface.co/google/flan-t5-small) is just better at everything!

For more, check out my LinkedIn [Ibrahim Sobh](https://www.linkedin.com/in/ibrahim-sobh-phd-8681757/)

![T5](https://camo.githubusercontent.com/623b4dea0b653f2ad3f36c71ebfe749a677ac0a1/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f343030362f312a44304a31674e51663876727255704b657944387750412e706e67)

"""


st.title('🤗 Text Summarizer: https://www.linkedin.com/in/ibrahim-sobh-phd-8681757/')


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
st.write('flan-t5-small is loaded')


result = []
with st.form('T5_form', clear_on_submit=False):
    txt_input = st.text_area('Enter text:', 'translate English to German: How old are you?', height=50)
    submitted = st.form_submit_button('Submit')
    if submitted: 
        with st.spinner('Calculating...'):
            input_ids = tokenizer(txt_input, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            result.append(outputs)

if len(result):
    st.info(tokenizer.decode(result[0][0]))            



# --------------------------------------------------
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
