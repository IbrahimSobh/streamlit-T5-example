import streamlit as st
import torch
from transformers import pipeline

# """
# # Welcome to Streamlit!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:
# """

st.title('🤗 Text Summarizer')

summarizer = pipeline(
    "summarization",
    "pszemraj/led-base-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)

st.write('summarizer pipeline is loaded')


# with st.form('my_form'):
#   text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#   submitted = st.form_submit_button('Submit')
#   if not openai_api_key.startswith('sk-'):
#     st.warning('Please enter your OpenAI API key!', icon='⚠')
#   if submitted and openai_api_key.startswith('sk-'):
#     generate_response(text)


#txt_input = st.text_area('Enter your text', '', height=200)

wall_of_text = "The majority of available text summarization datasets include short-form source documents that lack long-range causal and temporal dependencies, and often contain strong layout and stylistic biases. While relevant, such datasets will offer limited challenges for future generations of text summarization systems. We address these issues by introducing BookSum, a collection of datasets for long-form narrative summarization. Our dataset covers source documents from the literature domain, such as novels, plays and stories, and includes highly abstractive, human written summaries on three levels of granularity of increasing difficulty: paragraph-, chapter-, and book-level. The domain and structure of our dataset poses a unique set of challenges for summarization systems, which include: processing very long documents, non-trivial causal and temporal dependencies, and rich discourse structures. To facilitate future work, we trained and evaluated multiple extractive and abstractive summarization models as baselines for our dataset."
sum_res = ''
with st.form('summarize_form', clear_on_submit=True):
    txt_input = st.text_area('Enter text:', wall_of_text, height=200)
    submitted = st.form_submit_button('Submit')
    if submitted: 
        with st.spinner('Calculating...'):
            summary_txt = summarizer(
                txt_input,
                min_length=8,
                max_length=64,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,
                num_beams=2,
                do_sample=False,
                early_stopping=True,
            )
            if summary_txt:
                sum_res = summary_txt[0]["summary_text"]
            

if sum_res: 
    st.info(sum_res)
else:
    st.info('No generated summary!')

#st.info(summary_txt[0]["summary_text"])
#st.info('test')            


#----------------------------------------------------------------


# st.title('🎈 App Name')
# st.write('Hello world!')
# hf_name = "pszemraj/led-base-book-summary"
# st.write('Is cuda available?' ,torch.cuda.is_available())

# summarizer = pipeline(
#     "summarization",
#     hf_name,
#     device=0 if torch.cuda.is_available() else -1,
# )

# st.write('summarizer pipeline is loaded')

# wall_of_text = "The majority of available text summarization datasets include short-form source documents that lack long-range causal and temporal dependencies, and often contain strong layout and stylistic biases. While relevant, such datasets will offer limited challenges for future generations of text summarization systems. We address these issues by introducing BookSum, a collection of datasets for long-form narrative summarization. Our dataset covers source documents from the literature domain, such as novels, plays and stories, and includes highly abstractive, human written summaries on three levels of granularity of increasing difficulty: paragraph-, chapter-, and book-level. The domain and structure of our dataset poses a unique set of challenges for summarization systems, which include: processing very long documents, non-trivial causal and temporal dependencies, and rich discourse structures. To facilitate future work, we trained and evaluated multiple extractive and abstractive summarization models as baselines for our dataset."

# result = summarizer(
#     wall_of_text,
#     min_length=8,
#     max_length=64,
#     no_repeat_ngram_size=3,
#     encoder_no_repeat_ngram_size=3,
#     repetition_penalty=3.5,
#     num_beams=2,
#     do_sample=False,
#     early_stopping=True,
# )
# #print(result[0]["generated_text"])
# st.write('Summarizing ...')
# st.write(result[0]["summary_text"])


# with st.echo(code_location='below'):
#     total_points = st.slider("Sobh: Number of points in spiral", 1, 5000, 500)
#     num_turns = st.slider("Sobh: Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))
