import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Application run : stremalit run translation.py
st.title('Translation')

# 두 개의 컬럼 생성
col1, col2 = st.columns(2)

# 첫 번째 양식
with col1:
    st.subheader('번역하기')
    with st.form(key='form1'):
        content = st.text_area(label='영어', height=200)
        submit = st.form_submit_button(label='번역하기')
        if submit:
            with st.spinner('번역 중 ...'):
                model = AutoModelForSeq2SeqLM.from_pretrained("seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko")
                tokenizer = AutoTokenizer.from_pretrained("seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko")
                inputs = tokenizer.encode(content, return_tensors="pt")
                outputs = model.generate(
                        inputs,
                        length_penalty=1.0,
                        max_length=10000,
                        min_length=12,
                        num_beams=20,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=15,)
                # 두 번째 양식
                with col2:
                    st.subheader('번역완료')
                    with st.form(key='form2'):
                        input_placeholder = st.empty()
                        input_placeholder.text_area("한국어", value=tokenizer.decode(outputs[0], skip_special_tokens=True), height=200)
                        submit2 = st.form_submit_button(label='완료')
    

                