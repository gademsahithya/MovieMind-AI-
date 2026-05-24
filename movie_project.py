from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="MovieMind AI",
    page_icon="🎥",
    layout="centered"
)


# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #00E5FF;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #BBBBBB;
    margin-bottom: 35px;
}

.stTextArea textarea {
    background-color: #1F2937;
    color: white;
    border-radius: 12px;
    border: 1px solid #444;
}

.stButton button {
    width: 100%;
    background-color: #00E5FF;
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 12px;
}

.stButton button:hover {
    background-color: #00c8e0;
}

.result-box {
    background-color: #161B22;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #30363D;
    margin-top: 20px;
}

.limit-box {
    background-color: #1F2937;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
    color: #E5E7EB;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ---------------- #

st.markdown('<div class="title">🎥 MovieMind AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI Powered Movie Information Extractor</div>',
    unsafe_allow_html=True
)


# ---------------- REQUEST LIMIT ---------------- #

MAX_REQUESTS = 5

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

remaining = MAX_REQUESTS - st.session_state.request_count

st.markdown(
    f'<div class="limit-box">Remaining Requests: {remaining}/{MAX_REQUESTS}</div>',
    unsafe_allow_html=True
)


# ---------------- MODEL ---------------- #

model = ChatMistralAI(model="codestral-latest")


# ---------------- PYDANTIC SCHEMA ---------------- #

class Movie(BaseModel):
    title: str
    release_date: Optional[int]
    director: Optional[str]
    genre: List[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)


# ---------------- PROMPT ---------------- #

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract the information from the paragraph.

        {format_instructions}
        """
    ),
    ("human", "{paragraph}")
])


# ---------------- INPUT ---------------- #

paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=220,
    placeholder="Paste your movie paragraph here..."
)


# ---------------- BUTTON ---------------- #

if st.button("Extract Movie Details"):

    if st.session_state.request_count >= MAX_REQUESTS:
        st.error("Daily request limit reached.")

    elif paragraph.strip() == "":
        st.warning("Please enter a paragraph.")

    else:

        st.session_state.request_count += 1

        with st.spinner("Extracting details..."):

            final_prompt = prompt.invoke({
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

            movie_data = parser.parse(response.content)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            st.subheader("📄 Extracted JSON")

            st.json(movie_data.model_dump())

            st.markdown('</div>', unsafe_allow_html=True)