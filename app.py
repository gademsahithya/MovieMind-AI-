from dotenv import load_dotenv
import os
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel
from typing import List, Optional


# ---------------- LOAD ENV VARIABLES ---------------- #

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


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
    font-weight: 700;
    color: #00E5FF;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #A1A1AA;
    margin-bottom: 35px;
}

.stTextArea textarea {
    background-color: #1F2937;
    color: white;
    border-radius: 12px;
    border: 1px solid #374151;
    padding: 10px;
}

.stButton button {
    width: 100%;
    background-color: #00E5FF;
    color: black;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 12px;
    transition: 0.3s ease;
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
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ---------------- #

st.markdown(
    '<div class="title">🎥 MovieMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Movie Information Extractor</div>',
    unsafe_allow_html=True
)


# ---------------- SESSION REQUEST LIMIT ---------------- #

MAX_REQUESTS = 5

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

remaining_requests = MAX_REQUESTS - st.session_state.request_count

st.markdown(
    f'<div class="limit-box">Session Requests Remaining: {remaining_requests}/{MAX_REQUESTS}</div>',
    unsafe_allow_html=True
)


# ---------------- VALIDATE API KEY ---------------- #

if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY not found. Please configure environment variables.")
    st.stop()


# ---------------- MODEL ---------------- #

model = ChatMistralAI(
    model="codestral-latest",
    api_key=MISTRAL_API_KEY,
    temperature=0
)


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


# ---------------- PROMPT TEMPLATE ---------------- #

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract structured movie information from the paragraph.

        Return valid JSON only.

        {format_instructions}
        """
    ),
    (
        "human",
        "{paragraph}"
    )
])


# ---------------- USER INPUT ---------------- #

paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=220,
    placeholder="Paste a movie description or review here..."
)


# ---------------- BUTTON ---------------- #

if st.button("Extract Movie Details"):

    if st.session_state.request_count >= MAX_REQUESTS:
        st.error("Session request limit reached.")

    elif not paragraph.strip():
        st.warning("Please enter a movie paragraph.")

    else:

        st.session_state.request_count += 1

        try:

            with st.spinner("Extracting movie details..."):

                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)

                movie_data = parser.parse(response.content)

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.subheader("📄 Extracted Movie Data")

                st.json(movie_data.model_dump())

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")