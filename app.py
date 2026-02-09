import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI

from embeddings import Embeddings
from vector_db import VectorDatabase
from semantic_router.route import Route
from semantic_router.router import SemanticRouter
from semantic_router.samples import productsSample, chitchatSample
from reflection import Reflection
from reranker import Reranker
from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(page_title="Trợ lý bán hàng", layout="wide")
st.title("Trợ lý tư vấn sản phẩm")

@st.cache_resource
def init_engine():
    vector_db = VectorDatabase(db_type="mongodb")
    embedding = Embeddings("all-MiniLM-L6-v2", "sentence_transformers")

    router = SemanticRouter(
        embedding,
        [
            Route("products", productsSample),
            Route("chitchat", chitchatSample)
        ]
    )

    llm_client = OpenAI(
        api_key=os.getenv("MEGALLM_API_KEY"),
        base_url="https://ai.megallm.io/v1"
    )

    reflection = Reflection(llm_client)
    reranker = Reranker()

    base_sys_prompt = """Bạn là nhân viên tư vấn bán hàng chuyên nghiệp tại Quang Đạt Phone.
Chỉ sử dụng dữ liệu được cung cấp. Trả lời chính xác giá nếu có."""

    return RAGEngine(
        vector_db,
        embedding,
        router,
        reflection,
        reranker,
        llm_client,
        base_sys_prompt
    )

rag = init_engine()

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": rag.base_sys_prompt}
    ]

# CHAT HISTORY
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT
if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer, route = rag.chat(st.session_state.messages, prompt)
        st.markdown(answer)
        # st.caption(f"Route: {route}")
