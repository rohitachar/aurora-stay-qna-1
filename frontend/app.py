
import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("AURORA_BACKEND_URL", "http://localhost:7860").strip().rstrip("/")

st.set_page_config(page_title="Aurora Stays", page_icon="??", layout="wide")

with st.sidebar:
    st.header("Parameters")
    k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=2)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    max_tokens = st.slider("Maximum answer tokens", min_value=64, max_value=2048, value=512, step=64)

st.title("Aurora Stays")
st.write("Ask a question about AuroraStay policies.")
response_type = st.radio("Response mode", ["Answer only", "Answer + Relevant Chunks"], horizontal=True)
query = st.text_input("Your question", placeholder="What time can I check in at ASH-CHI Loop?")

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = "/v1/answer" if response_type == "Answer only" else "/v1/answer_with_relevant_chunks"
        payload = {
            "query": query.strip(),
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=(5, 90),
                )
                response.raise_for_status()
            result = response.json()
            st.subheader("Answer")
            st.write(result["answer"])
            if response_type == "Answer + Relevant Chunks":
                st.subheader("Relevant Chunks")
                for i, chunk in enumerate(result["relevant_chunks"], start=1):
                    with st.expander(f"Chunk {i}"):
                        st.write(chunk)
        except requests.exceptions.RequestException as exc:
            st.error(f"The RAG API request failed: {exc}")
