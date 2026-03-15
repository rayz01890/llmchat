import os
import base64
import streamlit as st
from dotenv import load_dotenv
from llm import PROVIDERS, get_llm, stream_response

load_dotenv()

SUPPORTED_TYPES = ["txt", "docx", "xlsx", "jpg", "jpeg"]


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_context" not in st.session_state:
        st.session_state.file_context = None
    if "file_image" not in st.session_state:
        st.session_state.file_image = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None


def extract_file_content(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8"), None
    elif name.endswith(".docx"):
        import docx
        doc = docx.Document(uploaded_file)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text, None
    elif name.endswith(".xlsx"):
        import pandas as pd
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df.to_markdown(index=False), None
    elif name.endswith((".jpg", ".jpeg")):
        data = base64.b64encode(uploaded_file.read()).decode("utf-8")
        return None, {"base64": data, "mime_type": "image/jpeg"}
    return None, None


def main():
    st.set_page_config(page_title="LLM Chat", layout="centered")
    init_state()

    # --- Sidebar ---
    st.sidebar.title("Settings")

    provider = st.sidebar.selectbox("Provider", options=list(PROVIDERS.keys()))
    model = st.sidebar.selectbox("Model", options=PROVIDERS[provider])

    if provider == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    elif provider == "Gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")

    system_prompt = st.sidebar.text_area(
        "System prompt (optional)",
        value="You are a helpful assistant.",
    )
    max_tokens = st.sidebar.slider("Max tokens", 256, 8192, 4096, step=256)

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file",
        type=SUPPORTED_TYPES,
    )
    if uploaded_file:
        if uploaded_file.name != st.session_state.file_name:
            text_content, image_data = extract_file_content(uploaded_file)
            st.session_state.file_context = text_content
            st.session_state.file_image = image_data
            st.session_state.file_name = uploaded_file.name
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        else:
            st.sidebar.success(f"Loaded: {st.session_state.file_name}")
    else:
        st.session_state.file_context = None
        st.session_state.file_image = None
        st.session_state.file_name = None

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    # --- Main area ---
    st.title("LLM Chat")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type a message…"):
        if not api_key:
            st.error(f"Please set your {provider} API key in the .env file.")
            st.stop()

        # Build the user message with file context if present
        display_text = prompt
        if st.session_state.file_context:
            prompt = (
                f"Here is the content of the uploaded file '{st.session_state.file_name}':\n\n"
                f"{st.session_state.file_context}\n\n"
                f"User question: {prompt}"
            )
        elif st.session_state.file_image:
            prompt = (
                f"I've uploaded an image file '{st.session_state.file_name}'. "
                f"{prompt}"
            )

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(display_text)

        with st.chat_message("assistant"):
            try:
                llm = get_llm(provider, model, api_key, max_tokens)
                chunks = stream_response(
                    llm, system_prompt, st.session_state.messages,
                    image_data=st.session_state.file_image,
                )
                full_text = st.write_stream(chunks)
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text}
        )


if __name__ == "__main__":
    main()
