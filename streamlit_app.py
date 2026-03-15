import os
import streamlit as st
import anthropic
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6-20250929",
    "claude-haiku-4-5-20251001",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

OPENAI_MODELS = [
    "gpt-5.4",
    "gpt-5-mini",
    "o3",
    "o3-pro",
]

PROVIDERS = {
    "Anthropic": ANTHROPIC_MODELS,
    "Gemini": GEMINI_MODELS,
    "OpenAI": OPENAI_MODELS,
}


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def stream_anthropic(api_key, model, system_prompt, messages, max_tokens):
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    ) as response:
        for text in response.text_stream:
            yield text


def stream_gemini(api_key, model, system_prompt, messages, max_tokens):
    client = genai.Client(api_key=api_key)
    # Convert messages to Gemini content format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
    )
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            yield chunk.text


def stream_openai(api_key, model, system_prompt, messages, max_tokens):
    client = OpenAI(api_key=api_key)
    oai_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=oai_messages,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


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
            st.error(f"Please enter your {provider.title()} API key in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if provider == "Anthropic":
                    chunks = stream_anthropic(
                        api_key, model, system_prompt,
                        st.session_state.messages, max_tokens,
                    )
                elif provider == "Gemini":
                    chunks = stream_gemini(
                        api_key, model, system_prompt,
                        st.session_state.messages, max_tokens,
                    )
                else:
                    chunks = stream_openai(
                        api_key, model, system_prompt,
                        st.session_state.messages, max_tokens,
                    )
                full_text = st.write_stream(chunks)
            except anthropic.AuthenticationError:
                st.error("Invalid Anthropic API key.")
                st.stop()
            except anthropic.APIError as e:
                st.error(f"Anthropic API error: {e.message}")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text}
        )


if __name__ == "__main__":
    main()
