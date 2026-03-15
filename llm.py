from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


def get_llm(provider, model, api_key, max_tokens):
    if provider == "Anthropic":
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=max_tokens)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, max_output_tokens=max_tokens,
        )
    else:
        return ChatOpenAI(model=model, api_key=api_key, max_tokens=max_tokens)


def _build_human_message(content, image_data=None):
    if image_data:
        parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}"
                },
            },
            {"type": "text", "text": content},
        ]
        return HumanMessage(content=parts)
    return HumanMessage(content=content)


def stream_response(llm, system_prompt, messages, image_data=None):
    lc_messages = [SystemMessage(content=system_prompt)]
    for i, msg in enumerate(messages):
        is_last = i == len(messages) - 1
        if msg["role"] == "user":
            if is_last and image_data:
                lc_messages.append(_build_human_message(msg["content"], image_data))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    for chunk in llm.stream(lc_messages):
        if chunk.content:
            yield chunk.content
