import base64
import html
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI


load_dotenv()


FONT_PATH = Path(__file__).with_name("Transcity DEMO.otf")


@st.cache_data
def load_transcity_font() -> str:
    font_data = base64.b64encode(FONT_PATH.read_bytes()).decode("ascii")
    return (
        "@font-face {"
        "font-family: 'Transcity';"
        "src: url('data:font/otf;base64," + font_data + "') format('opentype');"
        "font-weight: normal;"
        "font-style: normal;"
        "}"
    )


EMOTION_CONFIG = {
    "1": {
        "label": "Angry",
        "mode": "You are a angry Ai agent . you respond aggressively and impatiently",
        "title": "Angry Mode",
        "theme": {
            "background": "linear-gradient(135deg, #1f0a09 0%, #511313 42%, #8f1d14 100%)",
            "panel": "rgba(33, 12, 11, 0.78)",
            "panel_border": "rgba(255, 130, 92, 0.26)",
            "text": "#fff6f1",
            "muted": "#ffcab4",
            "accent": "#ff6b3d",
            "accent_soft": "rgba(255, 107, 61, 0.16)",
            "user_bubble": "linear-gradient(135deg, #ff8a52 0%, #ff5a36 100%)",
            "assistant_bubble": "rgba(255, 255, 255, 0.08)",
            "shadow": "0 22px 60px rgba(0, 0, 0, 0.32)",
        },
    },
    "2": {
        "label": "Funny",
        "mode": "you are a very Funny Ai agent, you respond with humor and jokes",
        "title": "Funny Mode",
        "theme": {
            "background": "linear-gradient(135deg, #fff4bf 0%, #ffd266 40%, #ff8b5e 100%)",
            "panel": "rgba(255, 249, 232, 0.78)",
            "panel_border": "rgba(255, 152, 0, 0.28)",
            "text": "#38220f",
            "muted": "#704a1a",
            "accent": "#ff9a00",
            "accent_soft": "rgba(255, 154, 0, 0.18)",
            "user_bubble": "linear-gradient(135deg, #ffb53f 0%, #ff8a00 100%)",
            "assistant_bubble": "rgba(255, 255, 255, 0.58)",
            "shadow": "0 22px 60px rgba(140, 92, 0, 0.18)",
        },
    },
    "3": {
        "label": "Sad",
        "mode": "You are a very sad Ai agent, you respond with sadness and sorrow",
        "title": "Sad Mode",
        "theme": {
            "background": "linear-gradient(135deg, #09131f 0%, #173a5e 45%, #4677a8 100%)",
            "panel": "rgba(10, 24, 39, 0.76)",
            "panel_border": "rgba(139, 191, 255, 0.24)",
            "text": "#edf6ff",
            "muted": "#b9d7f2",
            "accent": "#79b8ff",
            "accent_soft": "rgba(121, 184, 255, 0.16)",
            "user_bubble": "linear-gradient(135deg, #7fb4f3 0%, #5d8fdd 100%)",
            "assistant_bubble": "rgba(255, 255, 255, 0.08)",
            "shadow": "0 22px 60px rgba(0, 0, 0, 0.24)",
        },
    },
}


st.set_page_config(
    page_title="Emotion Based Chatbot",
    page_icon="AI",
    layout="wide",
)


@st.cache_resource
def get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-2506",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.7,
        max_tokens=2028,
    )


def apply_theme(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
            {load_transcity_font()}

            :root {{
                --bg: {theme["background"]};
                --panel: {theme["panel"]};
                --panel-border: {theme["panel_border"]};
                --text: {theme["text"]};
                --muted: {theme["muted"]};
                --accent: {theme["accent"]};
                --accent-soft: {theme["accent_soft"]};
                --user-bubble: {theme["user_bubble"]};
                --assistant-bubble: {theme["assistant_bubble"]};
                --shadow: {theme["shadow"]};
            }}

            .stApp,
            [data-testid="stAppViewContainer"] {{
                background: var(--bg);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(0, 0, 0, 0.10);
                border-right: 1px solid var(--panel-border);
            }}

            [data-testid="stSidebar"] * {{
                color: var(--text);
            }}

            [data-testid="stSidebar"] .stRadio > div {{
                gap: 0.95rem;
            }}

            [data-testid="stSidebar"] .stRadio label {{
                display: flex;
                align-items: center;
                min-height: 64px;
                padding: 1rem 1.05rem;
                border-radius: 20px;
                border: 1px solid var(--panel-border);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.04));
                font-size: 1.08rem;
                font-weight: 700;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.10);
                backdrop-filter: blur(12px);
                transition: border-color 120ms ease, transform 120ms ease, background 120ms ease;
            }}

            [data-testid="stSidebar"] .stRadio label:hover {{
                border-color: var(--accent);
                background: linear-gradient(180deg, var(--accent-soft), rgba(255, 255, 255, 0.06));
                transform: translateY(-1px);
            }}

            [data-testid="stSidebar"] .stRadio input {{
                accent-color: var(--accent);
                transform: scale(1.15);
            }}

            [data-testid="stSidebar"] .stMarkdown p {{
                margin-bottom: 0;
            }}

            .sidebar-section-title {{
                margin: 0 0 0.85rem;
                color: var(--muted);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }}

            .sidebar-model-card {{
                margin-top: 1.1rem;
                padding: 1rem;
                border-radius: 18px;
                border: 1px solid var(--panel-border);
                background: rgba(255, 255, 255, 0.06);
                box-shadow: var(--shadow);
            }}

            .sidebar-model-label {{
                color: var(--muted);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}

            .sidebar-model-name {{
                margin-top: 0.4rem;
                font-size: 1.02rem;
                font-weight: 700;
                color: var(--text);
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 8rem;
                max-width: 1080px;
            }}

            footer,
            [data-testid="stBottom"],
            [data-testid="stBottomBlockContainer"],
            [data-testid="stAppFooter"] {{
                background: transparent !important;
            }}

            footer {{
                visibility: hidden;
                height: 0;
            }}

            .topbar {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1rem;
                margin-bottom: 1rem;
                text-align: center;
            }}

            .brand-card {{
                display: inline-flex;
                flex-direction: column;
                align-items: center;
                gap: 0.3rem;
            }}

            .brand-title {{
                margin: 0;
                color: var(--text);
                font-family: "Transcity", "Trebuchet MS", sans-serif;
                font-size: 5.4rem;
                line-height: 0.95;
                letter-spacing: 0.03em;
                text-align: center;
            }}

            .brand-subtitle {{
                margin: 0;
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.4;
                text-align: center;
            }}

            .emotion-badge-wrap {{
                display: flex;
                justify-content: flex-end;
                margin-bottom: 0.9rem;
            }}

            .emotion-badge {{
                min-width: 220px;
                padding: 0.9rem 1rem;
                border-radius: 18px;
                border: 1px solid var(--panel-border);
                background: rgba(255, 255, 255, 0.05);
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.10);
                backdrop-filter: blur(14px);
                text-align: left;
            }}

            .emotion-badge-label {{
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }}

            .emotion-badge-value {{
                margin-top: 0.35rem;
                color: var(--text);
                font-size: 1.02rem;
                font-weight: 700;
            }}

            .chat-shell {{
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 28px;
                padding: 1.2rem;
                min-height: 60vh;
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
            }}

            .chat-scroll {{
                display: flex;
                flex-direction: column;
                gap: 0.9rem;
            }}

            .message-row {{
                display: flex;
                width: 100%;
            }}

            .message-row.user {{
                justify-content: flex-end;
            }}

            .message-row.assistant {{
                justify-content: flex-start;
            }}

            .bubble {{
                max-width: min(78%, 760px);
                padding: 0.95rem 1rem;
                border-radius: 20px;
                line-height: 1.65;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
                animation: rise 180ms ease-out;
                word-break: break-word;
            }}

            .bubble-user {{
                background: var(--user-bubble);
                color: #ffffff;
                border-bottom-right-radius: 8px;
            }}

            .bubble-assistant {{
                background: var(--assistant-bubble);
                color: var(--text);
                border: 1px solid var(--panel-border);
                border-bottom-left-radius: 8px;
                backdrop-filter: blur(8px);
            }}

            .typing-shell {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.8rem 1rem;
                border-radius: 20px;
                border-bottom-left-radius: 8px;
                background: var(--assistant-bubble);
                border: 1px solid var(--panel-border);
                color: var(--muted);
                backdrop-filter: blur(8px);
            }}

            .typing-dot {{
                width: 8px;
                height: 8px;
                border-radius: 999px;
                background: var(--accent);
                opacity: 0.32;
                animation: bounce 1.1s infinite;
            }}

            .typing-dot:nth-child(2) {{
                animation-delay: 0.15s;
            }}

            .typing-dot:nth-child(3) {{
                animation-delay: 0.3s;
            }}

            .stButton button {{
                width: 100%;
                min-height: 50px;
                border-radius: 14px;
                border: 1px solid var(--panel-border);
                background: var(--panel);
                color: var(--text);
                font-size: 1rem;
                transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
            }}

            .stButton button:hover {{
                border-color: var(--accent);
                transform: translateY(-1px);
                background: var(--accent-soft);
            }}

            .stChatInput {{
                position: fixed;
                left: 50%;
                bottom: 1.4rem;
                transform: translateX(-50%);
                width: min(920px, calc(100vw - 6rem));
                z-index: 999;
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 24px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
                padding: 0.4rem 0.5rem;
            }}

            .stChatInput > div {{
                background: transparent;
            }}

            .stChatInput textarea {{
                color: var(--text) !important;
                font-size: 1rem !important;
            }}

            .stChatInput textarea::placeholder {{
                color: var(--muted) !important;
            }}

            .stChatInput button {{
                border-radius: 16px !important;
                background: var(--accent) !important;
                color: white !important;
            }}

            @keyframes bounce {{
                0%, 80%, 100% {{ transform: translateY(0); opacity: 0.32; }}
                40% {{ transform: translateY(-5px); opacity: 1; }}
            }}

            @keyframes rise {{
                from {{ transform: translateY(4px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}

            @media (max-width: 900px) {{
                .topbar {{
                    flex-direction: column;
                    align-items: center;
                }}

                .brand-title {{
                    font-size: 3.8rem;
                }}

                .emotion-badge-wrap {{
                    justify-content: flex-start;
                }}

                .emotion-badge {{
                    min-width: 0;
                    width: 100%;
                }}

                .chat-shell {{
                    padding: 0.9rem;
                    min-height: 56vh;
                }}

                .bubble {{
                    max-width: 92%;
                }}

                .stChatInput {{
                    width: calc(100vw - 2rem);
                    bottom: 0.8rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bubble(content: str, role: str) -> str:
    safe_content = html.escape(content).replace("\n", "<br>")
    bubble_class = "bubble-user" if role == "user" else "bubble-assistant"
    return f'<div class="message-row {role}"><div class="bubble {bubble_class}">{safe_content}</div></div>'


def render_typing_indicator() -> str:
    return """
    <div class="message-row assistant">
        <div class="typing-shell">
            <span>Thinking</span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    </div>
    """


def render_chat_shell(chat_history: list[dict[str, str]], typing_html: str = "") -> str:
    messages_html = "".join(
        render_bubble(message["content"], message["role"]) for message in chat_history
    )
    if not messages_html:
        messages_html = '<div class="message-row assistant"><div class="bubble bubble-assistant">Start the conversation.</div></div>'
    return f'<div class="chat-shell"><div class="chat-scroll">{messages_html}{typing_html}</div></div>'


def initialize_chat(choice: str) -> None:
    mode = EMOTION_CONFIG[choice]["mode"]
    st.session_state.active_choice = choice
    st.session_state.messages = [SystemMessage(content=mode)]
    st.session_state.chat_history = []
    st.session_state.chat_closed = False


def normalize_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)


if "active_choice" not in st.session_state:
    initialize_chat("2")


with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Choose Emotion</div>', unsafe_allow_html=True)
    choice = st.radio(
        "Choose mode",
        options=list(EMOTION_CONFIG.keys()),
        format_func=lambda key: EMOTION_CONFIG[key]["label"],
        horizontal=False,
        label_visibility="collapsed",
    )
    if choice != st.session_state.active_choice:
        initialize_chat(choice)

    st.markdown(
        """
        <div class="sidebar-model-card">
            <div class="sidebar-model-label">LLM Used</div>
            <div class="sidebar-model-name">mistral-small-2506</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation"):
        initialize_chat(choice)
        st.rerun()


config = EMOTION_CONFIG[st.session_state.active_choice]
apply_theme(config["theme"])

st.markdown(
    """
    <div class="topbar">
        <div class="brand-card">
            <h1 class="brand-title">Emo</h1>
            <p class="brand-subtitle">a emotion based chatbot</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "MISTRAL_API_KEY" not in os.environ or not os.getenv("MISTRAL_API_KEY"):
    st.error("`MISTRAL_API_KEY` is missing from your environment. Add it to `.env` to run the chatbot.")
    st.stop()


st.markdown(
    f"""
    <div class="emotion-badge-wrap">
        <div class="emotion-badge">
            <div class="emotion-badge-label">Current Emotion</div>
            <div class="emotion-badge-value">{config["title"]}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_placeholder = st.empty()
chat_placeholder.markdown(render_chat_shell(st.session_state.chat_history), unsafe_allow_html=True)

prompt = st.chat_input(
    "Type your message here. Use 'exit' to end the conversation.",
    disabled=st.session_state.chat_closed,
)

if st.session_state.chat_closed:
    st.info("This conversation has been closed with `exit`. Use `Clear conversation` or switch emotion to start again.")


if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    chat_placeholder.markdown(render_chat_shell(st.session_state.chat_history), unsafe_allow_html=True)

    if prompt.lower() == "exit":
        farewell = "Goodbye!"
        st.session_state.chat_history.append({"role": "assistant", "content": farewell})
        st.session_state.chat_closed = True
        chat_placeholder.markdown(render_chat_shell(st.session_state.chat_history), unsafe_allow_html=True)
    else:
        try:
            llm = get_llm()
            chat_placeholder.markdown(
                render_chat_shell(st.session_state.chat_history, render_typing_indicator()),
                unsafe_allow_html=True,
            )
            time.sleep(0.8)

            response = llm.invoke(st.session_state.messages)
            response_text = normalize_content(response.content)

            animated_text = ""
            for character in response_text:
                animated_text += character
                preview_history = st.session_state.chat_history + [
                    {"role": "assistant", "content": f"{animated_text}|"}
                ]
                chat_placeholder.markdown(render_chat_shell(preview_history), unsafe_allow_html=True)
                time.sleep(0.012)

            st.session_state.messages.append(AIMessage(content=response_text))
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            chat_placeholder.markdown(render_chat_shell(st.session_state.chat_history), unsafe_allow_html=True)
        except Exception as error:
            error_message = f"Error: {error}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            chat_placeholder.markdown(render_chat_shell(st.session_state.chat_history), unsafe_allow_html=True)

