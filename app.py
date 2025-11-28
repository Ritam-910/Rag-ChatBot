import streamlit as st
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from chatbot import CHROMA_PATH, PROMPT_TEMPLATE

st.set_page_config(page_title="RAG ChatBot", page_icon="🤖", layout="wide")
st.markdown(f"""
    <style>
        body {{
            background-color: #000000 ;
            color: white;
        }}
        .chat-container {{
            max-width: 900px;
            margin: auto;
            padding: 20px;
        }}
        .user-bubble {{
            background: #2368ff;
            color: white;
            padding: 12px 18px;
            border-radius: 18px;
            margin-bottom: 10px;
            max-width: 75%;
            margin-left: auto;
            font-size: 17px;
        }}
        .bot-bubble {{
            background: #2a2d37;
            color: white;
            padding: 12px 18px;
            border-radius: 18px;
            margin-bottom: 10px;
            max-width: 75%;
            margin-right: auto;
            font-size: 17px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chat-title {{
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 10px;
            color: white;
        }}
        .chat-subtitle {{
            text-align: center;
            color: white;
            margin-bottom: 25px;
        }}
    </style>
""", unsafe_allow_html=True)

def typing_animation(text, bubble_class):
    placeholder = st.empty()
    typed_text = ""

    for char in text:
        typed_text += char
        placeholder.markdown(
            f"<div class='{bubble_class}'>{typed_text}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.001)

st.markdown("<div class='chat-title'>🤖 RAG ChatBot</div>", unsafe_allow_html=True)
st.markdown("<div class='chat-subtitle'>Ask questions based on your uploaded documents</div>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query_text = st.text_input(
    "Your message:",
    placeholder="Ask me anything from your documents...",
)

if st.button("Send"):
    if query_text.strip():

        st.session_state.chat_history.append({"role": "user", "content": query_text})

        with st.spinner("🔎 Searching your documents..."):
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function
            )

            results = db.similarity_search_with_relevance_scores(query_text, k=3)

        if len(results) == 0 or results[0][1] < 0.3:
            bot_reply = "I could not find relevant information in your uploaded documents."
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

        else:
            context_text = "\n\n".join([doc.page_content for doc, _ in results])

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            final_prompt = prompt_template.invoke(
                {"context": context_text, "question": query_text}
            )

            with st.spinner("🤖 Generating answer..."):
                model = ChatOllama(model="phi3", temperature=0.7)
                response = model.invoke(final_prompt)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.content
            })
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>{chat['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        typing_animation(chat["content"], "bot-bubble")

st.markdown("</div>", unsafe_allow_html=True)

