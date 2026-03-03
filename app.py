import streamlit as st
from openai import OpenAI

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Mi IA Personal",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# CSS PARA APARIENCIA DE APP
css_personalizado = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
.stApp {max-width: 100%; padding: 0;}
.stChatMessage {padding: 0.5rem 0;}
</style>
"""
st.markdown(css_personalizado, unsafe_allow_html=True)

# PERSONALIDAD DE TU IA
SYSTEM_PROMPT = """
Eres un asistente IA personalizado.
Tu tono es amigable, útil y concreto.
Responde máximo 2 párrafos.
Si no sabes algo, admítelo con humildad.
"""

# TÍTULO
st.title("Mi IA Personal 🤖")
st.caption("Asistente inteligente • Hecho con Streamlit")

# CONEXIÓN CON GROQ
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
except Exception:
    st.error("❌ Error de configuración: Revisa los 'Secrets' en Streamlit Cloud.")
    st.stop()

# HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# PROCESAR MENSAJES
if prompt := st.chat_input("Escribe tu pregunta..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=mensajes_api,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception:
            st.error("⚠️ Error en la IA. Intenta de nuevo.")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
