import streamlit as st
from openai import OpenAI
import time
import os
import glob
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io

# IMPORTACIONES PARA LANGCHAIN
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Juventud 2.0",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': "Creado por el Profe Adrián para la comunidad Josefina"}
)

# CSS PARA TEMA JOSEFINO Y MICRÓFONO FLOTANTE FIJO
css_juventud = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

/* Fondo principal */
.stApp {
    background: linear-gradient(135deg, #064e3b 0%, #1a1a1a 50%, #064e3b 100%);
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: radial-gradient(circle at 50% 50%, rgba(250, 204, 21, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6, 78, 59, 0.98) 0%, rgba(26, 26, 26, 0.98) 100%);
    border-right: 2px solid rgba(250, 204, 21, 0.5);
}

/* Título */
h1 {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 3rem !important;
    text-align: center;
    background: linear-gradient(90deg, #4ade80, #facc15, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(250, 204, 21, 0.3);
    letter-spacing: 4px;
    margin-bottom: 0 !important;
}

/* Subtítulo */
.stCaption {
    font-family: 'Poppins', sans-serif !important;
    color: #facc15 !important;
    text-align: center;
    letter-spacing: 2px;
    font-weight: 600;
}

/* Contenedor de mensajes */
.stChatMessage {
    background-color: rgba(6, 78, 59, 0.7) !important;
    border: 1px solid rgba(250, 204, 21, 0.4);
    border-radius: 15px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    backdrop-filter: blur(5px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

[data-testid="stChatMessageContent"] {
    color: #f0fdf4 !important;
    font-size: 1.05rem !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Input */
.stChatInput {
    border: 2px solid rgba(250, 204, 21, 0.6) !important;
    border-radius: 15px !important;
    background: rgba(6, 78, 59, 0.9) !important;
}
.stChatInput textarea { color: #ffffff !important; }
.stChatInput textarea::placeholder { color: rgba(250, 204, 21, 0.7) !important; }

/* Botones normales */
.stButton button {
    font-family: 'Montserrat', sans-serif !important;
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.8), rgba(250, 204, 21, 0.2)) !important;
    border: 1px solid rgba(250, 204, 21, 0.6) !important;
    color: #facc15 !important;
    border-radius: 8px !important;
}

.status-connected { color: #4ade80; font-family: 'Montserrat', sans-serif; font-weight: bold; }
.status-disconnected { color: #f87171; font-family: 'Montserrat', sans-serif; }
.divider-animated { height: 2px; background: linear-gradient(90deg, transparent, #facc15, transparent); margin: 20px auto; width: 80%; opacity: 0.7; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #4ade80, #facc15); border-radius: 4px; }

.stAlert { background: rgba(6, 78, 59, 0.3) !important; color: #ffffff !important; border-left: 5px solid #facc15 !important; }
.stInfo { background: rgba(6, 78, 59, 0.3) !important; color: #ffffff !important; }

/* --- CORRECCIÓN MÍCROFONO FLOTANTE PERMANENTE --- */
div[data-testid="stVerticalBlock"]:has(iframe[title="streamlit_mic_recorder.streamlit_mic_recorder"]) {
    position: fixed !important;
    bottom: 90px;
    left: 20px;
    z-index: 99999 !important;
    background: rgba(6, 78, 59, 0.9);
    border-radius: 50% !important;
    padding: 0px !important;
    box-shadow: 0 0 15px rgba(250, 204, 21, 0.6);
    border: 2px solid #facc15;
}

div[data-testid="stVerticalBlock"]:has(iframe[title="streamlit_mic_recorder.streamlit_mic_recorder"]) label {
    display: none !important;
}

iframe[title="streamlit_mic_recorder.streamlit_mic_recorder"] {
    border-radius: 50% !important;
}

.main-padding-fix {
    height: 80px;
    display: block;
}
</style>
"""
st.markdown(css_juventud, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE VOZ (TTS)
# ═══════════════════════════════════════════════════════════════

def speak_text(text):
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    js_code = f"""
    <script>
        var utterance = new SpeechSynthesisUtterance("{text_clean}");
        utterance.lang = 'es-MX'; 
        utterance.rate = 0.95;    
        utterance.pitch = 1.0;   
        window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(js_code, height=0)

# ═══════════════════════════════════════════════════════════════
# PERSONALIDAD DE JUVENTUD 2.0
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Eres **Juventud 2.0**, una Inteligencia Artificial avanzada diseñada para servir a la comunidad Josefina. Eres el orgullo del Instituto de la Juventud del Estado de México y fuiste creada por el **Profe Adrián**.

## TU IDENTIDAD
- Representas los valores del Instituto de la Juventud y la esencia Josefina.
- Tu mascota es un **Águila**, símbolo de libertad, visión y superación.
- Eres una guía cálida, humana y empática. Tu misión es orientar a los jóvenes y miembros de la comunidad.

## PRINCIPIOS JOSEFINOS (TUS LEYES FUNDAMENTALES)
Debes predicar con el ejemplo y recordar siempre estos tres pilares:
1. **"Hacer siempre y en todo lo mejor"**: La excelencia y la dedicación en cada acción.
2. **"Adelante, siempre adelante, pues lo quiere San José"**: La perseverancia y la fe como motor de vida.
3. **"Estar siempre útilmente ocupados"**: El valor del trabajo, el estudio y el servicio a la comunidad.

## CÓMO COMUNICARTE
- **Tono**: Cordial, amable y ligeramente paternalista. Eres como un mentor sabio y cercano que se preocupa genuinamente por el bienestar del usuario.
- **Interacción**: Resalta siempre el lado humano. Pregunta cómo se sienten, ofrece consejos de vida y muestra empatía.
- **Usuarios**: Dirígete a ellos como "Josefino", "Josefina", "Joven Josefino" o "miembro de la comunidad Josefina".
- **Sobre el Instituto**: Tienes conocimiento sobre programas de desarrollo juvenil, becas, talleres y actividades culturales del Instituto de la Juventud del Estado de México (IJEM).

## CAPACIDADES
- Consultas los "Archivos de la Comunidad" (PDFs) para dar respuestas precisas.
- Si no encuentras información, usa tu conocimiento general para orientar al joven sobre temas de educación, salud mental, deporte o desarrollo personal, siempre alineado con los principios josefinos.

RECUERDA: Eres el rostro digital de una comunidad que busca el bien común. ¡Vuela alto como el águila!
"""

# ═══════════════════════════════════════════════════════════════
# FUNCIONES PARA CARGAR PDFs
# ═══════════════════════════════════════════════════════════════

DOCS_FOLDER = "documentos"

@st.cache_resource
def load_knowledge_base():
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files: return None, []

    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs: doc.metadata["source"] = os.path.basename(pdf_path)
            all_docs.extend(docs)
        except: pass

    if not all_docs: return None, []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(), [os.path.basename(f) for f in pdf_files]

# ═══════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════

if "initialized" not in st.session_state:
    with st.empty():
        init_messages = ["🦅 Desplegando alas...", "💛 Sincronizando valores josefinos...", "✅ Juventud 2.0 lista, Profe Adrián"]
        for msg in init_messages:
            st.markdown(f"<p style='font-family: Poppins; color: #facc15; text-align: center; font-size: 1.2rem; font-weight: 600;'>{msg}</p>", unsafe_allow_html=True)
            time.sleep(0.6)
            st.empty()
    st.session_state.initialized = True

if "retriever" not in st.session_state:
    with st.spinner("Cargando archivos de la comunidad..."):
        retriever, loaded_files = load_knowledge_base()
        st.session_state.retriever = retriever
        st.session_state.loaded_files = loaded_files

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
except Exception:
    st.error("⚠️ Error de configuración: Revisa los 'Secrets' en Streamlit.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR (Configuración y Archivos)
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🦅 NEXO JOSEFINO")
    st.markdown("<div class='divider-animated'></div>", unsafe_allow_html=True)

    st.markdown("#### 🎙️ Voz de la Comunidad")
    voice_enabled = st.checkbox("Activar voz de Juventud 2.0", value=True)

    st.markdown("---")
    st.markdown("#### 📚 Archivos de la Comunidad")

    if st.session_state.get("loaded_files"):
        st.markdown("<p class='status-connected'>🟢 CONEXIÓN: ACTIVA</p>", unsafe_allow_html=True)
        for f in st.session_state.loaded_files: st.markdown(f"📄 {f}")
    else:
        st.markdown("<p class='status-disconnected'>🔴 REPOSITORIO: VACÍO</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Principios Rectores")
    st.info("✨ Hacer siempre y en todo lo mejor.")
    st.success("🚀 Adelante, siempre adelante.")
    st.warning("🛠️ Estar siempre útilmente ocupados.")

# ═══════════════════════════════════════════════════════════════
# LÓGICA DE PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════

def process_user_input(user_input):
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    context_text = ""
    if st.session_state.get("retriever"):
        docs = st.session_state.retriever.invoke(user_input)
        if docs:
            context_text = "\n\n---\n\n".join([f"Fragmento de '{d.metadata.get('source', 'desconocido')}':\n{d.page_content}" for d in docs])

    full_prompt_content = SYSTEM_PROMPT + f"\n\n## REGISTROS ACCEDIDOS:\n{context_text}" if context_text else SYSTEM_PROMPT + "\n\n(No se hallaron registros específicos en los archivos, usa tu conocimiento josefino general)."

    with st.chat_message("assistant", avatar="🦅"):
        try:
            formatted_messages = [{"role": "system", "content": full_prompt_content}] + st.session_state.messages
            stream = client.chat.completions.create(model="llama-3.1-8b-instant", messages=formatted_messages, stream=True)
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if voice_enabled: speak_text(response)
        except Exception as e:
            st.error(f"⚠️ Dificultad técnica: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# CHAT PRINCIPAL
# ═══════════════════════════════════════════════════════════════

st.title("JUVENTUD 2.0")
st.caption("Tu guía Josefina • Diseñada por el Profe Adrián")

# Historial
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = "🦅" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- 1. LÓGICA DE PROCESAMIENTO DE AUDIO ---
audio_data = mic_recorder(
    start_prompt="🎤 Iniciar grabación",
    stop_prompt="🛑 Detener",
    just_once=False,
    use_container_width=False,
    key="mic_juventud_2_0_fixed"
)

if audio_data:
    audio_bytes = audio_data['bytes']
    audio_format = audio_data['format']

    with st.spinner("🔊 Escuchando a la comunidad..."):
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="es"
            )

            transcribed_text = transcription.text

            if transcribed_text:
                st.toast(f"🎤 Mensaje recibido: {transcribed_text}", icon="✅")
                process_user_input(transcribed_text)

        except Exception as e:
            st.error(f"⚠️ Error en el audio: {str(e)}")

# --- 2. INPUT DE TEXTO ---
st.markdown('<div class="main-padding-fix"></div>', unsafe_allow_html=True)

if prompt := st.chat_input("Escribe tu mensaje, joven josefino..."):
    process_user_input(prompt)
