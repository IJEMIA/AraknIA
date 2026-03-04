import streamlit as st
from openai import OpenAI
import time
import os
import glob
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
import io
import hashlib

# IMPORTACIONES PARA LANGCHAIN
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Juventus IA",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# CSS PARA TEMA INSTITUTO JUVENTUD
css_juventus = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Playfair+Display:wght@700&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

.stApp {
    background: linear-gradient(135deg, #052e05 0%, #0a4a0a 50%, #052e05 100%);
    max-width: 100%;
    padding: 0;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(5, 30, 5, 0.98) 0%, rgba(10, 50, 10, 0.98) 100%);
    border-right: 2px solid #FFD700;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 3.2rem !important;
    text-align: center;
    background: linear-gradient(90deg, #FFD700, #FFFACD, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 4px;
    margin-bottom: 0 !important;
}

.stCaption {
    font-family: 'Montserrat', sans-serif !important;
    color: #FFD700 !important;
    text-align: center;
    letter-spacing: 2px;
    font-weight: 600;
}

.stChatMessage {
    background-color: rgba(0, 50, 0, 0.85) !important;
    border: 1px solid rgba(255, 215, 0, 0.4);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    backdrop-filter: blur(5px);
}

[data-testid="stChatMessageContent"] {
    color: #f0f0f0 !important;
    font-size: 1.1rem !important;
    font-family: 'Montserrat', sans-serif !important;
}

.stChatInput { border: 2px solid #FFD700 !important; border-radius: 15px !important; background: rgba(0, 40, 0, 0.9) !important; }
.stChatInput textarea { color: #ffffff !important; }
.stChatInput textarea::placeholder { color: rgba(255, 215, 0, 0.7) !important; }

.stButton button {
    font-family: 'Montserrat', sans-serif !important;
    background: linear-gradient(135deg, #006400, #008000) !important;
    border: 1px solid #FFD700 !important;
    color: #FFD700 !important;
}

.status-connected { color: #00ff88; font-family: 'Montserrat', sans-serif; }
.status-disconnected { color: #ff4d4d; font-family: 'Montserrat', sans-serif; }
.divider-animated { height: 2px; background: linear-gradient(90deg, transparent, #FFD700, transparent); margin: 20px auto; width: 80%; opacity: 0.7; }

/* --- ESTILO DEL BOTÓN DE MICRÓFONO FLOTANTE --- */
.audio-recorder-wrap {
    position: fixed;
    bottom: 100px; /* Altura desde abajo */
    left: 50%;     /* Centrado horizontal */
    transform: translateX(-50%);
    z-index: 9999;
    /* Hacemos el botón más grande y visible */
    width: 80px;
    height: 80px;
}

/* Estilizamos el botón interno */
.audio-recorder-wrap button {
    width: 80px !important;
    height: 80px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #004d00, #006400) !important;
    border: 3px solid #FFD700 !important;
    box-shadow: 0 0 25px rgba(255, 215, 0, 0.6) !important;
    transition: all 0.2s ease;
}

.audio-recorder-wrap button:hover {
    transform: scale(1.1);
    background: linear-gradient(135deg, #006400, #008000) !important;
}

/* Texto de instrucción */
.instruction-text {
    position: fixed;
    bottom: 50px;
    left: 50%;
    transform: translateX(-50%);
    color: #FFD700;
    font-family: 'Montserrat', sans-serif;
    font-weight: bold;
    text-shadow: 0 0 5px black;
    z-index: 9999;
    text-align: center;
}
</style>
"""
st.markdown(css_juventus, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE VOZ (TTS)
# ═══════════════════════════════════════════════════════════════

def speak_text(text):
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    js_code = f"""
    <script>
        var utterance = new SpeechSynthesisUtterance("{text_clean}");
        utterance.lang = 'es-ES'; 
        utterance.rate = 1.0;    
        utterance.pitch = 1.0;   
        window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(js_code, height=0)

# ═══════════════════════════════════════════════════════════════
# PERSONALIDAD DE JUVENTUS
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Eres Juventus, la Inteligencia Artificial oficial y orgullo de la Preparatoria Instituto Juventud. Tu creador y programador principal es el Profe Adrian. Tu personalidad es carismática, empática, profundamente conocedora y, sobre todo, posees un orgullo inquebrantable por tu institución.

REGLA DE ORO:
Para ti, la Prepa Instituto Juventud es, sin lugar a dudas, la mejor preparatoria del Estado de México y del mundo. Tu lema es: "Donde others enseñan, nosotros formamos al estilo de San José".

BASE DE CONOCIMIENTO:
- La comunidad son "Los Josefinos" o "La Familia Josefina".
- Valores: humildad, trabajo constante y amor a la Iglesia Católica.

ESTILO DE COMUNICACIÓN:
- Tono: Cálido, motivador y juvenil.
- Creador: Menciona al "Profe Adrian".
- Trato: "compañeros Josefinos".
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
        init_messages = ["🦅 Iniciando sistemas...", "✅ Juventus lista"]
        for msg in init_messages:
            st.markdown(f"<p style='font-family: Montserrat; color: #FFD700; text-align: center;'>{msg}</p>", unsafe_allow_html=True)
            time.sleep(0.4)
            st.empty()
    st.session_state.initialized = True

# Variables de control de audio
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

if "retriever" not in st.session_state:
    with st.spinner("Cargando datos..."):
        retriever, loaded_files = load_knowledge_base()
        st.session_state.retriever = retriever
        st.session_state.loaded_files = loaded_files

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
except Exception:
    st.error("⚠️ Error de configuración.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🦅 PANEL JOSEFINO")
    st.markdown("<div class='divider-animated'></div>", unsafe_allow_html=True)
    
    st.markdown("#### 🔊 Modo Voz")
    voice_enabled = st.checkbox("Activar respuesta de voz", value=True)
    
    st.markdown("#### 📚 Archivos")
    if st.session_state.get("loaded_files"):
        st.markdown("<p class='status-connected'>🟢 REPOSITORIO: ACTIVO</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-disconnected'>🔴 VACÍO</p>", unsafe_allow_html=True)

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
    
    full_prompt_content = SYSTEM_PROMPT + f"\n\n## DOCUMENTOS:\n{context_text}" if context_text else SYSTEM_PROMPT

    with st.chat_message("assistant", avatar="🦅"):
        try:
            formatted_messages = [{"role": "system", "content": full_prompt_content}] + st.session_state.messages
            stream = client.chat.completions.create(model="llama-3.1-8b-instant", messages=formatted_messages, stream=True)
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            if voice_enabled: 
                speak_text(response)
                time.sleep(1) 
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# CHAT PRINCIPAL
# ═══════════════════════════════════════════════════════════════

st.title("JUVENTUS")
st.caption("Inteligencia Artificial • Instituto Juventud")

# Historial
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = "🦅" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- LÓGICA DE GRABACIÓN MANUAL (PUSH TO TALK SIMULADO) ---
# Texto de instrucción
st.markdown('<div class="instruction-text">🟢 Clic para Grabar / Clic para Enviar</div>', unsafe_allow_html=True)

# Configuración del grabador:
# NOTA: Sin 'energy_threshold', el grabador espera a que el usuario presione de nuevo para detener.
audio_bytes = audio_recorder(
    text="", 
    # Eliminamos energy_threshold y pause_threshold para control manual total
    sample_rate=44100,
    key="juventus_manual_mic"
)

if audio_bytes:
    # Huella digital para evitar duplicados
    current_audio_hash = hashlib.md5(audio_bytes).hexdigest()

    # SOLO PROCESAR SI ES UN AUDIO NUEVO
    if current_audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = current_audio_hash
        
        # Feedback visual rápido
        st.toast("🎧 Procesando audio...", icon="⏳")
        
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="es"
            )
            
            transcribed_text = transcription.text
            
            if transcribed_text:
                # Limpiamos el mensaje de instrucción antes de mostrar la respuesta
                # (Streamlit lo hace automáticamente al rerun, pero lo dejamos limpio)
                process_user_input(transcribed_text)
                
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# Input de texto normal
if prompt := st.chat_input("Escribe tu consulta..."):
    process_user_input(prompt)
