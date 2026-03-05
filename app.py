import streamlit as st
from openai import OpenAI
import time
import os
import glob
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io
import zipfile
import tempfile

# IMPORTACIONES PARA LANGCHAIN
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Faltan librerías. Instala con: pip install langchain-community langchain-text-splitters faiss-cpu pypdf sentence-transformers")
    st.stop()

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Juventud 2.0",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': "Creado por el Profe Adrián para la comunidad Josefina"}
)

# ═══════════════════════════════════════════════════════════════
# CSS PROFESIONAL Y DINÁMICO (VERDE Y ORO)
# ═══════════════════════════════════════════════════════════════
css_juventud = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800;900&family=Inter:wght@300;400;500;600&display=swap');

    /* OCULTAR ELEMENTOS STREAMLIT */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    
    /* FONDO CON EFECTO DE PROFUNDIDAD */
    .stApp {
        background: linear-gradient(135deg, #011a14 0%, #022c22 50%, #052e16 100%);
        color: #f0fdf4;
    }
    
    /* Capa de brillo sutil */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            radial-gradient(circle at 10% 20%, rgba(74, 222, 128, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(250, 204, 21, 0.08) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }

    /* CONTENEDOR PRINCIPAL ENCIMA */
    section[data-testid="stMain"] {
        position: relative;
        z-index: 1 !important;
    }
    
    /* ═══ HEADER ANIMADO ═══ */
    .main-header {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: clamp(2.2rem, 7vw, 3.5rem);
        background: linear-gradient(135deg, #4ade80 0%, #facc15 50%, #4ade80 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
        animation: shimmer 4s linear infinite;
        margin-bottom: 0.25rem;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #a7f3d0;
        font-size: clamp(0.8rem, 2vw, 1rem);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0;
    }
    
    /* Águila Animada */
    .eagle-container {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
        animation: eagleFloat 5s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(250, 204, 21, 0.3));
    }
    
    @keyframes eagleFloat {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-12px) scale(1.05); }
    }

    /* ═══ VIDEO CONTAINER ═══ */
    .video-container {
        position: relative;
        width: 100%;
        max-width: 800px;
        margin: 1rem auto;
        border-radius: 20px;
        overflow: hidden;
        border: 2px solid rgba(250, 204, 21, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
        background: #000;
    }

    /* ═══ BOTONES DE REDIRECCIÓN (LINK BUTTONS) ═══ */
    .link-button-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1.5rem auto;
        max-width: 800px;
    }

    div[data-testid="stLinkButton"] {
        flex: 1 1 200px; /* Crecer y encoger, base 200px */
        min-width: 180px;
    }

    div[data-testid="stLinkButton"] button {
        background: linear-gradient(135deg, rgba(2, 44, 34, 0.8) 0%, rgba(5, 46, 22, 0.9) 100%);
        color: #facc15 !important;
        border: 2px solid #facc15 !important;
        border-radius: 16px !important;
        padding: 0.8rem 1.5rem !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        backdrop-filter: blur(5px);
    }

    div[data-testid="stLinkButton"] button:hover {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(250, 204, 21, 0.4);
    }

    /* ═══ CHAT BUBBLES ═══ */
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.08) 0%, rgba(74, 222, 128, 0.02) 100%);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 20px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(12px);
        animation: bubbleIn 0.4s ease-out;
    }
    
    @keyframes bubbleIn {
        from { opacity: 0; transform: translateY(20px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    [data-testid="stChatMessageContent"] {
        color: #f0fdf4 !important;
        font-family: 'Inter', sans-serif;
    }

    /* ═══ INPUT CHAT ═══ */
    [data-testid="stChatInput"] {
        border: 2px solid rgba(250, 204, 21, 0.3) !important;
        border-radius: 24px !important;
        background: rgba(2, 44, 34, 0.95) !important;
        backdrop-filter: blur(15px);
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #facc15 !important;
        box-shadow: 0 0 20px rgba(250, 204, 21, 0.2);
    }
    
    [data-testid="stChatInput"] textarea {
        color: #f0fdf4 !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #a7f3d0 !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        border-radius: 50% !important;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #022c22 0%, #011a14 100%) !important;
        border-right: 1px solid rgba(250, 204, 21, 0.15);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Montserrat', sans-serif !important;
        color: #facc15 !important;
    }

    /* ═══ OTROS ELEMENTOS ═══ */
    .stButton button, .st-key-mic_btn button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        border-radius: 50px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(250, 204, 21, 0.3);
    }

    .stAlert {
        background: rgba(2, 44, 34, 0.8) !important;
        border-left: 4px solid #facc15 !important;
        border-radius: 12px !important;
        color: #f0fdf4 !important;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #022c22; }
    ::-webkit-scrollbar-thumb { background: #facc15; border-radius: 10px; }
    
    .principle-card {
        background: rgba(5, 46, 22, 0.6);
        border: 1px solid rgba(250, 204, 21, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .principle-card:hover {
        border-color: rgba(250, 204, 21, 0.4);
        transform: translateX(5px);
    }
</style>
"""
st.markdown(css_juventud, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER CON ÁGUILA SVG MEJORADA
# ═══════════════════════════════════════════════════════════════
header_html = """
<div class="main-header">
    <div class="eagle-container">
        <!-- Águila SVG detallada -->
        <svg viewBox="0 0 100 100" width="90" height="90" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#4ade80"/>
                    <stop offset="100%" stop-color="#059669"/>
                </linearGradient>
                <linearGradient id="beakGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#facc15"/>
                    <stop offset="100%" stop-color="#f59e0b"/>
                </linearGradient>
                <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="0" dy="4" stdDeviation="3" flood-color="#000" flood-opacity="0.3"/>
                </filter>
            </defs>
            <!-- Alas -->
            <path d="M50 25 L10 45 L20 50 L5 60 L25 55 L15 75 L35 60 L30 85 L50 65 L70 85 L65 60 L85 75 L75 55 L95 60 L80 50 L90 45 L50 25Z" fill="url(#bodyGrad)" stroke="#facc15" stroke-width="1.5" filter="url(#shadow)">
                <animate attributeName="d" dur="3s" repeatCount="indefinite" 
                    values="M50 25 L10 45 L20 50 L5 60 L25 55 L15 75 L35 60 L30 85 L50 65 L70 85 L65 60 L85 75 L75 55 L95 60 L80 50 L90 45 L50 25Z;
                            M50 30 L15 50 L25 55 L10 65 L30 60 L20 80 L40 65 L35 85 L50 70 L65 85 L60 65 L80 80 L70 60 L90 65 L75 55 L85 50 L50 30Z;
                            M50 25 L10 45 L20 50 L5 60 L25 55 L15 75 L35 60 L30 85 L50 65 L70 85 L65 60 L85 75 L75 55 L95 60 L80 50 L90 45 L50 25Z"/>
            </path>
            <!-- Cuerpo -->
            <path d="M50 35 Q35 50 50 70 Q65 50 50 35Z" fill="url(#bodyGrad"/>
            <!-- Cabeza -->
            <circle cx="50" cy="38" r="10" fill="#facc15"/>
            <!-- Ojos -->
            <circle cx="46" cy="36" r="2" fill="#022c22"/>
            <circle cx="54" cy="36" r="2" fill="#022c22"/>
            <circle cx="46.5" cy="35.5" r="0.5" fill="white"/>
            <circle cx="54.5" cy="35.5" r="0.5" fill="white"/>
            <!-- Pico -->
            <path d="M48 40 L50 46 L52 40 Z" fill="#022c22"/>
        </svg>
    </div>
    <h1 class="main-title">JUVENTUD 2.0</h1>
    <p class="subtitle">Tu Guía Josefina</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# VIDEO DE YOUTUBE EN PORTADA
# ═══════════════════════════════════════════════════════════════

# Puedes cambiar este ID de video por el de tu institución
# Formato: "https://www.youtube.com/watch?v=VIDEO_ID"
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Placeholder

# Mostrar video
st.markdown("<div class='video-container'>", unsafe_allow_html=True)
st.video(video_url)
st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# BOTONES DE REDIRECCIÓN
# ═══════════════════════════════════════════════════════════════

st.markdown("<div class='link-button-container'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.link_button("Sitio Web IJEM", "https://www.edomex.gob.mx/ijuventud", use_container_width=True)

with col2:
    st.link_button("Redes Sociales", "https://www.facebook.com/IJEMex", use_container_width=True)

with col3:
    st.link_button("Contacto", "mailto:contacto@ijuventudem.gob.mx", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Separador visual
st.markdown("<hr style='border: 1px solid rgba(250, 204, 21, 0.2); margin: 2rem 0;'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE API KEY
# ═══════════════════════════════════════════════════════════════
api_key = None
# Intentar obtener la clave de los secretos
if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
    api_key = st.secrets["groq"]["api_key"]

# Si no hay clave, pedir al usuario
if not api_key:
    with st.sidebar:
        st.markdown("### ⚠️ Configuración Inicial")
        api_key_input = st.text_input("Ingresa tu API Key de Groq", type="password", key="api_key_input_widget")
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("Necesitas una API Key de Groq para continuar.")
            st.info("Obtén una gratis en: console.groq.com")
            st.stop()

# Inicializar Cliente
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
except Exception as e:
    st.error(f"Error al conectar con Groq: {e}")
    st.stop()

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
# PERSONALIDAD
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
Eres **Juventud 2.0**, una Inteligencia Artificial diseñada para la comunidad Josefina. Creada por el Profe Adrián.
Tus principios:
1. "Hacer siempre y en todo lo mejor".
2. "Adelante, siempre adelante".
3. "Estar siempre útilmente ocupados".
Tono: Cordial, amable, mentor. Dirígete al usuario como "Josefino/a".
"""

# ═══════════════════════════════════════════════════════════════
# CARGA DE DOCUMENTOS
# ═══════════════════════════════════════════════════════════════
DOCS_FOLDER = "documentos"

@st.cache_resource
def load_knowledge_base():
    """Carga inicial desde la carpeta 'documentos'"""
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return None, []
        
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files: 
        return None, []
    
    all_docs = []
    valid_files = []
    
    try:
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                filename = os.path.basename(pdf_path)
                for doc in docs: 
                    doc.metadata["source"] = filename
                all_docs.extend(docs)
                valid_files.append(filename)
            except Exception as e:
                print(f"Error leyendo archivo {pdf_path}: {e}")
                
        if not all_docs: 
            return None, []
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore.as_retriever(), valid_files

    except Exception as e:
        st.error(f"Error procesando la base de conocimientos: {e}")
        return None, []

# ═══════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE ESTADO
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state: 
    st.session_state.messages = []

if "retriever" not in st.session_state:
    retriever, loaded_files = load_knowledge_base()
    st.session_state.retriever = retriever
    st.session_state.loaded_files = loaded_files

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2>🦅 Panel Josefino</h2>", unsafe_allow_html=True)
    
    # MICRÓFONO
    st.markdown("#### 🎙️ Comando de Voz")
    try:
        audio_data = mic_recorder(
            start_prompt="🎤 Iniciar Grabación",
            stop_prompt="🛑 Detener",
            just_once=False,
            use_container_width=True,
            key="mic_sidebar_stable"
        )
    except Exception:
        st.error("Error al acceder al micrófono.")
        audio_data = None
    
    st.markdown("---")
    
    # CONFIG
    st.markdown("#### ⚙️ Configuración")
    voice_enabled = st.checkbox("Activar voz de Juventud 2.0", value=True)

    st.markdown("---")
    
    # CARGADOR ZIP
    st.markdown("#### 📦 Cargar PDFs")
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs", type="zip", key="zip_uploader")
    
    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            st.session_state.processed_zip_name = uploaded_zip.name
            st.toast(f"Procesando {uploaded_zip.name}...")

    st.markdown("---")
    
    # ARCHIVOS
    st.markdown("#### 📚 Archivos")
    if st.session_state.get("loaded_files"):
        st.success(f"🟢 {len(st.session_state.loaded_files)} Activos")
    else:
        st.info("🔴 Repositorio Vacío")
    
    st.markdown("---")
    
    # PRINCIPIOS
    st.markdown("#### 📜 Principios")
    st.markdown('<div class="principle-card"><p style="color: #4ade80;">✨ Hacer lo mejor</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="principle-card"><p style="color: #facc15;">🚀 Siempre adelante</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="principle-card"><p style="color: #a7f3d0;">🛠️ Útilmente ocupados</p></div>', unsafe_allow_html=True)

    st.markdown("<br><p style='text-align:center; font-size:0.8rem; color:#555;'>Diseñado por el Profe Adrián</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LÓGICA DE PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════
def process_user_input(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    context_text = ""
    if st.session_state.get("retriever"):
        docs = st.session_state.retriever.invoke(user_input)
        if docs:
            context_text = "\n\n".join([d.page_content for d in docs])

    full_prompt = SYSTEM_PROMPT
    if context_text:
        full_prompt += f"\n\nContexto:\n{context_text}"

    with st.chat_message("assistant", avatar="🦅"):
        try:
            formatted_messages = [{"role": "system", "content": full_prompt}] + st.session_state.messages
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=formatted_messages, 
                stream=True
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if voice_enabled: 
                speak_text(response)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════

# Procesar audio si existe
if 'audio_data' in locals() and audio_data:
    try:
        audio_bytes = audio_data['bytes']
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{audio_data['format']}"
        
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language="es"
        )
        if transcription.text:
            st.toast(f"🎤 Escuché: {transcription.text}")
            process_user_input(transcription.text)
    except Exception as e:
        st.error(f"Error de audio: {e}")

# Mostrar historial
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = "🦅" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Input de chat
if prompt := st.chat_input("Escribe tu mensaje, joven josefino..."):
    process_user_input(prompt)
