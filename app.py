import streamlit as st
from openai import OpenAI
import os
import glob
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io
import zipfile
import requests  # Necesario para conectar con Hugging Face

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
# CSS PROFESIONAL Y DINÁMICO
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
    
    /* FONDO */
    .stApp {
        background: linear-gradient(135deg, #011a14 0%, #022c22 50%, #052e16 100%);
        color: #f0fdf4;
    }
    
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

    section[data-testid="stMain"] {
        position: relative;
        z-index: 1 !important;
    }
    
    /* HEADER */
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
    }

    /* VIDEO CONTAINER */
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

    /* BOTONES REDIRECCIÓN */
    .link-button-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1.5rem auto;
        max-width: 800px;
    }

    div[data-testid="stLinkButton"] {
        flex: 1 1 200px;
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
        transition: all 0.3s ease !important;
    }

    div[data-testid="stLinkButton"] button:hover {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(250, 204, 21, 0.4);
    }

    /* MIC BUTTON TOP */
    .mic-container-top {
        display: flex;
        justify-content: center;
        margin: 1.5rem auto 1rem auto;
    }
    
    .st-key-mic_main_btn button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        font-weight: 700 !important;
        border-radius: 50px !important;
        padding: 1rem 2rem !important;
        box-shadow: 0 5px 20px rgba(250, 204, 21, 0.3);
    }

    /* TABS ESTILOS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(2, 44, 34, 0.6);
        border-radius: 12px 12px 0 0;
        color: #a7f3d0;
        border: 1px solid rgba(250, 204, 21, 0.2);
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #022c22 0%, #052e16 100%);
        color: #facc15 !important;
        border-bottom: 2px solid #facc15;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #022c22 0%, #011a14 100%) !important;
        border-right: 1px solid rgba(250, 204, 21, 0.15);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Montserrat', sans-serif !important;
        color: #facc15 !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        border-radius: 50px !important;
        border: none !important;
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
    }
    
    /* CSS PARA IMAGEN GENERADA */
    .generated-image-container {
        background: rgba(2, 44, 34, 0.4);
        border: 1px solid rgba(250, 204, 21, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
</style>
"""
st.markdown(css_juventud, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
header_html = """
<div class="main-header">
    <h1 class="main-title">JUVENTUD 2.0</h1>
    <p class="subtitle">Tu Guía Josefina</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# VIDEO Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════
st.markdown("<div class='video-container'>", unsafe_allow_html=True)
st.video("https://www.youtube.com/watch?v=9R1cIkiMkrU")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='link-button-container'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.link_button("Sitio Web IJEM", "https://juventud.edu.mx/", use_container_width=True)
with col2:
    st.link_button("Redes Sociales", "https://www.facebook.com/InstitutoJuventudMX/?locale=es_LA", use_container_width=True)
with col3:
    st.link_button("Contacto", "mailto:contacto@ijuventudem.gob.mx", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE API KEYS
# ═══════════════════════════════════════════════════════════════
api_key = None
if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
    api_key = st.secrets["groq"]["api_key"]

hf_token = None
if "huggingface" in st.secrets and "token" in st.secrets["huggingface"]:
    hf_token = st.secrets["huggingface"]["token"]

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2>🦅 Panel Josefino</h2>", unsafe_allow_html=True)
    
    # Configuración Chat
    st.markdown("#### ⚙️ Chat (Groq)")
    if not api_key:
        api_key_input = st.text_input("API Key de Groq", type="password", key="api_key_input_groq")
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("Necesitas API Key de Groq.")
    else:
        st.success("Configurado ✅")

    # Configuración Imágenes
    st.markdown("#### 🎨 Imágenes (Hugging Face)")
    if not hf_token:
        hf_token_input = st.text_input("Token de HF", type="password", key="api_key_input_hf", help="Obtén uno gratis en huggingface.co -> Settings -> Access Tokens")
        if hf_token_input:
            hf_token = hf_token_input
        else:
            st.info("Gratis en huggingface.co")
    else:
        st.success("Configurado ✅")

    voice_enabled = st.checkbox("Activar voz", value=True)
    st.markdown("---")

    st.markdown("#### 📦 Cargar PDFs")
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs", type="zip", key="zip_uploader")

    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            st.session_state.processed_zip_name = uploaded_zip.name
            st.toast(f"Procesando {uploaded_zip.name}...")

    st.markdown("---")
    st.markdown("#### 📚 Archivos")
    if st.session_state.get("loaded_files"):
        st.success(f"🟢 {len(st.session_state.loaded_files)} Activos")
    else:
        st.info("🔴 Repositorio Vacío")

    st.markdown("---")
    st.markdown("#### 📜 Principios")
    st.markdown('<div class="principle-card"><p style="color: #4ade80;">✨ Hacer lo mejor</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="principle-card"><p style="color: #facc15;">🚀 Siempre adelante</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="principle-card"><p style="color: #a7f3d0;">🛠️ Útilmente ocupados</p></div>', unsafe_allow_html=True)
    st.markdown("<br><p style='text-align:center; font-size:0.8rem; color:#555;'>Diseñado por el Profe Adrián</p>", unsafe_allow_html=True)

# Validación
if not api_key:
    st.warning("Por favor, ingresa tu API Key de Groq en el panel lateral para usar el chat.")
    # No detenemos la app para permitir configurar imágenes primero

# Inicialización de cliente Groq
if api_key:
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error al conectar con Groq: {e}")

# ═══════════════════════════════════════════════════════════════
# LÓGICA CHAT Y DOCUMENTOS
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
Eres **Juventud 2.0**, una Inteligencia Artificial diseñada para la comunidad Josefina. Creada por el Profe Adrián.
Tus principios:
1. "Hacer siempre y en todo lo mejor".
2. "Adelante, siempre adelante".
3. "Estar siempre útilmente ocupados".
Tono: Cordial, amable, mentor. Dirígete al usuario como "Josefino/a".
"""

DOCS_FOLDER = "documentos"

@st.cache_resource
def load_knowledge_base():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return None, []

    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files: return None, []

    all_docs = []
    valid_files = []
    try:
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                filename = os.path.basename(pdf_path)
                for doc in docs: doc.metadata["source"] = filename
                all_docs.extend(docs)
                valid_files.append(filename)
            except: pass

        if not all_docs: return None, []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever(), valid_files
    except:
        return None, []

if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state:
    retriever, loaded_files = load_knowledge_base()
    st.session_state.retriever = retriever
    st.session_state.loaded_files = loaded_files

def get_audio_button_html(text, key):
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    return f"""
    <div style="margin-top: 10px; text-align: right;">
        <button onclick="
            var u = new SpeechSynthesisUtterance('{text_clean}');
            u.lang = 'es-MX';
            u.rate = 0.95;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        " style="
            background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%);
            color: #022c22;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            cursor: pointer;
            font-family: 'Montserrat', sans-serif;
            font-size: 0.85rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        ">Escuchar Respuesta</button>
    </div>
    """

# ═══════════════════════════════════════════════════════════════
# PESTAÑAS
# ═══════════════════════════════════════════════════════════════
tab_chat, tab_img = st.tabs(["🦅 Chat Josefino", "🎨 Generador de Imágenes"])

# --- CHAT ---
with tab_chat:
    if not api_key:
        st.info("Ingresa tu API Key de Groq en el sidebar para comenzar.")
    else:
        st.markdown("<div class='mic-container-top'>", unsafe_allow_html=True)
        audio_data = mic_recorder(start_prompt="🎤 Iniciar Grabación", stop_prompt="🛑 Detener", just_once=False, key="mic_main_btn")
        st.markdown("</div>", unsafe_allow_html=True)

        if audio_data:
            try:
                audio_bytes = audio_data['bytes']
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = f"audio.{audio_data['format']}"
                transcription = client.audio.transcriptions.create(file=audio_file, model="whisper-large-v3", language="es")
                if transcription.text:
                    st.toast(f"🎤 Escuché: {transcription.text}")
                    st.session_state.messages.append({"role": "user", "content": transcription.text})
                    # Lógica de respuesta (resumida)
                    docs = st.session_state.retriever.invoke(transcription.text) if st.session_state.get("retriever") else []
                    context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""
                    full_prompt = SYSTEM_PROMPT + (f"\n\nContexto:\n{context_text}" if context_text else "")
                    formatted_messages = [{"role": "system", "content": full_prompt}] + st.session_state.messages
                    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=formatted_messages)
                    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            except Exception as e:
                st.error(f"Error de audio: {e}")

        if prompt := st.chat_input("Escribe tu mensaje..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Lógica de respuesta
            docs = st.session_state.retriever.invoke(prompt) if st.session_state.get("retriever") else []
            context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""
            full_prompt = SYSTEM_PROMPT + (f"\n\nContexto:\n{context_text}" if context_text else "")
            formatted_messages = [{"role": "system", "content": full_prompt}] + st.session_state.messages
            try:
                response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=formatted_messages)
                st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            except Exception as e:
                st.error(f"Error: {e}")

        # Contenedor Chat
        st.markdown("<div class='fixed-chat-wrapper' style='background: rgba(2, 44, 34, 0.4); border-radius: 20px; padding: 1rem;'>", unsafe_allow_html=True)
        chat_container = st.container(height=450, key="chat_container")
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] != "system":
                    avatar = "🦅" if message["role"] == "assistant" else "👤"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])
                        if message["role"] == "assistant" and voice_enabled:
                            components.html(get_audio_button_html(message["content"], f"audio_{i}"), height=50)
        st.markdown("</div>", unsafe_allow_html=True)

# --- IMÁGENES (Estable con Hugging Face) ---
with tab_img:
    st.markdown("### 🎨 Taller de Creación Josefino")
    if not hf_token:
        st.warning("⚠️ Para generar imágenes, necesitas un Token de Hugging Face (es gratis). ingrésalo en el panel lateral.")
        st.markdown("**¿Cómo obtenerlo?**")
        st.code("1. Entra a huggingface.co y regístrate.\n2. Ve a Settings -> Access Tokens.\n3. Crea un token tipo 'Read' y pégalo en el sidebar.", language="python")
    else:
        st.success("✅ Motor de imágenes activado (Stable Diffusion).")
        
        img_col1, img_col2 = st.columns([3, 1])
        with img_col1:
            img_prompt = st.text_area("Descripción detallada de la imagen", placeholder="Ej: Un águila majestuosa sobre un volcán al atardecer, estilo realista...", height=100)
        with img_col2:
            style_option = st.selectbox("Estilo", ["Fotorrealista", "Arte Digital", "Pintura al Óleo", "Cyberpunk", "Anime"])
            st.caption("Tarda aprox 10-20 seg")

        if st.button("✨ Generar Imagen", use_container_width=True):
            if not img_prompt:
                st.warning("Escribe una descripción.")
            else:
                with st.spinner("Conectando con Hugging Face y pintando..."):
                    try:
                        # Configuración API Hugging Face
                        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
                        headers = {"Authorization": f"Bearer {hf_token}"}
                        
                        # Prompt mejorado
                        final_prompt = f"{img_prompt}, estilo {style_option}, 4k, alta calidad, detallado"
                        payload = {"inputs": final_prompt}
                        
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                        
                        if response.status_code == 200:
                            image_bytes = response.content
                            st.session_state['last_img_bytes'] = image_bytes
                            st.session_state['last_img_prompt'] = img_prompt
                            st.toast("¡Imagen lista!")
                        elif response.status_code == 503:
                            st.error("El modelo se está calentando. Espera 1 minuto e intenta de nuevo.")
                        else:
                            st.error(f"Error {response.status_code}: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error de conexión: {e}")

        # Mostrar imagen
        if 'last_img_bytes' in st.session_state:
            st.markdown("<div class='generated-image-container'>", unsafe_allow_html=True)
            st.image(st.session_state['last_img_bytes'], caption=st.session_state.get('last_img_prompt', ''), use_column_width=True)
            st.download_button(
                label="📥 Descargar Imagen",
                data=st.session_state['last_img_bytes'],
                file_name="juventud_2_0_imagen.png",
                mime="image/png",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)