Aquí tienes el código completo actualizado. He añadido una nueva pestaña llamada **"Generador de Imágenes"** junto al chat.

**Cambios importantes realizados:**
1.  **Nueva Pestaña:** El chat ahora convive con el generador de imágenes en `st.tabs`.
2.  **Configuración de API para Imágenes:** Dado que Groq no genera imágenes, he añadido en la barra lateral un campo para ingresar una **API Key de OpenAI** (necesaria para DALL-E). Si no la tienes, la función te avisará.
3.  **Estilo Visual:** Se adaptó el CSS para que las pestañas y los botones de descarga de imágenes sigan la estética "Juventud 2.0" (verde y dorado).

Copia y pega este código completo:

```python
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
import requests  # Necesario para descargar la imagen generada

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
# CSS PROFESIONAL Y DINÁMICO (Actualizado con estilos para Tabs)
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

    .st-key-mic_main_btn button:hover {
        transform: scale(1.05);
    }

    /* CHAT CONTENEDOR FIJO */
    .fixed-chat-wrapper {
        background: rgba(2, 44, 34, 0.4);
        border: 1px solid rgba(250, 204, 21, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .st-key-chat_container > div > div {
        border: none !important;
        background: transparent !important;
    }

    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.08) 0%, rgba(74, 222, 128, 0.02) 100%);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 20px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    
    [data-testid="stChatMessageContent"] {
        color: #f0fdf4 !important;
        font-family: 'Inter', sans-serif;
    }

    /* INPUT CHAT - CONTRASTE MEJORADO */
    [data-testid="stChatInput"] {
        border: 2px solid rgba(250, 204, 21, 0.3) !important;
        border-radius: 24px !important;
        background: rgba(2, 44, 34, 0.95) !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #d1d5db !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        border-radius: 50% !important;
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

    /* OTROS */
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
    
    .generated-image-container img {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        max-width: 100%;
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
# CONFIGURACIÓN DE API KEY
# ═══════════════════════════════════════════════════════════════
api_key = None
if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
    api_key = st.secrets["groq"]["api_key"]

# Variable para la API de OpenAI (Imágenes)
openai_img_key = None
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    openai_img_key = st.secrets["openai"]["api_key"]

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2>🦅 Panel Josefino</h2>", unsafe_allow_html=True)
    
    # 1. Configuración Groq (Chat)
    st.markdown("#### ⚙️ Configuración Chat")
    if not api_key:
        api_key_input = st.text_input("API Key de Groq", type="password", key="api_key_input_groq")
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("Necesitas API Key de Groq para el chat.")
            st.info("Obtén una en: console.groq.com")
    
    # 2. Configuración OpenAI (Imágenes)
    st.markdown("#### 🎨 Configuración Imágenes")
    if not openai_img_key:
        openai_img_input = st.text_input("API Key de OpenAI", type="password", key="api_key_input_openai_img", help="Necesaria para generar imágenes con DALL-E")
        if openai_img_input:
            openai_img_key = openai_img_input
    else:
        st.success("API OpenAI configurada ✨")

    voice_enabled = st.checkbox("Activar voz de Juventud 2.0", value=True)
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

# Validación final para el chat
if not api_key:
    st.stop()

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
except Exception as e:
    st.error(f"Error al conectar con Groq: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE VOZ Y PERSONALIDAD
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
            except Exception:
                pass

        if not all_docs: 
            return None, []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)

        return vectorstore.as_retriever(), valid_files

    except Exception:
        return None, []

# Inicialización de sesión
if "messages" not in st.session_state: 
    st.session_state.messages = []

if "retriever" not in st.session_state:
    retriever, loaded_files = load_knowledge_base()
    st.session_state.retriever = retriever
    st.session_state.loaded_files = loaded_files

# ═══════════════════════════════════════════════════════════════
# LÓGICA DE PROCESAMIENTO
# ═══════════════════════════════════════════════════════════════

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
# PESTAÑAS PRINCIPALES (CHAT vs IMÁGENES)
# ═══════════════════════════════════════════════════════════════
tab_chat, tab_img = st.tabs(["🦅 Chat Josefino", "🎨 Generador de Imágenes"])

# --- LÓGICA CHAT ---
with tab_chat:
    st.markdown("<div class='mic-container-top'>", unsafe_allow_html=True)
    audio_data = mic_recorder(
        start_prompt="🎤 Iniciar Grabación de Voz",
        stop_prompt="🛑 Detener Grabación",
        just_once=False,
        key="mic_main_btn"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Procesar audio
    if audio_data:
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
                st.session_state.messages.append({"role": "user", "content": transcription.text})

                context_text = ""
                if st.session_state.get("retriever"):
                    docs = st.session_state.retriever.invoke(transcription.text)
                    if docs:
                        context_text = "\n\n".join([d.page_content for d in docs])

                full_prompt = SYSTEM_PROMPT
                if context_text:
                    full_prompt += f"\n\nContexto:\n{context_text}"

                formatted_messages = [{"role": "system", "content": full_prompt}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant", 
                    messages=formatted_messages
                )

                ai_response = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            st.error(f"Error de audio: {e}")

    # Input de chat dentro de la pestaña
    if prompt := st.chat_input("Escribe tu mensaje, joven josefino..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        context_text = ""
        if st.session_state.get("retriever"):
            docs = st.session_state.retriever.invoke(prompt)
            if docs:
                context_text = "\n\n".join([d.page_content for d in docs])

        full_prompt = SYSTEM_PROMPT
        if context_text:
            full_prompt += f"\n\nContexto:\n{context_text}"

        formatted_messages = [{"role": "system", "content": full_prompt}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=formatted_messages
            )
            ai_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            st.error(f"Error: {e}")

    # Contenedor de chat
    st.markdown("<div class='fixed-chat-wrapper'>", unsafe_allow_html=True)
    chat_container = st.container(height=450, key="chat_container")

    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] != "system":
                avatar = "🦅" if message["role"] == "assistant" else "👤"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and voice_enabled:
                        components.html(
                            get_audio_button_html(message["content"], f"audio_{i}"),
                            height=50,
                        )

    st.markdown("</div>", unsafe_allow_html=True)


# --- LÓGICA GENERADOR DE IMÁGENES ---
with tab_img:
    st.markdown("### 🎨 Taller de Creación Josefino")
    st.write("Describe la imagen que deseas crear. Sé específico con los detalles, estilos y colores.")
    
    img_col1, img_col2 = st.columns([3, 1])
    with img_col1:
        img_prompt = st.text_area("Descripción de la imagen", placeholder="Ej: Un águila majestuosa volando sobre los volcanes de México al atardecer, estilo arte digital...", height=100)
    
    with img_col2:
        img_size = st.selectbox("Tamaño", ["1024x1024", "512x512", "1792x1024"])
        img_quality = st.selectbox("Calidad", ["standard", "hd"]) if img_size != "512x512" else "standard"
        img_style = st.selectbox("Estilo", ["vivid", "natural"])

    if st.button("✨ Generar Imagen", use_container_width=True):
        if not openai_img_key:
            st.error("🚫 Necesitas ingresar una API Key de OpenAI en la barra lateral para generar imágenes.")
        elif not img_prompt:
            st.warning("✏️ Por favor escribe una descripción.")
        else:
            with st.spinner("Creando tu obra maestra..."):
                try:
                    # Cliente OpenAI temporal para imágenes (apunta a api.openai.com por defecto)
                    img_client = OpenAI(api_key=openai_img_key)
                    
                    response_img = img_client.images.generate(
                        model="dall-e-3",
                        prompt=f"{img_prompt}. Estilo: {img_style}",
                        size=img_size,
                        quality=img_quality,
                        n=1,
                    )
                    
                    image_url = response_img.data[0].url
                    st.session_state['last_img_url'] = image_url
                    st.session_state['last_img_prompt'] = img_prompt
                    
                except Exception as e:
                    st.error(f"Error al crear la imagen: {e}")

    # Mostrar imagen si existe en sesión
    if 'last_img_url' in st.session_state:
        st.markdown("<div class='generated-image-container'>", unsafe_allow_html=True)
        st.image(st.session_state['last_img_url'], caption=st.session_state.get('last_img_prompt', ''), use_column_width=True)
        
        # Botón de descarga
        try:
            img_response = requests.get(st.session_state['last_img_url'])
            if img_response.status_code == 200:
                st.download_button(
                    label="📥 Descargar Imagen",
                    data=img_response.content,
                    file_name="juventud_2_0_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        except Exception:
            pass
        st.markdown("</div>", unsafe_allow_html=True)
```