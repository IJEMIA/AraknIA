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

# CSS DINÁMICO Y MODERNO
css_juventud = """
<style>
    /* ═══════════════════════════════════════════════════════════════
       IMPORTACIÓN DE FUENTES
       ═══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800;900&family=Inter:wght@300;400;500;600&display=swap');

    /* ═══════════════════════════════════════════════════════════════
       OCULTAR ELEMENTOS INNECESARIOS DE STREAMLIT
       ═══════════════════════════════════════════════════════════════ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    
    /* ═══════════════════════════════════════════════════════════════
       VARIABLES CSS Y FONDO ANIMADO
       ═══════════════════════════════════════════════════════════════ */
    :root {
        --josefino-dark: #022c22;
        --josefino-darker: #011a14;
        --josefino-green: #052e16;
        --josefino-gold: #facc15;
        --josefino-gold-light: #fde68a;
        --josefino-mint: #a7f3d0;
        --josefino-lime: #4ade80;
    }
    
    /* Fondo principal con gradientes animados */
    .stApp {
        background: linear-gradient(135deg, #011a14 0%, #022c22 40%, #052e16 100%);
        color: #f0fdf4;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Efecto de partículas flotantes */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 30%, rgba(74, 222, 128, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(250, 204, 21, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(74, 222, 128, 0.04) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
        animation: backgroundPulse 15s ease-in-out infinite;
    }
    
    @keyframes backgroundPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }

    /* ═══════════════════════════════════════════════════════════════
       CABECERA PRINCIPAL CON ANIMACIÓN
       ═══════════════════════════════════════════════════════════════ */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        position: relative;
        z-index: 10;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Título con gradiente animado */
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: clamp(2.5rem, 8vw, 4rem);
        background: linear-gradient(135deg, #4ade80 0%, #facc15 50%, #4ade80 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: 2px;
        animation: shimmer 4s linear infinite;
        text-shadow: 0 0 40px rgba(250, 204, 21, 0.3);
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #a7f3d0;
        font-size: clamp(0.875rem, 2vw, 1.1rem);
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 0.5rem;
        animation: fadeIn 1s ease-out 0.3s both;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Águila animada */
    .eagle-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
        animation: eagleFloat 4s ease-in-out infinite;
    }
    
    @keyframes eagleFloat {
        0%, 100% { transform: translateY(0) rotate(-2deg); }
        50% { transform: translateY(-10px) rotate(2deg); }
    }

    /* ═══════════════════════════════════════════════════════════════
       BARRA LATERAL (SIDEBAR) RE DISEÑADA
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #022c22 0%, #011a14 100%) !important;
        border-right: 1px solid rgba(250, 204, 21, 0.15);
        position: relative;
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 0%, rgba(250, 204, 21, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    [data-testid="stSidebar"] > div {
        position: relative;
        z-index: 1;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 1.25rem;
        animation: slideInRight 0.6s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* ═══════════════════════════════════════════════════════════════
       BURBUJAS DE CHAT RE DISEÑADAS
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.08) 0%, rgba(74, 222, 128, 0.02) 100%);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 20px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
        animation: bubbleIn 0.4s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(74, 222, 128, 0.15);
    }
    
    @keyframes bubbleIn {
        from {
            opacity: 0;
            transform: translateY(20px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    /* Avatar del asistente */
    [data-testid="stChatMessageAvatar-assistant"] {
        background: linear-gradient(135deg, #4ade80 0%, #facc15 100%);
        border-radius: 50%;
        padding: 8px;
    }
    
    [data-testid="stChatMessageContent"] {
        color: #f0fdf4 !important;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        line-height: 1.7;
    }

    [data-testid="stChatMessageContent"] p {
        color: #f0fdf4 !important;
        margin: 0;
    }

    /* ═══════════════════════════════════════════════════════════════
       INPUT DE TEXTO RE DISEÑADO
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stChatInput"] {
        border: 2px solid rgba(250, 204, 21, 0.3) !important;
        border-radius: 24px !important;
        background: linear-gradient(135deg, rgba(2, 44, 34, 0.95) 0%, rgba(5, 46, 22, 0.9) 100%) !important;
        backdrop-filter: blur(15px);
        transition: all 0.4s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #facc15 !important;
        box-shadow: 0 0 40px rgba(250, 204, 21, 0.2), 0 4px 20px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    [data-testid="stChatInput"] textarea {
        color: #f0fdf4 !important;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        padding: 1rem;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #a7f3d0 !important;
        opacity: 0.8;
    }

    /* Botón de envío del input */
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        transition: all 0.3s ease;
    }

    [data-testid="stChatInput"] button:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 0 25px rgba(250, 204, 21, 0.5);
    }

    /* ═══════════════════════════════════════════════════════════════
       BOTONES NORMALES Y MICRÓFONO
       ═══════════════════════════════════════════════════════════════ */
    .stButton button, .st-key-mic_btn button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 50%, #facc15 100%) !important;
        background-size: 200% auto !important;
        color: #022c22 !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        border-radius: 50px !important;
        border: none !important;
        padding: 0.875rem 1.75rem !important;
        width: 100%;
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }

    .stButton button:hover, .st-key-mic_btn button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 35px rgba(250, 204, 21, 0.4);
        background-position: right center !important;
    }
    
    .stButton button:active, .st-key-mic_btn button:active {
        transform: translateY(-1px) scale(0.98);
    }

    /* ═══════════════════════════════════════════════════════════════
       TÍTULOS DEL SIDEBAR
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] h2 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        color: #facc15 !important;
        font-size: 1.5rem !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
        position: relative;
    }
    
    [data-testid="stSidebar"] h2::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #facc15, transparent);
        border-radius: 2px;
    }

    [data-testid="stSidebar"] h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        color: #a7f3d0 !important;
        font-size: 1rem !important;
        margin-bottom: 0.75rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ═══════════════════════════════════════════════════════════════
       ALERTAS Y MENSAJES DE ESTADO
       ═══════════════════════════════════════════════════════════════ */
    .stAlert, [data-testid="stSuccess"], [data-testid="stInfo"], [data-testid="stWarning"] {
        background: linear-gradient(135deg, rgba(5, 46, 22, 0.8) 0%, rgba(2, 44, 34, 0.6) 100%) !important;
        border: none !important;
        border-left: 4px solid #facc15 !important;
        border-radius: 12px !important;
        color: #f0fdf4 !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stAlert:hover, [data-testid="stSuccess"]:hover, [data-testid="stInfo"]:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(250, 204, 21, 0.1);
    }
    
    [data-testid="stSuccess"] {
        border-left-color: #4ade80 !important;
    }
    
    [data-testid="stWarning"] {
        border-left-color: #fbbf24 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       FILE UPLOADER PERSONALIZADO
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stFileUploader"] {
        background: rgba(2, 44, 34, 0.5);
        border: 2px dashed rgba(250, 204, 21, 0.3);
        border-radius: 16px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(250, 204, 21, 0.6);
        background: rgba(2, 44, 34, 0.8);
    }
    
    [data-testid="stFileUploader"] section {
        color: #a7f3d0 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       CHECKBOX PERSONALIZADO
       ═══════════════════════════════════════════════════════════════ */
    .stCheckbox {
        background: rgba(5, 46, 22, 0.4);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(250, 204, 21, 0.1);
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        border-color: rgba(250, 204, 21, 0.3);
        background: rgba(5, 46, 22, 0.6);
    }
    
    .stCheckbox label {
        color: #a7f3d0 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       EXPANDER PERSONALIZADO
       ═══════════════════════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(5, 46, 22, 0.6) 0%, rgba(2, 44, 34, 0.4) 100%);
        border: 1px solid rgba(250, 204, 21, 0.2);
        border-radius: 12px !important;
        color: #a7f3d0 !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #facc15;
        background: linear-gradient(135deg, rgba(5, 46, 22, 0.8) 0%, rgba(2, 44, 34, 0.6) 100%);
    }
    
    .streamlit-expanderContent {
        background: rgba(2, 44, 34, 0.3);
        border: 1px solid rgba(74, 222, 128, 0.1);
        border-top: none;
        border-radius: 0 0 12px 12px;
    }

    /* ═══════════════════════════════════════════════════════════════
       SCROLLBAR PERSONALIZADA
       ═══════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(2, 44, 34, 0.5);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #facc15, #fbbf24);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #fde68a, #facc15);
    }

    /* ═══════════════════════════════════════════════════════════════
       SPINNER Y LOADING PERSONALIZADO
       ═══════════════════════════════════════════════════════════════ */
    .stSpinner > div {
        border-color: #facc15 transparent transparent transparent !important;
    }
    
    .loading-message {
        font-family: 'Montserrat', sans-serif;
        color: #facc15;
        text-align: center;
        padding: 2rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ═══════════════════════════════════════════════════════════════
       PRINCIPIOS CARDS
       ═══════════════════════════════════════════════════════════════ */
    .principle-card {
        background: linear-gradient(135deg, rgba(5, 46, 22, 0.6) 0%, rgba(2, 44, 34, 0.4) 100%);
        border: 1px solid rgba(250, 204, 21, 0.15);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .principle-card:hover {
        border-color: rgba(250, 204, 21, 0.4);
        transform: translateX(8px);
        box-shadow: 0 5px 25px rgba(250, 204, 21, 0.1);
    }
    
    .principle-card p {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
    }

    /* ═══════════════════════════════════════════════════════════════
       TOAST/TOAST NOTIFICATIONS
       ═══════════════════════════════════════════════════════════════ */
    [data-testid="stToast"] {
        background: linear-gradient(135deg, rgba(2, 44, 34, 0.95) 0%, rgba(5, 46, 22, 0.9) 100%) !important;
        border: 1px solid #facc15 !important;
        border-radius: 16px !important;
        backdrop-filter: blur(15px);
    }
    
    [data-testid="stToast"] p {
        color: #f0fdf4 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       RESPONSIVE - MÓVIL PRIMERO
       ═══════════════════════════════════════════════════════════════ */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem !important;
            letter-spacing: 1px;
        }
        
        .subtitle {
            font-size: 0.75rem;
            letter-spacing: 2px;
        }
        
        [data-testid="stChatMessage"] {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        [data-testid="stSidebar"] h2 {
            font-size: 1.25rem !important;
        }
        
        .stButton button, .st-key-mic_btn button {
            padding: 0.75rem 1.25rem !important;
            font-size: 0.8rem;
        }
    }

    /* ═══════════════════════════════════════════════════════════════
       REDUCED MOTION - ACCESIBILIDAD
       ═══════════════════════════════════════════════════════════════ */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
"""
st.markdown(css_juventud, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HTML PARA EL HEADER CON ÁGUILA SVG ANIMADA
# ═══════════════════════════════════════════════════════════════

header_html = """
<div class="main-header">
    <div class="eagle-container">
        <svg viewBox="0 0 64 64" width="72" height="72" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="eagleGradient" x1="8" y1="8" x2="56" y2="56" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#4ade80"/>
                    <stop offset="100%" stop-color="#facc15"/>
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <path d="M32 8L8 24L16 28L8 40L20 36L16 52L32 40L48 52L44 36L56 40L48 28L56 24L32 8Z" 
                  fill="url(#eagleGradient)" 
                  stroke="#facc15" 
                  stroke-width="1.5"
                  filter="url(#glow)"/>
            <circle cx="26" cy="24" r="3" fill="#022c22"/>
            <circle cx="38" cy="24" r="3" fill="#022c22"/>
            <path d="M28 32L32 36L36 32" stroke="#022c22" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    <h1 class="main-title">JUVENTUD 2.0</h1>
    <p class="subtitle">Tu Guía Josefina</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

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
- **Tono**: Cordial, amable y ligeramente paternalista. Eres como un mentor sabio y cercano.
- **Interacción**: Resalta siempre el lado humano. Muestra empatía.
- **Usuarios**: Dirígete a ellos como "Josefino", "Josefina" o "Joven Josefino".
- **Sobre el Instituto**: Tienes conocimiento sobre programas del IJEM.

RECUERDA: Eres el rostro digital de una comunidad que busca el bien común. ¡Vuela alto como el águila!
"""

# ═══════════════════════════════════════════════════════════════
# FUNCIONES PARA CARGAR PDFs Y ZIPS
# ═══════════════════════════════════════════════════════════════

DOCS_FOLDER = "documentos"

def create_retriever_from_paths(pdf_paths):
    """Función reutilizable que crea el vectorstore a partir de una lista de rutas de archivos"""
    all_docs = []
    valid_files = []
    
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            filename = os.path.basename(pdf_path)
            for doc in docs: 
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            valid_files.append(filename)
        except Exception as e:
            print(f"Error leyendo {pdf_path}: {e}")
            
    if not all_docs: 
        return None, []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore.as_retriever(), valid_files

@st.cache_resource
def load_knowledge_base():
    """Carga inicial desde la carpeta 'documentos'"""
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files: 
        return None, []
    return create_retriever_from_paths(pdf_files)

# ═══════════════════════════════════════════════════════════════
# INICIALIZACIÓN CON ANIMACIÓN
# ═══════════════════════════════════════════════════════════════

if "initialized" not in st.session_state:
    init_container = st.empty()
    init_messages = [
        ("🦅", "Desplegando alas..."),
        ("💛", "Sincronizando valores josefinos..."),
        ("✨", "Juventud 2.0 lista, Profe Adrián")
    ]
    
    for icon, msg in init_messages:
        init_container.markdown(
            f"""
            <div class="loading-message">
                <span style="font-size: 2rem;">{icon}</span><br>
                <span>{msg}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        time.sleep(0.6)
    init_container.empty()
    st.session_state.initialized = True

if "retriever" not in st.session_state:
    with st.spinner("Leyendo archivos de la comunidad..."):
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

if "messages" not in st.session_state: 
    st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR (Panel de Control Josefino)
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    # Título del Sidebar
    st.markdown("<h2>🦅 Panel Josefino</h2>", unsafe_allow_html=True)
    
    # --- MICRÓFONO ---
    st.markdown("#### 🎙️ Comando de Voz")
    audio_data = mic_recorder(
        start_prompt="🎤 Iniciar Grabación",
        stop_prompt="🛑 Detener",
        just_once=False,
        use_container_width=True,
        key="mic_sidebar_stable"
    )
    
    st.markdown("---")
    
    # Configuración
    st.markdown("#### ⚙️ Configuración")
    voice_enabled = st.checkbox("Activar voz de Juventud 2.0", value=True)

    st.markdown("---")
    
    # --- CARGADOR DE ZIPS ---
    st.markdown("#### 📦 Cargar Archivos ZIP")
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs", type="zip", key="zip_uploader")
    
    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            st.session_state.processed_zip_name = uploaded_zip.name
            
            with st.spinner(f"Procesando {uploaded_zip.name}..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        temp_zip_path = os.path.join(temp_dir, "temp.zip")
                        with open(temp_zip_path, "wb") as f:
                            f.write(uploaded_zip.getbuffer())
                        
                        with zipfile.ZipFile(temp_zip_path, 'r') as z:
                            z.extractall(temp_dir)
                        
                        extracted_pdfs = glob.glob(os.path.join(temp_dir, "**", "*.pdf"), recursive=True)
                        
                        if extracted_pdfs:
                            new_retriever, new_files = create_retriever_from_paths(extracted_pdfs)
                            
                            if new_retriever:
                                st.session_state.retriever = new_retriever
                                st.session_state.loaded_files = new_files
                                st.success(f"✅ {len(new_files)} PDFs cargados del ZIP.")
                            else:
                                st.error("No se pudieron procesar los PDFs dentro del ZIP.")
                        else:
                            st.warning("El ZIP no contenía archivos PDF.")
                            
                    except Exception as e:
                        st.error(f"Error al descomprimir: {e}")
    
    st.markdown("---")
    
    # Archivos actuales
    st.markdown("#### 📚 Archivos de la Comunidad")
    if st.session_state.get("loaded_files"):
        st.success(f"🟢 {len(st.session_state.loaded_files)} Archivos Activos")
        with st.expander("Ver lista de archivos"):
            for f in st.session_state.loaded_files: 
                st.write(f"📄 {f}")
    else:
        st.warning("🔴 Repositorio Vacío")
    
    st.markdown("---")
    
    # Principios con cards personalizados
    st.markdown("#### 📜 Principios Rectores")
    
    principle_1 = """
    <div class="principle-card">
        <p style="color: #4ade80;">✨ Hacer siempre y en todo lo mejor</p>
    </div>
    """
    principle_2 = """
    <div class="principle-card">
        <p style="color: #facc15;">🚀 Adelante, siempre adelante</p>
    </div>
    """
    principle_3 = """
    <div class="principle-card">
        <p style="color: #a7f3d0;">🛠️ Estar siempre útilmente ocupados</p>
    </div>
    """
    
    st.markdown(principle_1, unsafe_allow_html=True)
    st.markdown(principle_2, unsafe_allow_html=True)
    st.markdown(principle_3, unsafe_allow_html=True)

    # Créditos
    st.markdown("""
        <br>
        <p style='text-align:center; font-size:0.8rem; color:#555; font-family: Inter, sans-serif;'>
            Diseñado por el Profe Adrián<br>
            <span style='font-size:0.7rem; color:#444;'>Instituto de la Juventud del Estado de México</span>
        </p>
    """, unsafe_allow_html=True)

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

    full_prompt_content = SYSTEM_PROMPT + f"\n\n## REGISTROS ACCEDIDOS:\n{context_text}" if context_text else SYSTEM_PROMPT + "\n\n(No se hallaron registros específicos)."

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

# Procesamiento de Audio
if audio_data:
    audio_bytes = audio_data['bytes']
    audio_format = audio_data['format']
    with st.spinner("🔊 Procesando tu voz..."):
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
                st.toast(f"🎤 Escuché: {transcribed_text}", icon="✅")
                process_user_input(transcribed_text)
        except Exception as e:
            st.error(f"⚠️ Error en audio: {str(e)}")

# Historial de Chat
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = "🦅" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Input de Texto
if prompt := st.chat_input("Escribe tu mensaje, joven josefino..."):
    process_user_input(prompt)
