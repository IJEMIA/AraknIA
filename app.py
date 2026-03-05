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
    st.error("Faltan librerías. Por favor instala: pip install langchain-community langchain-text-splitters faiss-cpu pypdf sentence-transformers")
    st.stop()

# CONFIGURACIÓN DE PÁGINA (Debe ser lo primero)
st.set_page_config(
    page_title="Juventud 2.0",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': "Creado por el Profe Adrián para la comunidad Josefina"}
)

# CSS DINÁMICO Y MODERNO (CORREGIDO PARA VISIBILIDAD)
css_juventud = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800;900&family=Inter:wght@300;400;500;600&display=swap');

    /* OCULTAR ELEMENTOS INNECESARIOS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    
    /* FONDO Y BODY */
    .stApp {
        background: linear-gradient(135deg, #011a14 0%, #022c22 40%, #052e16 100%);
        color: #f0fdf4;
    }
    
    /* Capa de brillo superior (Z-Index bajo para no tapar contenido) */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            radial-gradient(circle at 20% 30%, rgba(74, 222, 128, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(250, 204, 21, 0.06) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }

    /* IMPORTANTE: Asegurar que el contenido principal esté encima */
    section[data-testid="stMain"] {
        position: relative;
        z-index: 1 !important;
    }
    
    /* HEADER ANIMADO */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: clamp(2.5rem, 8vw, 4rem);
        background: linear-gradient(135deg, #4ade80 0%, #facc15 50%, #4ade80 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
        animation: shimmer 4s linear infinite;
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
    }
    
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

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #022c22 0%, #011a14 100%) !important;
        border-right: 1px solid rgba(250, 204, 21, 0.15);
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        color: #facc15 !important;
        text-align: center;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #a7f3d0 !important;
        font-family: 'Montserrat', sans-serif !important;
    }

    /* CHAT BURBUJAS */
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

    /* INPUT DE CHAT */
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

    /* BOTONES NORMALES */
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

    /* ALERTAS */
    .stAlert, [data-testid="stSuccess"], [data-testid="stInfo"], [data-testid="stWarning"] {
        background: rgba(2, 44, 34, 0.8) !important;
        border-left: 4px solid #facc15 !important;
        border-radius: 12px !important;
        color: #f0fdf4 !important;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #022c22; }
    ::-webkit-scrollbar-thumb { background: #facc15; border-radius: 10px; }
    
    /* PRINCIPIOS CARDS */
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
# HTML PARA EL HEADER
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
            </defs>
            <path d="M32 8L8 24L16 28L8 40L20 36L16 52L32 40L48 52L44 36L56 40L48 28L56 24L32 8Z" 
                  fill="url(#eagleGradient)" stroke="#facc15" stroke-width="1.5"/>
            <circle cx="26" cy="24" r="3" fill="#022c22"/>
            <circle cx="38" cy="24" r="3" fill="#022c22"/>
            <path d="M28 32L32 36L36 32" stroke="#022c22" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
    </div>
    <h1 class="main-title">JUVENTUD 2.0</h1>
    <p class="subtitle">Tu Guía Josefina</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE API KEY (MANEJO SEGURO)
# ═══════════════════════════════════════════════════════════════

# Intentar cargar desde secrets
api_key = None
try:
    api_key = st.secrets["groq"]["api_key"]
except:
    pass # Si falla, mostraremos input en el sidebar

# Si no hay secret, pedir en sidebar
if not api_key:
    with st.sidebar:
        st.markdown("### ⚠️ Configuración Inicial")
        api_key_input = st.text_input("Ingresa tu API Key de Groq", type="password")
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
