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

# CSS PROFESIONAL JOSEFINO (VERDE Y ORO)
css_juventud = """
<style>
    /* IMPORTACIÓN DE FUENTES */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

    /* OCULTAR ELEMENTOS INNECESARIOS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    /* FONDO DE LA APP (Degradado Elegante) */
    .stApp {
        background: linear-gradient(135deg, #022c22 0%, #052e16 100%);
        color: #ffffff;
    }

    /* CABECERA PRINCIPAL */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    
    h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(to right, #4ade80, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: 1px;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #a7f3d0;
        font-size: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 5px;
    }

    /* BARRA LATERAL (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #022c22;
        border-right: 1px solid rgba(250, 204, 21, 0.2);
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 1rem;
    }

    /* BURBUJAS DE CHAT */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(250, 204, 21, 0.15);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    [data-testid="stChatMessageContent"] {
        color: #f0fdf4;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
    }

    [data-testid="stChatMessageContent"] p {
        color: #f0fdf4 !important;
    }

    /* INPUT DE TEXTO (Abajo) */
    .stChatInput {
        border: 1px solid #facc15 !important;
        border-radius: 12px;
        background-color: rgba(5, 46, 22, 0.8) !important;
    }
    
    .stChatInput textarea {
        color: white !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stChatInput textarea::placeholder {
        color: #a7f3d0 !important;
    }

    /* BOTÓN DE ENVÍO DEL INPUT */
    .stChatInput button {
        background-color: #facc15 !important;
        color: #022c22 !important;
    }

    /* BOTONES NORMALES