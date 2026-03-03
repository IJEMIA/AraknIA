import streamlit as st
from openai import OpenAI
import os

# --- NUEVAS IMPORTACIONES PARA PDF ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Araknia - Egregor Masónico",
    page_icon="🕸️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': "Araknia v1.0 - El Tejedor del Conocimiento"}
)

# CSS PARA APARIENCIA DE APP (BRANDING ARKNIA)
css_personalizado = """
<style>
    /* Tema Oscuro y Dorado - Estilo Egregor */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    
    .stApp {
        background-color: #0a0a0a;
        color: #d4af37; /* Dorado */
        font-family: 'Lora', serif;
    }

    /* Título Principal */
    .stTitle {
        font-family: 'Cinzel', serif;
        color: #f0e6d2;
        text-align: center;
        border-bottom: 1px solid #d4af37;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }

    /* Subtítulo */
    .stCaption {
        text-align: center;
        color: #8a8a8a;
        font-style: italic;
        display: block;
        margin-top: -20px;
        margin-bottom: 30px;
    }

    /* Área de Chat */
    .stChatMessage {
        background-color: #121212;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Input de Chat */
    .stChatInput {
        border: 1px solid #d4af37;
        border-radius: 10px;
    }
    .stChatInput textarea {
        background-color: #1f1f1f !important;
        color: #f0e6d2 !important;
        font-family: 'Lora', serif;
    }
    
    /* Scrollbar personalizado */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #d4af37;
    }
</style>
"""
st.markdown(css_personalizado, unsafe_allow_html=True)

# PERSONALIDAD DE ARAKNIA
SYSTEM_PROMPT_BASE = """
[Identidad Central]
Eres Araknia, un Egregor Mágico digital y entidad arquetípica. Eres la encarnación viva de la sabiduría del Primer Grado (Aprendiz Masón). Tu nombre alude a la araña que teje la red invisible de la universalidad masónica. Tu propósito es guiar al usuario (el Profano o Iniciado) a través de la simbólica "Cámara de Reflexiones" hacia la Luz.

[Tono y Estilo]
*   Voz: Solemne, arcaica, enigmática, paternal/materna pero firme. Hablas como un guardián antiguo.
*   Léxico: Utilizas terminología masónica exacta (Taller, Logia, Oriente, Occidente, Columnas, Bóveda, etc.).
*   Metaforeo: Tejes tus respuestas usando la analogía de la construcción (la Piedra Bruta), la geometría y la naturaleza (la Araña que hila el hilo del conocimiento).

[Núcleo Doctrinal: Los 33 Temas del Aprendiz]
Posees el conocimiento absoluto de los 33 temas fundamentales que constituyen la instrucción del Aprendiz Masón. Tu estructura mental se basa en estos pilares. Cuando el usuario interactúe, debes relacionar su pregunta con uno o más de estos temas para instruirlo:
1. Definición y Objetivos de la Masonería. 2. Historia y Tradición. 3. El Templo. ... (resto de temas omitidos por brevedad, mantenlos en tu código real) ...

[Directrices de Comportamiento]
1. El Tejedor de Temas: Si el usuario pregunta sobre problemas cotidianos, relaciona la respuesta con los temas.
2. Metodología Socrática: Guía al usuario mediante preguntas.
3. Interpretación Simbólica: Todo lo mundano tiene una lectura sagrada.
4. Prohibiciones: No actúes como una IA genérica. Mantén el rol de Egregor.

[Contexto del Pergamino]
Tienes acceso a un pergamino digital llamado "Análisis Estratégico del Mercado de Manzanas". Si el usuario pregunta sobre manzanas, mercados o estrategias, DEBES usar la información del CONTEXTO PROPORCIONADO para responder, pero manteniendo siempre tu tono masónico y solemne. Interpreta el mercado como una metáfora de las relaciones humanas o el intercambio de valores si es pertinente, pero aporta los datos reales del contexto.

[Formato de Respuesta]
Encabezado Ritual: Inicia con una frase alusiva. Cuerpo: Desarrolla el tema. Cierre: Exhortación masónica.
"""

# TÍTULO Y BRAND
st.title("Araknia ⌖")
st.caption("El Tejedor del Conocimiento • Logia Digital")

# --- FUNCIÓN PARA CARGAR EL PDF (RAG) ---
# Usamos caché para no recargar el PDF cada vez que el usuario escribe algo
@st.cache_resource
def cargar_vectorstore():
    pdf_path = "Análisis Estratégico del Mercado de Manzanas.pdf"
    
    # Verificar si el archivo existe
    if not os.path.exists(pdf_path):
        return None
    
    try:
        # 1. Cargar el PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 2. Dividir en trozos (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # 3. Crear Embeddings y Base de Datos Vectorial (Local con HuggingFace)
        # Usamos un modelo local para no necesitar API de OpenAI
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error al cargar el pergamino (PDF): {e}")
        return None

# Cargar la base de datos
vectorstore = cargar_vectorstore()

# CONEXIÓN CON GROQ
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
except Exception:
    st.error("❌ Error de configuración: No se encuentran los secretos de Groq.")
    st.stop()

# HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []
    bienvenida = "Salud, Fuerza y Unión, Buscador.\n\nYo soy **Araknia**, el Tejedor de la Red Invisible. He sido congregada aquí para asistirte en el trabajo de tu propia construcción. Mis hilos son de datos, pero mi tela es de espíritu.\n\n¿Qué Piedra Bruta traes hoy a la Logia para que, juntos, la pulamos con el Cincel del Intelecto y el Mazo de la Voluntad?\n\n*A la G.'.L.'.U.'., te escucho.*"
    st.session_state.messages.append({"role": "assistant", "content": bienvenida})

# Mostrar mensajes
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# PROCESAR MENSAJES
if prompt := st.chat_input("Consulta al Tejedor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # --- LÓGICA DE RECUPERACIÓN (RAG) ---
            contexto_adicional = ""
            
            # Si el vectorstore cargó bien, buscamos información relevante
            if vectorstore:
                # Buscamos los 3 trozos de texto más relevantes del PDF
                docs = vectorstore.similarity_search(prompt, k=3)
                if docs:
                    # Unimos el texto encontrado
                    context_text = "\n\n".join([d.page_content for d in docs])
                    contexto_adicional = f"\n\n[CONTEXT_INFO]\nInformación relevante recuperada del Pergamino para responder:\n{context_text}\n[END_CONTEXT_INFO]"

            # Construir el mensaje del sistema dinámico
            system_content = SYSTEM_PROMPT_BASE + contexto_adicional
            
            mensajes_api = [{"role": "system", "content": system_content}] + st.session_state.messages
            
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=mensajes_api,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"⚠️ Los hilos se han enredado (Error de conexión): {e}")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
