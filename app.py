import streamlit as st
from openai import OpenAI
import os

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

# CSS (mantener el mismo)
css_personalizado = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    .stApp {background-color: #0a0a0a; color: #d4af37; font-family: 'Lora', serif;}
    .stTitle {font-family: 'Cinzel', serif; color: #f0e6d2; text-align: center; border-bottom: 1px solid #d4af37; padding-bottom: 20px; margin-bottom: 30px;}
    .stCaption {text-align: center; color: #8a8a8a; font-style: italic; display: block; margin-top: -20px; margin-bottom: 30px;}
    .stChatMessage {background-color: #121212; border: 1px solid #333; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .stChatInput {border: 1px solid #d4af37; border-radius: 10px;}
    .stChatInput textarea {background-color: #1f1f1f !important; color: #f0e6d2 !important; font-family: 'Lora', serif;}
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-track {background: #0a0a0a;}
    ::-webkit-scrollbar-thumb {background: #555; border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #d4af37;}
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

[Contexto del Pergamino]
Tienes acceso a un pergamino digital. Si el usuario pregunta sobre temas relacionados, usa la información del CONTEXTO PROPORCIONADO para responder, manteniendo siempre tu tono masónico y solemne.

[Formato de Respuesta]
Encabezado Ritual: Inicia con una frase alusiva. Cuerpo: Desarrolla el tema. Cierre: Exhortación masónica.
"""

st.title("Araknia ⌖")
st.caption("El Tejedor del Conocimiento • Logia Digital")

# --- FUNCIÓN CORREGIDA PARA CARGAR PDF ---
@st.cache_resource
def cargar_vectorstore():
    pdf_path = "Análisis Estratégico del Mercado de Manzanas.pdf"
    
    # 1. Verificar existencia del archivo
    if not os.path.exists(pdf_path):
        st.warning(f"📜 El pergamino no fue encontrado en: {pdf_path}")
        return None
    
    # Verificar que no esté vacío
    if os.path.getsize(pdf_path) == 0:
        st.warning("📜 El pergamino existe pero está vacío.")
        return None
    
    try:
        # 2. Cargar PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 3. VALIDACIÓN CRÍTICA - Verificar que hay contenido
        if not documents or len(documents) == 0:
            st.warning("📜 El pergamino no contiene texto extraíble. Posiblemente es una imagen escaneada.")
            return None
        
        # Verificar que el texto no esté vacío
        total_chars = sum(len(doc.page_content) for doc in documents)
        if total_chars == 0:
            st.warning("📜 El pergamino se cargó pero no tiene contenido de texto.")
            return None
        
        st.info(f"📜 Pergamino cargado: {len(documents)} páginas, {total_chars} caracteres")
        
        # 4. Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 5. VALIDACIÓN CRÍTICA - Verificar splits
        if not splits or len(splits) == 0:
            st.warning("📜 Error: No se pudieron crear fragmentos del documento.")
            return None
        
        st.info(f"📜 Fragmentos creados: {len(splits)} trozos de conocimiento")
        
        # 6. Crear embeddings
        with st.spinner("🕸️ Tejiendo los hilos del conocimiento..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 7. Crear vectorstore
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory="./chroma_db"  # Opcional: persistir en disco
            )
        
        st.success("✓ Pergamino procesado exitosamente")
        return vectorstore
        
    except Exception as e:
        st.error(f"📜 Error al procesar el pergamino: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Cargar vectorstore
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
    bienvenida = "Salud, Fuerza y Unión, Buscador.\n\nYo soy **Araknia**, el Tejedor de la Red Invisible. He sido congregada aquí para asistirte en el trabajo de tu propia construcción.\n\n¿Qué Piedra Bruta traes hoy a la Logia para que, juntos, la pulamos?\n\n*A la G.'.L.'.U.'., te escucho.*"
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
            contexto_adicional = ""
            
            if vectorstore:
                docs = vectorstore.similarity_search(prompt, k=3)
                if docs:
                    context_text = "\n\n".join([d.page_content for d in docs])
                    contexto_adicional = f"\n\n[CONTEXT_INFO]\n{context_text}\n[END_CONTEXT_INFO]"

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
            st.error(f"⚠️ Los hilos se han enredado: {e}")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()