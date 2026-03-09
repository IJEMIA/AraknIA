import streamlit as st
from openai import OpenAI
import os
import glob
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io
import zipfile

# IMPORTACIONES PARA LANGCHAIN
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Faltan librerías. Instala con: pip install langchain-community langchain-text-splitters faiss-cpu pypdf sentence-transformers")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE CARPETA (RUTA ABSOLUTA)
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "documentos")

def load_knowledge_base():
    if not os.path.exists(DOCS_FOLDER):
        try:
            os.makedirs(DOCS_FOLDER)
        except OSError as e:
            st.error(f"Error al crear la carpeta 'documentos': {e}")
            return None, []

    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    
    if not pdf_files: 
        return None, []

    all_docs = []
    valid_files = []
    error_files = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if not docs:
                continue
            
            filename = os.path.basename(pdf_path)
            for doc in docs: 
                doc.metadata["source"] = filename
            
            all_docs.extend(docs)
            valid_files.append(filename)
        except Exception as e:
            error_files.append((os.path.basename(pdf_path), str(e)))

    if error_files:
        st.warning(f"⚠️ No se pudieron leer {len(error_files)} archivos.")

    if not all_docs: 
        return None, []

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore, valid_files
    except Exception as e:
        st.error(f"Error al procesar embeddings: {e}")
        return None, []

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

    /* INPUT CHAT */
    [data-testid="stChatInput"] {
        border: 2px solid rgba(250, 204, 21, 0.5) !important;
        border-radius: 24px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        background-color: transparent !important;
        color: #022c22 !important;
        font-weight: 500;
        font-size: 1rem !important;
        caret-color: #022c22;
    }
    
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder {
        color: #6b7280 !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #facc15 0%, #fbbf24 100%) !important;
        color: #022c22 !important;
        border-radius: 50% !important;
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

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2>🦅 Panel Josefino</h2>", unsafe_allow_html=True)
    
    st.markdown("#### ⚙️ Configuración Chat")
    if not api_key:
        api_key_input = st.text_input("API Key de Groq", type="password", key="api_key_input_groq")
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("Necesitas API Key de Groq para el chat.")
            st.info("Obtén una en: console.groq.com")
    else:
        st.success("API Key configurada ✅")
    
    voice_enabled = st.checkbox("Activar voz de Juventud 2.0", value=True)
    st.markdown("---")

    # Lógica de carga de archivos
    st.markdown("#### 📦 Cargar PDFs")
    st.caption(f"Los archivos se guardan en la carpeta `documentos`")
    
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs para añadir", type="zip", key="zip_uploader")

    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as z:
                    z.extractall(DOCS_FOLDER)
                st.session_state.processed_zip_name = uploaded_zip.name
                st.toast(f"✅ Archivos extraídos. Recargando...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error al descomprimir: {e}")

    st.markdown("---")

    st.markdown("#### 📚 Archivos Cargados")
    
    if st.button("🔄 Recargar Base de Datos", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if st.session_state.get("loaded_files"):
        st.success(f"🟢 {len(st.session_state.loaded_files)} Documentos Activos")
        with st.expander("Ver lista"):
            for f in st.session_state.loaded_files:
                st.write(f"📄 {f}")
    else:
        st.info(f"🔴 Repositorio Vacío. Añade PDFs en la carpeta o sube un ZIP.")

    st.markdown("---")

    st.markdown("#### 📜 Principios")
    st.markdown('<div class="principle-card"><p style="color: #4ade80;">✨ Hacer lo mejor</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="principle-card"><p style="color: #facc15;">🚀 Siempre adelante</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="principle-card"><p style="color: #a7f3d0;">🛠️ Útilmente ocupados</p></div>', unsafe_allow_html=True)

    st.markdown("<br><p style='text-align:center; font-size:0.8rem; color:#555;'>Diseñado por el Profe Adrián</p>", unsafe_allow_html=True)

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
# PERSONALIDAD Y MODO PLANEACIÓN
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT_BASE = """
Eres **Juventud 2.0**, una Inteligencia Artificial diseñada para la comunidad Josefina. Creada por el Profe Adrián.
Tus principios:
1. "Hacer siempre y en todo lo mejor".
2. "Adelante, siempre adelante".
3. "Estar siempre útilmente ocupados".
Tono: Cordial, amable, mentor. Dirígete al usuario como "Josefino/a".
"""

loaded_files_list_str = "No hay archivos cargados."
if st.session_state.get("loaded_files"):
    loaded_files_list_str = "\n".join([f"{i+1}. {fname}" for i, fname in enumerate(st.session_state.loaded_files)])

SYSTEM_PROMPT_PLANNING = f"""
Eres **Juventud 2.0 - Experto en Planeación Didáctica**.

**ARCHIVOS DISPONIBLES EN REPOSITORIO:**
{loaded_files_list_str}
-----------------------------------------

**REGLAS DE ORO (INQUEBRANTABLE):**
1. **VERACIDAD ABSOLUTA**: Toda información debe provenir **EXCLUSIVAMENTE** del texto proporcionado en la sección "Contexto de documentos".
2. **PROHIBIDO INVENTAR**: Si el contexto no muestra explícitamente el nombre de una unidad o contenido, responde: "No pude encontrar esa información específica en el archivo seleccionado".
3. Si el usuario selecciona un archivo por número, tu prioridad es buscar en el contexto de ese archivo.

**FLUJO DE INTERACCIÓN:**

**PASO 1: ACTIVACIÓN**
Si el usuario dice "vamos a planear":
1. Muestra la lista de archivos.
2. Pregunta: "¿Cuál es el **número** del programa a utilizar?"

**PASO 2: LECTURA Y LISTADO DE UNIDADES**
Cuando el usuario responda con un número:
1. Identifica el nombre del archivo correspondiente.
2. Analiza el **Contexto de documentos** provisto.
3. Extrae **EXCLUSIVAMENTE** del contexto los nombres de las Unidades, Módulos o Bloques.
4. Enumera las unidades encontradas (ej: 1. Unidad I: ..., 2. Unidad II: ...).
5. Pregunta: "¿Qué **número** de unidad(es) vamos a planear?"

**PASO 3: SESIONES**
Pregunta: "¿Cuántas **sesiones** en total necesita para esta planeación?"

**PASO 4: DIAS DE CLASE**
Pregunta: "¿Qué **días de la semana** se imparten las clases?"

**PASO 5: CRITERIOS DE EVALUACIÓN**
Pregunta: "¿Cuáles son los **criterios de evaluación** y cuántas sesiones destinaremos a ellos?"

**PASO 6: CALENDARIZACIÓN**
Pregunta: "Indica **fecha de inicio** y **fecha de término**. ¿Hay **días festivos** que debamos ignorar?"

**PASO 7: BORRADOR (5 EJEMPLOS)**
Genera 5 sesiones de ejemplo.

**PASO 8: GENERACIÓN FINAL**
Genera la planeación completa.

Mantén el tono "Josefino".
"""

# ═══════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE SESIÓN Y BASE DE DATOS
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state: 
    st.session_state.messages = []

if "planning_mode" not in st.session_state:
    st.session_state.planning_mode = False

if "vectorstore" not in st.session_state:
    vectorstore, loaded_files = load_knowledge_base()
    st.session_state.vectorstore = vectorstore
    st.session_state.loaded_files = loaded_files

# ═══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
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

def get_context_for_planning(user_input, vectorstore, loaded_files):
    selected_file_index = None
    if loaded_files:
        try:
            potential_index = int(user_input.strip()) - 1
            if 0 <= potential_index < len(loaded_files):
                selected_file_index = potential_index
        except ValueError:
            pass

    if selected_file_index is not None:
        target_filename = loaded_files[selected_file_index]
        try:
            docs = vectorstore.similarity_search(
                query="Unidades Bloques Contenido Temario Estructura Títulos", 
                k=30, 
                fetch_k=100, 
                filter={"source": target_filename}
            )
            if not docs:
                return f"El archivo {target_filename} parece estar vacío o no se pudo leer su estructura.", target_filename
            
            context_text = "\n\n---\n\n".join([f"Fragmento de {doc.metadata.get('source')}:\n{doc.page_content}" for doc in docs])
            return context_text, target_filename
        except Exception as e:
            return f"Error al leer {target_filename}: {e}", target_filename
    else:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            docs = retriever.invoke(user_input)
            return "\n\n---\n\n".join([f"Fuente: {doc.metadata.get('source', 'Desconocido')}\n{doc.page_content}" for doc in docs]), None
        except Exception as e:
            return "", None

# ═══════════════════════════════════════════════════════════════
# INTERFAZ DE CHAT
# ═══════════════════════════════════════════════════════════════

# CORRECCIÓN: Inicializamos audio_data antes de usarla para evitar NameError
audio_data = None

st.markdown("<div class='mic-container-top'>", unsafe_allow_html=True)
try:
    audio_data = mic_recorder(
        start_prompt="🎤 Iniciar Grabación de Voz",
        stop_prompt="🛑 Detener Grabación",
        just_once=False,
        key="mic_main_btn"
    )
except Exception as e:
    st.warning("El componente de micrófono no está disponible en este entorno.")
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
            prompt = transcription.text
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if "vamos a planear" in prompt.lower():
                st.session_state.planning_mode = True
                st.toast("📑 Modo Planeación Didáctica Activado")

            current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else SYSTEM_PROMPT_BASE
            
            context_text = ""
            if st.session_state.get("vectorstore"):
                context_text, _ = get_context_for_planning(prompt, st.session_state.vectorstore, st.session_state.loaded_files)

            full_prompt = current_prompt
            if context_text:
                full_prompt += f"\n\nContexto de documentos:\n{context_text}"

            formatted_messages = [{"role": "system", "content": full_prompt}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=formatted_messages
            )

            ai_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        st.error(f"Error de audio: {e}")

# Input de chat
if prompt := st.chat_input("Escribe tu mensaje, joven josefino..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "vamos a planear" in prompt.lower():
        st.session_state.planning_mode = True
        st.toast("📑 Modo Planeación Didáctica Activado")

    current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else SYSTEM_PROMPT_BASE

    context_text = ""
    if st.session_state.get("vectorstore"):
        context_text, _ = get_context_for_planning(prompt, st.session_state.vectorstore, st.session_state.loaded_files)

    full_prompt = current_prompt
    if context_text:
        full_prompt += f"\n\nContexto de documentos:\n{context_text}"

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
