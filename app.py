import streamlit as st
from openai import OpenAI
import time
import os
import glob
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io
import zipfile  # NUEVO: Para leer zips
import tempfile  # NUEVO: Para crear carpetas temporales

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

    /* BOTONES NORMALES Y MICRÓFONO EN SIDEBAR */
    .stButton button, .st-key-mic_btn button {
        background: linear-gradient(to right, #facc15, #fbbf24) !important;
        color: #022c22 !important;
        font-weight: 600;
        border-radius: 50px;
        border: none;
        padding: 0.8rem 1.5rem;
        width: 100%;
        transition: transform 0.2s;
    }

    .stButton button:hover, .st-key-mic_btn button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(250, 204, 21, 0.4);
    }

    /* TÍTULOS DEL SIDEBAR */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #facc15 !important;
        font-family: 'Montserrat', sans-serif;
        border-bottom: 2px solid rgba(250, 204, 21, 0.3);
        padding-bottom: 10px;
        margin-bottom: 15px;
    }

    /* ALERTAS Y MENSAJES DE ESTADO */
    .stAlert, .stSuccess, .stInfo, .stWarning {
        background-color: rgba(5, 46, 22, 0.6) !important;
        border-left: 5px solid #facc15 !important;
        color: white !important;
    }

    /* SCROLLBAR PERSONALIZADA */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #022c22; }
    ::-webkit-scrollbar-thumb { background: #facc15; border-radius: 10px; }

    /* ICONOS DE REPRODUCTOR DE AUDIO */
    .stAudio { display: none; }
</style>
"""
st.markdown(css_juventud, unsafe_allow_html=True)

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
            # Asignamos el nombre del archivo a los metadatos
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
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════

if "initialized" not in st.session_state:
    with st.empty():
        init_messages = ["🦅 Desplegando alas...", "💛 Sincronizando valores josefinos...", "✅ Juventud 2.0 lista, Profe Adrián"]
        for msg in init_messages:
            st.markdown(f"<p style='font-family: Montserrat; color: #facc15; text-align: center; font-size: 1.2rem;'>{msg}</p>", unsafe_allow_html=True)
            time.sleep(0.5)
            st.empty()
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

if "messages" not in st.session_state: st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR (Panel de Control Josefino)
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    # Título del Sidebar
    st.markdown("<h2 style='text-align: center; border:none;'>🦅 Panel Josefino</h2>", unsafe_allow_html=True)
    
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
    
    # --- CARGADOR DE ZIPS (NUEVA FUNCIONALIDAD) ---
    st.markdown("#### 📦 Cargar Archivos ZIP")
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs", type="zip", key="zip_uploader")
    
    if uploaded_zip:
        # Verificamos si ya procesamos este archivo para no repetir en cada rerun
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            st.session_state.processed_zip_name = uploaded_zip.name
            
            with st.spinner(f"Procesando {uploaded_zip.name}..."):
                # Creamos una carpeta temporal
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # Guardar el zip temporalmente
                        temp_zip_path = os.path.join(temp_dir, "temp.zip")
                        with open(temp_zip_path, "wb") as f:
                            f.write(uploaded_zip.getbuffer())
                        
                        # Extraer
                        with zipfile.ZipFile(temp_zip_path, 'r') as z:
                            z.extractall(temp_dir)
                        
                        # Buscar PDFs extraídos
                        extracted_pdfs = glob.glob(os.path.join(temp_dir, "**", "*.pdf"), recursive=True)
                        
                        if extracted_pdfs:
                            # Crear nuevo retriever
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
    
    # Principios
    st.markdown("#### 📜 Principios Rectores")
    st.info("✨ Hacer siempre y en todo lo mejor.")
    st.success("🚀 Adelante, siempre adelante.")
    st.warning("🛠️ Estar siempre útilmente ocupados.")

    # Créditos
    st.markdown("<br><p style='text-align:center; font-size:0.8rem; color:#555;'>Diseñado por el Profe Adrián</p>", unsafe_allow_html=True)

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
# CHAT PRINCIPAL (Interfaz Limpia)
# ═══════════════════════════════════════════════════════════════

# Encabezado Principal
st.markdown("<div class='main-header'><h1>JUVENTUD 2.0</h1><div class='subtitle'>Tu Guía Josefina</div></div>", unsafe_allow_html=True)

# Procesamiento de Audio (Si se usó el del sidebar)
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

# Input de Texto (Siempre al fondo)
if prompt := st.chat_input("Escribe tu mensaje, joven josefino..."):
    process_user_input(prompt)

