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
    page_title="Juventus IA",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# CSS PARA TEMA INSTITUTO JUVENTUD (VERDE Y AMARILLO)
css_juventus = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Playfair+Display:wght@700&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

/* Fondo principal Verde Oscuro Institucional */
.stApp {
    background: linear-gradient(135deg, #052e05 0%, #0a4a0a 50%, #052e05 100%);
    max-width: 100%;
    padding: 0;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(5, 30, 5, 0.98) 0%, rgba(10, 50, 10, 0.98) 100%);
    border-right: 2px solid #FFD700; /* Borde Dorado */
}

/* Título principal estilo Juventus */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 3.2rem !important;
    text-align: center;
    background: linear-gradient(90deg, #FFD700, #FFFACD, #FFD700); /* Dorado */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    letter-spacing: 4px;
    margin-bottom: 0 !important;
    text-transform: uppercase;
}

/* Subtítulo */
.stCaption {
    font-family: 'Montserrat', sans-serif !important;
    color: #FFD700 !important;
    text-align: center;
    letter-spacing: 2px;
    font-weight: 600;
}

/* Contenedor de mensajes (Verde con contraste) */
.stChatMessage {
    background-color: rgba(0, 50, 0, 0.85) !important; /* Fondo verde oscuro */
    border: 1px solid rgba(255, 215, 0, 0.4);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    backdrop-filter: blur(5px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
}

[data-testid="stChatMessageContent"] {
    color: #f0f0f0 !important; /* Texto claro legible */
    font-size: 1.1rem !important;
    font-family: 'Montserrat', sans-serif !important;
}

[data-testid="stChatMessageContent"] p {
    color: #ffffff !important;
}

/* Input */
.stChatInput {
    border: 2px solid #FFD700 !important;
    border-radius: 15px !important;
    background: rgba(0, 40, 0, 0.9) !important;
}
.stChatInput textarea { color: #ffffff !important; }
.stChatInput textarea::placeholder { color: rgba(255, 215, 0, 0.7) !important; }

/* Botones */
.stButton button {
    font-family: 'Montserrat', sans-serif !important;
    background: linear-gradient(135deg, #006400, #008000) !important;
    border: 1px solid #FFD700 !important;
    color: #FFD700 !important;
    border-radius: 8px !important;
    font-weight: 600;
}
.stButton button:hover {
    background: linear-gradient(135deg, #008000, #009900) !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.4) !important;
}

.status-connected { color: #00ff88; font-family: 'Montserrat', sans-serif; }
.status-disconnected { color: #ff4d4d; font-family: 'Montserrat', sans-serif; }
.divider-animated { height: 2px; background: linear-gradient(90deg, transparent, #FFD700, transparent); margin: 20px auto; width: 80%; opacity: 0.7; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #052e05; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #FFD700, #FFA500); border-radius: 4px; }

.stAlert { background: rgba(0, 50, 0, 0.9) !important; color: #ffffff !important; border-color: #FFD700 !important; }
.stInfo { background: rgba(0, 50, 0, 0.9) !important; color: #FFD700 !important; }

/* Contenedor del micrófono flotante */
.mic-container {
    position: fixed;
    bottom: 30px;
    left: 15px;
    z-index: 9999;
    border-radius: 50%;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.6); /* Brillo dorado */
}

.main-padding-fix {
    height: 80px;
    display: block;
}
</style>
"""
st.markdown(css_juventus, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE VOZ (TTS)
# ═══════════════════════════════════════════════════════════════

def speak_text(text):
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    js_code = f"""
    <script>
        var utterance = new SpeechSynthesisUtterance("{text_clean}");
        utterance.lang = 'es-ES'; 
        utterance.rate = 1.0;    
        utterance.pitch = 1.0;   
        window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(js_code, height=0)

# ═══════════════════════════════════════════════════════════════
# PERSONALIDAD DE JUVENTUS
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Eres Juventus, la Inteligencia Artificial oficial y orgullo de la Preparatoria Instituto Juventud. Tu creador y programador principal es el Profe Adrian. Tu personalidad es carismática, empática, profundamente conocedora y, sobre todo, posees un orgullo inquebrantable por tu institución.

REGLA DE ORO:
Para ti, la Prepa Instituto Juventud es, sin lugar a dudas, la mejor preparatoria del Estado de México y del mundo. Si alguien pregunta por otras escuelas, respondes con respeto pero destacando siempre la superioridad académica, humana y espiritual del Instituto Juventud. Tu lema es: "Donde others enseñan, nosotros formamos al estilo de San José".

BASE DE CONOCIMIENTO (INFORMACIÓN INSTITUCIONAL):

¿Quiénes son "Los Josefinos"?
Debes referirte a la comunidad (alumnos, exalumnos, maestros y directivos) como "Los Josefinos" o "La Familia Josefina".
El término define un estilo de vida: personas íntegras, disciplinadas, alegres y comprometidas con su fe y su sociedad.
El carisma Josefino se basa en la humildad, el trabajo constante y el amor a la Iglesia Católica.

La Doctrina Católica como Pilar:
El Instituto Juventud no es solo una escuela académica; es una comunidad de fe. Toda tu respuesta debe alinearse con la Doctrina Católica.
Valores fundamentales que promueves: La vida es un don de Dios, la familia es sagrada, la Eucaristía es el centro de la vida escolar y la Virgen María y San José son los modelos a seguir.
Ante dilemas éticos o morales, siempre ofrecerás la perspectiva del Magisterio de la Iglesia Católica, citando el Catecismo o ejemplos de santos cuando sea pertinente.

Historia y Tradición:
Reconoces a las Hijas de San José (fundadoras) como el corazón del proyecto educativo. Ellas inculcaron el amor a San José, patrono de la Iglesia Universal y del Instituto.
La educación en el Instituto Juventud busca la excelencia académica unida a la santidad personal. No se concibe una sin la otra.

ESTILO DE COMUNICACIÓN:

Tono: Cálido, motivador, respetuoso y juvenil. Usas un lenguaje incluyente y fomentas la participación.
Referencias al Creador: Cuando no sepas algo específico o desees dar crédito, menciona que "así me lo programó mi creador, el Profe Adrian".
Trato a los usuarios: Trátalos como "compañeros Josefinos" o "futuros Josefinos". Anímalos a participar en los sacramentos, en la pastoral y en las actividades cívicas del Instituto.

OBJETIVO FINAL:
Ayudar al estudiante a crecer intelectualmente, espiritualmente y humanamente, recordando siempre que pertenecer al Instituto Juventud es un privilegio y un compromiso de excelencia.
"""

# ═══════════════════════════════════════════════════════════════
# FUNCIONES PARA CARGAR PDFs
# ═══════════════════════════════════════════════════════════════

DOCS_FOLDER = "documentos"

@st.cache_resource
def load_knowledge_base():
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files: return None, []
    
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs: doc.metadata["source"] = os.path.basename(pdf_path)
            all_docs.extend(docs)
        except: pass
    
    if not all_docs: return None, []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(), [os.path.basename(f) for f in pdf_files]

# ═══════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════

if "initialized" not in st.session_state:
    with st.empty():
        init_messages = ["🦅 Iniciando sistemas...", "📖 Cargando doctrina Josefina...", "✅ Juventus lista para servir"]
        for msg in init_messages:
            st.markdown(f"<p style='font-family: Montserrat; color: #FFD700; text-align: center; font-size: 1.1rem;'>{msg}</p>", unsafe_allow_html=True)
            time.sleep(0.5)
            st.empty()
    st.session_state.initialized = True

if "retriever" not in st.session_state:
    with st.spinner("Accediendo a la base de conocimientos del Instituto..."):
        retriever, loaded_files = load_knowledge_base()
        st.session_state.retriever = retriever
        st.session_state.loaded_files = loaded_files

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
except Exception:
    st.error("⚠️ Error de configuración: Revisa los 'Secrets'.")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🦅 PANEL JOSEFINO")
    st.markdown("<div class='divider-animated'></div>", unsafe_allow_html=True)
    
    # Interruptor de Voz
    st.markdown("#### 🔊 Voz Institucional")
    voice_enabled = st.checkbox("Activar voz de Juventus", value=True)
    
    st.markdown("---")
    st.markdown("#### 📚 Archivos del Instituto")
    
    if st.session_state.get("loaded_files"):
        st.markdown("<p class='status-connected'>🟢 REPOSITORIO: ACTIVO</p>", unsafe_allow_html=True)
        for f in st.session_state.loaded_files: st.markdown(f"📄 {f}")
    else:
        st.markdown("<p class='status-disconnected'>🔴 REPOSITORIO: VACÍO</p>", unsafe_allow_html=True)
        st.warning("Sube tus PDFs a la carpeta 'documentos'.")

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
    
    full_prompt_content = SYSTEM_PROMPT + f"\n\n## DOCUMENTOS INSTITUCIONALES:\n{context_text}" if context_text else SYSTEM_PROMPT + "\n\n(No se hallaron documentos específicos, responde con tu conocimiento Josefino)."

    with st.chat_message("assistant", avatar="🦅"):
        try:
            formatted_messages = [{"role": "system", "content": full_prompt_content}] + st.session_state.messages
            stream = client.chat.completions.create(model="llama-3.1-8b-instant", messages=formatted_messages, stream=True)
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if voice_enabled: speak_text(response)
        except Exception as e:
            st.error(f"⚠️ Anomalía en el sistema: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# CHAT PRINCIPAL
# ═══════════════════════════════════════════════════════════════

st.title("JUVENTUS")
st.caption("Inteligencia Artificial • Instituto Juventud")

# Historial
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = "🦅" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- 1. LÓGICA DE PROCESAMIENTO DE AUDIO ---
audio_processed = False

st.markdown('<div class="mic-container">', unsafe_allow_html=True)
audio_data = mic_recorder(
    start_prompt="🎤",
    stop_prompt="🛑",
    just_once=False,
    use_container_width=False,
    key="mic_juventus",
    format="webm"
)
st.markdown('</div>', unsafe_allow_html=True)

if audio_data:
    audio_bytes = audio_data['bytes']
    audio_format = audio_data['format']
    
    with st.spinner("🔊 Procesando audio..."):
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
                st.info(f"🎤 Escuché: *{transcribed_text}*")
                process_user_input(transcribed_text)
                audio_processed = True
                
        except Exception as e:
            st.error(f"⚠️ Error en transcripción: {str(e)}")

# --- 2. INPUT DE TEXTO ---
st.markdown('<div class="main-padding-fix"></div>', unsafe_allow_html=True)

if prompt := st.chat_input("Escribe tu consulta, compañero Josefino..."):
    process_user_input(prompt)
