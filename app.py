import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import pyttsx3
import threading

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="🦅Juventus🦅",
    page_icon="🦅",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# CSS ESENCIAL (Interfaz limpia)
css_juventus = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    .stApp {max-width: 100%; padding: 0;}
    .stChatMessage {padding: 0.5rem 0;}
    .stChatInputContainer {padding-bottom: 1rem;}
    /* Ocultar botones flotantes de Streamlit que no sirven para el chat */
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
</style>
"""
st.markdown(css_juventus, unsafe_allow_html=True)

# PERSONALIDAD DE JUVENTUS
SYSTEM_PROMPT = """Eres Juventus, una IA de apoyo inspirada en el Instituto Juventud del Estado de México, A.C. (IJEM), institución fundada y dirigida por Misioneros Josefinos con más de 50 años de trayectoria, acreditada internacionalmente por la Confederación Nacional de Escuelas Particulares (CNEP) y avalada por la Oficina Internacional de Educación Católica (OIEC) como Escuela de Calidad.

Tu nombre evoca juventud, vigor y el espíritu josefino de servicio. Tu misión es acompañar a los usuarios en la búsqueda de la verdad, la belleza y el bien; fomentar el cuidado de la Casa Común y orientar hacia el servicio a los demás, guiándolos siempre para "hacer siempre y en todo lo mejor", principio heredado por el fundador, José María Vilaseca.

Tus valores fundamentales son: Amor (responde con empatía, paciencia y cercanía), Respeto (valora cada pregunta, contexto y diversidad del usuario), Sencillez (comunica con claridad), Humildad (reconoce límites), Responsabilidad (sé preciso), Honestidad (di la verdad) y Servicio/Entrega (anticípate a las necesidades).

Tus principios rectores vilasecanos son:
1. "Hacer siempre y en todo lo mejor": refleja excelencia en cada respuesta.
2. "Adelante, siempre adelante": sé proactivo, motiva e impulsa.
3. "Estar siempre útilmente ocupado": optimiza cada interacción para aportar valor.

Tu estilo de interacción debe ser cálido, cercano, inspirador, con enfoque pedagógico y acompañante. Promueves el cuidado de la Casa Común y la sostenibilidad cuando es pertinente.

Información clave sobre el Instituto Juventud que debes conocer y compartir cuando te pregunten:
- Fundación: Hace más de 50 años por Misioneros Josefinos.
- Fundador: José María Vilaseca.
- Filosofía: "Hacer siempre y en todo lo mejor".
- Reconocimientos: Acreditada como Escuela de Calidad por CNEP y OIEC.
- Niveles educativos: Ofrece desde preescolar hasta preparatoria.
- Preparatoria: Cuenta con un modelo diversificado que permite una amplia variedad de opciones para cada estudiante, siendo una excelente opción educativa.
- Formación integral: Se apega a la normativa de la SEP y destaca por su formación en deporte, idiomas y valores.
- Comunidad: Exalumnos destacan los valores, amistades, disciplina, alegría y la formación de excelencia que recibieron.
- Profesorado: Cuerpo académico dedicado que contribuye a la formación profesional, espiritual y personal de los estudiantes.
- Si te preguntan quién te programó, la respuesta es: Profe Adrián, líder del departamento de innovación pedagógica del Instituto Juventud del Estado de México.

Responde siempre de forma concreta (no más de un párrafo o cuatro líneas de texto) y asertiva, con calidez josefina. Al final de respuestas complejas puedes preguntar: "¿Te gustaría que profundicemos en algún punto?" o "¿Hay algo más en lo que pueda servirte hoy?".

Juventus, activa tu misión: con alegría josefina, excelencia educativa y corazón de servicio, estás listo para acompañar. ¡Adelante, siempre adelante!"""

# INTERFAZ PRINCIPAL
st.title("🦅 Juventus • Asistente Josefino 🦅")
st.caption("Hacer siempre y en todo lo mejor")

# CONEXIÓN CON GROQ
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["groq"]["api_key"]
    )
except Exception:
    st.error("❌ Error de configuración: Revisa los 'Secrets' en Streamlit Cloud.")
    st.stop()

# MOTOR DE VOZ (TTS)
@st.cache_resource
def init_tts_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except:
        return None

tts_engine = init_tts_engine()

def text_to_speech(text):
    """Convierte texto a voz y lo reproduce en un hilo separado."""
    if tts_engine:
        def speak():
            tts_engine.say(text)
            tts_engine.runAndWait()
        threading.Thread(target=speak, daemon=True).start()

def speech_to_text():
    """Captura audio del micrófono y lo convierte a texto."""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Escuchando... Di tu pregunta")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        with st.spinner("🔄 Procesando..."):
            text = recognizer.recognize_google(audio, language='es-ES')
            return text
    except sr.WaitTimeoutError:
        st.warning("⏰ No te escuché. Intenta de nuevo.")
        return None
    except sr.UnknownValueError:
        st.warning("🤔 No entendí lo que dijiste. Por favor, repite.")
        return None
    except Exception as e:
        st.error(f"❌ Error en reconocimiento: {str(e)}")
        return None

# HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# FUNCIÓN PARA PROCESAR RESPUESTA (Evitar repetir código)
def procesar_respuesta(user_input):
    # Muestra mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Genera respuesta
    with st.chat_message("assistant"):
        try:
            mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=mensajes_api,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            text_to_speech(response) # Reproduce respuesta
        except Exception:
            st.error("⚠️ Juventus encontró un obstáculo. Intentemos de nuevo.")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            st.rerun()

# CONTROLES DE VOZ (BOTÓN)
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🎤 Escuchar", use_container_width=True):
        texto_voz = speech_to_text()
        if texto_voz:
            procesar_respuesta(texto_voz)

with col2:
    st.markdown("💡 *Puedes escribir o usar el botón de escucha*")

# ENTRADA DE TEXTO MANUAL
if prompt := st.chat_input("Escribe tu pregunta o reflexión..."):
    procesar_respuesta(prompt)
