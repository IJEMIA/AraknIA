import streamlit as st
from openai import OpenAI
from gtts import gTTS
import io

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

# MOTOR DE VOZ WEB (gTTS)
def text_to_speech_web(text):
    """Convierte texto a audio MP3 para reproducir en el navegador."""
    try:
        tts = gTTS(text=text, lang='es', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        # Si falla la generación de audio, no detiene la app, solo no reproduce
        return None

# HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# FUNCIÓN PARA PROCESAR RESPUESTA
def procesar_respuesta(user_input):
    # Muestra mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Genera respuesta
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=mensajes_api,
                stream=True,
            )
            
            # Stream de texto (efecto de escritura)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # --- REPRODUCCIÓN DE VOZ ---
            if full_response:
                audio_bytes = text_to_speech_web(full_response)
                if audio_bytes:
                    # autoplay=True intenta reproducir automáticamente
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        except Exception as e:
            st.error(f"⚠️ Juventus encontró un obstáculo: {str(e)}")
            # Limpieza en caso de error
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

# ENTRADA DE TEXTO MANUAL
if prompt := st.chat_input("Escribe tu pregunta o reflexión..."):
    procesar_respuesta(prompt)
