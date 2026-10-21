import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando el modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

texto = "Hola, bienvenidos a Inteligencia artificial generativa, vamos a generar un audio."

print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7,       # Expresividad o emoción
    cfg_weight=0.4,
    temperature=0.8,        # Controla la variabilidad
    repetition_penalty=1.2, # Penaliza repeticiones
    min_p=0.05,             # Filtra opciones poco probables
    top_p=1.0,              # Controla el muestreo
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio generado: audio.wav")
