from gtts import gTTS
from io import BytesIO
from IPython.display import Audio
#don't forget to add gTTS to requirement.txt

def speak(str):
    audio = gTTS(str, lang='en', tld='com.au')#,slow=True)
    filename = "voice.mp3"
    audio.save(filename)
    return filename

words = speak(str(prediction[0]))

Audio(words)
