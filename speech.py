import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3
import webbrowser
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the default voice to id=1
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Dictionary to map keywords to URLs and system commands
keyword_to_action = {
    "open youtube": lambda: webbrowser.open("https://www.youtube.com"),
    "open google": lambda: webbrowser.open("https://www.google.com"),
    "open facebook": lambda: webbrowser.open("https://www.facebook.com"),
    "open twitter": lambda: webbrowser.open("https://www.twitter.com"),
    "open camera": lambda: os.system("start microsoft.windows.camera:" if os.name == 'nt' else "open /Applications/Photo/ Booth.app"),
    "open photos": lambda: os.system("start ms-photos:" if os.name == 'nt' else "open /System/Applications/Photos.app"),
    "open cmd": lambda: os.system("start cmd" if os.name == 'nt' else "open -a Terminal"),
    "open notepad": lambda: os.system("start notepad" if os.name == 'nt' else "open -a TextEdit"),
    "open spotify": lambda: os.system("start spotify:" if os.name == 'nt' else "open -a Spotify"),
    "open calculator": lambda: os.system("start calc" if os.name == 'nt' else "open -a Calculator"),
    "open steam": lambda: os.system("start steam:" if os.name == 'nt' else "open -a Steam"),
    "open epic games": lambda: os.system("start com.epicgames.launcher:" if os.name == 'nt' else "open -a Epic/ Games/ Launcher"),
    "open discord": lambda: os.system("start discord:" if os.name == 'nt' else "open -a Discord"),
    "open file explorer": lambda: os.system("explorer" if os.name == 'nt' else "open .")
}

def listen():
    recognizer = sr.Recognizer()
    print("Listening...")
    duration = 7  # Record for 10 seconds
    sample_rate = 16000
    try:
        # Record audio with sounddevice
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished

        # Convert the numpy array to bytes
        audio_data = np.squeeze(audio)
        audio_data_bytes = audio_data.tobytes()

        # Create an AudioData object from the bytes
        audio_data_obj = sr.AudioData(audio_data_bytes, sample_rate, 2)

        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio_data_obj)
        print(f"User: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def handel_conversation():
    context = ""
    print("Welcome to the AI chatbot! Say 'exit' to quit.")
    while True:
        user_input = listen()
        
        # If speech recognition fails, fallback to text input
        if user_input is None:
            user_input = input("You: ")

        if user_input.lower() == "exit":
            break
        elif user_input.lower() in keyword_to_action:
            keyword_to_action[user_input.lower()]()
            print(f"Executing {user_input.lower()}...")
        else:
            # Generate the response using the model
            result = chain.invoke({"context": context, "question": user_input})
            print("Nim-Llama: ", result)
            speak(result)
            context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handel_conversation()
