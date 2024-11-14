import os
import time
from typing import Any
import torch
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
from gtts import gTTS
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def progress_bar(amount_of_time: int) -> None:
    """
    Displays a progress bar while processing.
    """
    progress_text = "Generating content, please wait..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(amount_of_time):
        time.sleep(0.03)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

def generate_text_from_image(uploaded_file) -> str:
    """
    Uses the BLIP model to generate a caption from an uploaded image.
    """
    image = Image.open(uploaded_file)
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text = image_to_text(image)[0]["generated_text"]
    return generated_text

def generate_story_from_text(scenario: str) -> str:
    """
    Uses LangChain with GPT to generate a short story based on the provided scenario.
    """
    prompt_template = """
    You are a creative storyteller. Generate a short story (max 50 words) based on the scenario below:

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    generated_story = story_llm.predict(scenario=scenario)
    return generated_story

def save_and_play_audio(text: str):
    """
    Converts text to audio using gTTS and plays it.
    """
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    st.audio(audio_bytes, format='audio/mp3')
    return audio_bytes

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="🖼️ Image-to-Story Converter", page_icon="📖", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Image-to-Story Converter</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Turn your images into captivating short stories!</p>", unsafe_allow_html=True)

    # Sidebar for uploading image
    with st.sidebar:
        st.write("---")
        st.markdown("AI App created by **Muhammad Umer**")
        st.header("Upload Your Image")
        uploaded_file = st.file_uploader("Please upload a JPG image", type=["jpg", "jpeg", "png"])
        st.write("")

    # Main content area
    st.header("Convert Image to Story")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Display progress bar
        progress_bar(100)

        # Generate scenario and story
        scenario = generate_text_from_image(uploaded_file)
        story = generate_story_from_text(scenario)
        
        # Display generated text and story
        with st.expander("Generated Image Scenario", expanded=True):
            st.write(scenario)
        with st.expander("Generated Short Story", expanded=True):
            st.write(story)

        # Convert story to audio and play it
        st.subheader("🎧 Listen to the Generated Story")
        audio_file = save_and_play_audio(story)

        # Add download buttons
        st.download_button(label="Download Story as Text", data=story, file_name="story.txt", mime="text/plain")
        st.download_button(label="Download Story Audio", data=audio_file.getvalue(), file_name="story_audio.mp3", mime="audio/mp3")
    else:
        st.info("👈 Please upload an image to get started.")

if __name__ == "__main__":
    main()
