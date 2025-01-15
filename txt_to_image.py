import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

#Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

# Initialize Gemini Pro LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Function to query Gemini Pro for text generation
def generate_text(context):
    prompt_template = PromptTemplate.from_template("Write a descriptive script for a background image based on this context: {context}")
    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = chain.run(context=context)
    return response

# Function to generate an image using the provided prompt
def generate_image(description):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
    payload = {"inputs": description}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Error generating image: {response.status_code}")
        return None

# Streamlit UI
st.title("Context to Image Generator with Gemini Pro")

# Step 1: Input context
context = st.text_area("Provide a context for the background image:")

if st.button("Generate Text"):
    if context:
        with st.spinner("Generating descriptive text..."):
            generated_text = generate_text(context)
            st.text_area("Generated Text (Modify if needed):", value=generated_text, key="editable_text")
    else:
        st.warning("Please provide a context.")

# Step 2: text for image generation
modified_text = st.text_area("Final Text for Image Generation (Edit if necessary):", key="final_text")

if st.button("Generate Image"):
    if modified_text:
        with st.spinner("Generating image..."):
            image_data = generate_image(modified_text)
            if image_data:
                st.image(image_data, caption="Generated Image", use_container_width=True)
    else:
        st.warning("Please provide or confirm the final text for image generation.")
