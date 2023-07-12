###############################
# Imports for App
###############################
import json
import streamlit as st
import openai
import torch
import os
import requests
from streamlit_lottie import st_lottie # Lottie Animations
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from diffusers import StableDiffusionPipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

##############################
# API Keys
##############################

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

###############################
# Function For Lottie Animations
###############################
def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

###############################
# Function to Generate Images
###############################
def generate_images(text):
    response = openai.Image.create(prompt= text, n=1, size="512x512")
    image_url = response['data'][0]['url']
    return image_url


###############################
# Functions Story Generation
###############################
def img2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(
        url)[0]["generated_text"]

    print(text)
    return text

# LLM chain
def generate_story(scenario):
    template = """
    You are a story Dungeon Master for a DnD game:
    You can generate a short story based on a simple narrative, the story should be more than 20 words

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=ChatOpenAI(
        model_name="gpt-3.5-turbo-0613", temperature=1), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

# Text to Speech
def text2speech(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
        }

    }

    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX?optimize_streaming_latency=0", json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open("audio.mp3", "wb") as file:
            file.write(response.content)


###############################
# Function For PDF Chat
###############################

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Break up PDF text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Save to Vectorstore
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    chat_placeholder = st.container()
    st.session_state.chat_history = response['chat_history']

    with chat_placeholder:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

############################
# Main Function
############################

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    #Lottie Animation loading
    lottie_flag = load_lottiefile("static/dragon-flag.json")
    lottie_dice = load_lottiefile("static/d20.json")
    lottie_monster = load_lottiefile("static/monster.json")
    lottie_wizard = load_lottiefile("static/wizard.json")

    # Streamlit setup
    st.set_page_config(page_title="Dungeon AI", page_icon="dragon")
    st.write(css, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Chat", "Image Generator", "Audio Generator"])
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # User Question
    with tab1:
        st.header("Dungeon Chat:")
        question_placeholder = st.container()

        with question_placeholder:
            st_lottie(lottie_dice, height=150, speed=.85)
            user_question = st.text_input("A chatbot for all your D&D needs:", placeholder="What can I help you find today Dungeon Master?...")

            if user_question:
                handle_userinput(user_question)

    # Image Generator
    with tab2:
        st.header("Monster Generator:")
        st_lottie(lottie_monster, height=175, speed=.85)
        input_prompt = st.text_input("Name your monster:", placeholder="What monster would you like to generate?...")
        if input_prompt is not None:
            if st.button("Generate Monster Image"):
                image_url = generate_images(input_prompt)
                st.image(image_url, caption="Monster Image")

    # Audio Generator
    with tab3:
        st.header("Story Generator")
        st_lottie(lottie_wizard, height=200, speed=.85)
        uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            print(uploaded_file)
            bytes_data = uploaded_file.getvalue()
            with open(uploaded_file.name, "wb") as file:
                file.write(bytes_data)
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            scenario = img2text(uploaded_file.name)
            story = generate_story(scenario)
            text2speech(story)

            with st.expander("scenario"):
                st.write(scenario)
            with st.expander("story"):
                st.write(story)

            st.audio("audio.mp3", format="audio/mp3")

    # Side bar
    with st.sidebar:
        st_lottie(lottie_flag, height=250, speed=.85)
        st.subheader("Load your Tomes")
        pdf_docs = st.file_uploader(
            "Upload your magical tomes here and see the magic of Dungeon AI!", accept_multiple_files=True)
        if st.button("Roll a D20 to Proceed"):
            with st.spinner("Processing..."):
                # Get the PDFs Text - Returns the text is a single string
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Create vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == "__main__":
    main()
