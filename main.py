#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import TextLoader
#from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# Groq API-Schlüssel aus secrets.toml laden
groq_api_key = st.secrets["GROQ_API_KEY"]

# Groq API-Schlüssel setzen
os.environ["GROQ_API_KEY"] = groq_api_key

# PDF-Verzeichnis
pdf_folder_path = "./IFRS_TEXT"

# Dokumente laden und verarbeiten
loader = DirectoryLoader(pdf_folder_path, glob="*.pdf")
documents = loader.load()

# Texte aufteilen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Embeddings erstellen
embeddings = HuggingFaceEmbeddings()

# Vektorstore erstellen
#vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore = Chroma.from_documents(texts, embeddings)

# Groq-Modell initialisieren
#llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature = 0.7)
llm = ChatGroq(model="llama-3.1-70b-versatile",temperature=0)

# Konversationskette erstellen
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    return_source_documents=True
)

# Streamlit UI



st.set_page_config(
    page_title="Chatbot",
    layout="centered"
)

# Setze das Theme
# CSS für dunkles Theme
dark_theme = """
<style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4CAF50;
        border: none;
    }
    .stTextInput>div>div>input {
        color: #FFFFFF;
        background-color: #333333;
    }
    .st
    .stSelectbox>div>div>select {
        color: #FFFFFF;
        background-color: #333333;
    }
</style>
"""

# Anwenden des dunklen Themes
#st.markdown(dark_theme, unsafe_allow_html=True)




st.title("KI-Assistent für Fragen zu IFRS 9")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Was möchtest Du wissen?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        result = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})

        full_response = result["answer"]
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
