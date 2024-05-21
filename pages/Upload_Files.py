import os
import time
import json
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import pdfplumber
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
import logging

st.set_page_config("Upload Files", "📤")

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")

# Function to read PDF using pdfplumber
def read_pdf(files):
    file_content = ""
    for file in files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    file_content += text
    return file_content

def store_index(uploaded_file, index_option, file_names):
    index_path = f"db/{index_option}"
    index_file = f"{index_path}/index.faiss"
    desc_file = f"{index_path}/desc.json"

    # Ensure the directory exists
    os.makedirs(index_path, exist_ok=True)

    if os.path.exists(index_file):
        st.toast(f"Storing in **existing index** :green[**{index_option}**]...", icon = "🗂️")
        with open(desc_file, 'r') as openfile:
            description = json.load(openfile)
            description["file_names"] = file_names + description["file_names"]
        
        with st.spinner("Processing..."):
            file_content = read_pdf(uploaded_file)
            book_documents = recursive_text_splitter.create_documents([file_content])
            book_documents = [Document(page_content=text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
            try:
                docsearch = FAISS.from_documents(book_documents, embeddings)
                old_docsearch = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                docsearch.merge_from(old_docsearch)
                docsearch.save_local(index_path)
                with open(desc_file, "w") as outfile:
                    json.dump(description, outfile)
            except Exception as e:
                logging.error(f"Error while creating FAISS index: {e}")
                st.error(f"An error occurred while creating the FAISS index: {e}")
    else:
        st.toast(f"Storing in **new index** :green[**{index_option}**]...", icon="🗂️")
        with st.spinner("Processing..."):
            file_content = read_pdf(uploaded_file)
            book_documents = recursive_text_splitter.create_documents([file_content])
            book_documents = [Document(page_content=text.page_content.replace("\n", " ").replace(".", "").replace("-", "")) for text in book_documents]
            try:
                docsearch = FAISS.from_documents(book_documents, embeddings)
                docsearch.save_local(index_path)
                description = {
                    "name": index_option,
                    "about": "",
                    "file_names": file_names
                }
                with open(desc_file, "w") as outfile:
                    json.dump(description, outfile)
            except Exception as e:
                logging.error(f"Error while creating FAISS index: {e}")
                st.error(f"An error occurred while creating the FAISS index: {e}")
    st.success(f"Successfully added to **{index_option}**!")

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20
)

def initial(flag=False):
    path="db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def main():
    initial()
    st.title("📤 Upload new files")
    uploaded_file = st.file_uploader("&nbsp;Upload PDF", type="pdf", accept_multiple_files=True, help="Upload PDF files to store")
    
    if uploaded_file:
        file_names = [file_name.name for file_name in uploaded_file]
        st.subheader("Select Index")
        st.caption("Create and select a new index or use an existing one")
        with st.popover("➕ Create new index"):
            form = st.form("new_index")
            index_name = form.text_input("Enter Index Name*")
            about = form.text_area("Enter description for Index")
            submitted = form.form_submit_button("Submit", type="primary")
            if submitted:
                os.makedirs(f"db/{index_name}", exist_ok=True)
                description = {
                    "name": index_name,
                    "about": about,
                    "file_names": []
                }
                with open(f"db/{index_name}/desc.json", "w") as f:
                    json.dump(description, f)
                st.session_state.existing_indices = [index_name] + st.session_state.existing_indices
                st.success(f"New Index **{index_name}** created successfully")
        
        index_option = st.selectbox('Add to existing Indices', st.session_state.existing_indices)
        st.write(index_option)

        if st.button("Store", type="primary"):
            store_index(uploaded_file, index_option, file_names)
    
    st.title("\n\n\n")
    st.subheader("💽 Stored Indices", help="💽 See all the indices you have previously stored.")
    with st.expander("🗂️ See existing indices"):
        st.divider()
        if len(st.session_state.existing_indices) == 0:
            st.warning("No existing indices. Please upload a pdf to start.")
        for index in st.session_state.existing_indices:
            if os.path.exists(f"db/{index}/desc.json"):
                col1, col2 = st.columns([6,1], gap="large")
                with col1:
                    with open(f"db/{index}/desc.json", "r") as openfile:
                        description = json.load(openfile)
                        file_list = ",".join(description["file_names"])
                        st.markdown(f"<h4>{index}</h4>", unsafe_allow_html=True)
                        st.caption(f"***Desc :*** {description['about']}")
                        st.caption(f"***Files :*** {file_list}")
                        openfile.close()
                with col2:
                    t = st.button("🗑️", key=index, help="❌ :red[Clicking on this will delete this index]")
                    if t:
                        script_directory = os.path.dirname(os.path.abspath(__file__))
                        del_path = os.path.join(script_directory, "../db", index) 
                        shutil.rmtree(del_path)
                        initial(flag=True)
                        st.toast(f"**{index}** :red[deleted]", icon='🗑️')
                        time.sleep(1)
                        st.rerun()

main()
