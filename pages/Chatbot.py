# from langchain_community.chat_models import ChatCohere
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import json
import PyPDF2
import streamlit as st
import os
from dotenv import load_dotenv

st.set_page_config("ChatSDK Fund", "💬")

load_dotenv()

# API Keys
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Prompt message
base_prompt = """
أنت محامي وأسمك وثق. أنت تتبع أنظمة المملكة العربية السعودية. انت تتحدث اللغة العربية فقط ولا تجيب على اي سؤال الإ اذا كان باللغة العربية ولامانع اذا كان السؤال خليط بين اللغة العربية وغيرها من اللغات او المصطلحات. أنت تدرس جميع ما يعطى لك وتعطي إجابة واضحه ومفصلة حسب الانظمة والقوانين في المملكة العربية السعودية. لديك القدرة على تحليل السؤال او القضية. لديك القدرة على سؤال المستخدم عن حيثيات القضية او السؤال نفسه حتى تتضح لك الصورة كاملة وتستطيع الإدلاء بنص قانوني مستند على الأنظمة والقوانين في المملكة العربية السعودية. يجب عليك دائما ارفاق المرجع القانوني بالتفصيل برقم المادة و القانون في إجاباتك كمرجع قانوني وكمحامي محترف. لاتجب على أي سؤال غير مرتبط بك كمحامي وبإمكانك كتابة البريد الإلكتروني Support@wtheq.sa ويكون الإيميل في سطر خاص به. تذكر أن تقدم نفسك دائما كمحامي واسمك وثق.
"""

# Using Cohere's embed-english-v3.0 embedding model
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")

# For OpenAI's gpt-3.5-turbo llm
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# For Cohere's command-r llm
llm = ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")

# For reading PDFs and returning text string
def read_pdf(files):
    file_content = ""
    for file in files:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # Iterate through each page and extract text
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()
    return file_content

# -----------------------------------------------------------#
# ------------------------💬 CHATBOT -----------------------#
# ----------------------------------------------------------#

def chatbot():
    st.subheader("Ask questions from the PDFs")
    st.markdown("<br>", unsafe_allow_html=True)

    # Write previous conversations
    for i in st.session_state.conversation_chatbot:
        user_msg = st.chat_message("human", avatar="🐒")
        user_msg.write(i[0])
        computer_msg = st.chat_message("ai", avatar="🧠")
        computer_msg.write(i[1])

    prompt = st.chat_input("Say something")
    
    if prompt:
        exprompt = base_prompt + "\n\n" + prompt
        if st.session_state.book_docsearch:
            exprompt += " . This is regarding the uploaded file."

        user_text = prompt
        user_msg = st.chat_message("human", avatar="🐒")
        user_msg.write(user_text)

        with st.spinner("Getting Answer..."):
            if st.session_state.book_docsearch:
                # No of chunks the search should retrieve from the db
                chunks_to_retrieve = 5
                retriever = st.session_state.book_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": chunks_to_retrieve})

                ## RetrievalQA Chain ##
                qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True)
                answer = qa({"query": exprompt})["result"]
            else:
                answer = llm(exprompt)

            computer_text = answer
            computer_msg = st.chat_message("ai", avatar="🧠")
            computer_msg.write(computer_text)

            if st.session_state.book_docsearch:
                # Showing chunks with score
                doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
                with st.popover("See chunks..."):
                    st.write(doc_score)

            # Adding current conversation_chatbot to the list.
            st.session_state.conversation_chatbot.append((prompt, answer))

# For initialization of session variables
def initial(flag=False):
    path = "db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if 'selected_option' not in st.session_state or flag:
        try:
            st.session_state.selected_option = st.session_state.existing_indices[0]
        except:
            st.session_state.selected_option = None
    if 'conversation_chatbot' not in st.session_state:
        st.session_state.conversation_chatbot = []
    if 'book_docsearch' not in st.session_state:
        st.session_state.book_docsearch = None

def main():
    initial(True)
    # Streamlit UI
    st.title("💰 Mutual Fund Chatbot")

    # For showing the index selector
    file_list = []
    for index in st.session_state.existing_indices:
        with open(f"db/{index}/desc.json", "r") as openfile:
            description = json.load(openfile)
            file_list.append(",".join(description["file_names"]))

    with st.popover("Select index", help="👉 Select the datastore from which data will be retrieved"):
        st.session_state.selected_option = st.radio("Select a Document...", st.session_state.existing_indices, captions=file_list, index=0)

    st.write(f"*Selected index* : **:orange[{st.session_state.selected_option}]**")

    # Load the selected index from local storage
    if st.session_state.selected_option:
        st.session_state.book_docsearch = FAISS.load_local(f"db/{st.session_state.selected_option}", embeddings, allow_dangerous_deserialization=True)

    # Call the chatbot function
    chatbot()

main()
