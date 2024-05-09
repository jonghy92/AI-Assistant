import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings


import os
import time
import tempfile
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS

# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain


## session_state
def initialize_session_state():
    if "history_doc" not in st.session_state:
        st.session_state["history_doc"] = []
    if "generated_doc" not in st.session_state:
        st.session_state["generated_doc"] = ["Hello!, Ask me anything. I'm a Doc. Assistant"]
    if "past_doc" not in st.session_state:
        st.session_state["past_doc"] = ["Hey!"]


## Load Model - LLM
def load_model():
    local_model = "mistral"
    llm = ChatOllama(model=local_model)
    return llm


def create_conversational_chain(vector_store):
    llm = load_model()
    print("#### ==== Model Loaded ==== ####")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory,
        chain_type="stuff",
        # combine_docs_chain_kwargs={}
    )
    return chain



## conversation chat
def conversation_chat(query, chain, history):
    result = chain({"question" : query, "chat_history" : history})
    #print("### result ###")
    values  = result.values()
    val_list = list(values)
    output_txt = val_list[2]
    #print(output_txt)
    history.append((query, output_txt))
    #print("### history ###")
    #print(history)

    return output_txt


## display chat history
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents ...!", key=" ")
            submit_button = st.form_submit_button(label="Send")
        if submit_button and user_input:
            with st.spinner("Generating response ...!"):
                output = conversation_chat(user_input, chain, st.session_state["history_doc"])
                st.session_state['past_doc'].append(user_input)
                st.session_state["generated_doc"].append(output)
    if st.session_state["generated_doc"]:
        with reply_container:
            for i in range(len(st.session_state["generated_doc"])):
                message(st.session_state["past_doc"][i], is_user=True, key=str(i) + "_user", avatar_style="thumbs")
                message(st.session_state["generated_doc"][i], key=str(i), avatar_style="fun-emoji")



## Frontend - Streamlit
def main():
    #### streamlit main
    initialize_session_state()
    st.set_page_config(
        page_title="Multi AI Assistant App",
    )


    st.sidebar.success("Doc Assistant selected")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)

    st.title("Doc - Assistant 🌏")
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')



    #    chain = create_conversational_chain_normal()
    #    display_chat_history(chain)


    if uploaded_files:
        with st.spinner("Preparing AI based on your Docs...!"):
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name
            
            loader = None
            if file_extension == ".pdf":
                loader = PyMuPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
                st.sidebar.success("Successfully load Documents.")

            # text split --> create chunks
            text_splitter = CharacterTextSplitter(separator="\n",
                                                chunk_size=1000,
                                                chunk_overlap=100,
                                                length_function=len)
            text_chunks = text_splitter.split_documents(text)


            # create Embeddings
            embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                                model_kwargs={'device':'cpu'})


            # create a vector store
            vector_store = FAISS.from_documents(documents=text_chunks,
                                                embedding=embeddings,
                                                )
            
            # chain object
            chain = create_conversational_chain(vector_store)

            # display chat history
            display_chat_history(chain)

    else:
        st.markdown("##### Upload Your Documents to start the chat...!")


if __name__ == "__main__":
    main()