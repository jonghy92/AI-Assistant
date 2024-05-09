import os
import time
import streamlit as st
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama


def llm_chain():
    ## Create Template for LLM use
    template = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
    If you don't know the answer to a question, please don't share false information.
    If you get questions in other language from human, reply back using the same language.

    {chat_history}

    Human: {question}
    Assistant:
    """
    prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"])

    ## LLM from Ollama - Chain
    local_model = "gemma"
    llm = ChatOllama(model=local_model)
    memory = ConversationBufferMemory(memory_key="chat_history", k=3) # k를 지정하면 최근 k개의 대화만 기억하고 이전 대화는 삭제함.
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain



def conversation_chat(query):
    chain = llm_chain()
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["text"]))
    return result["text"]


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything ~ \n If you need 'Docs. Assistant' please select on the sidebar."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi ~ "]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask any question ...!", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner("Generating response ...!"):
                output = conversation_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")





## Frontend - Streamlit
def main():
    st.set_page_config(
        page_title="Multi AI Assistant App",
    )

    st.title("AI - Assistant 🌎")
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.sidebar.success("Select Assistant above.")

    initialize_session_state()
    initialize_session_state()
    display_chat_history()

if __name__ == "__main__":
    main()