import streamlit as st
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tempfile
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.base import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

def get_document_loader(file_path, file_type):
    if file_type == "text/plain":
        return TextLoader(file_path)
    elif file_type == "text/csv":
        return CSVLoader(file_path)
    elif file_type == "application/pdf":
        return PyPDFLoader(file_path)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return Docx2txtLoader(file_path)
    else:
        return None

def summarize_chunks(chunks, llm, parser):
    chunk_summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        chunk_prompt = ChatPromptTemplate.from_template(
            "You are a highly skilled AI model tasked with summarizing text."
            "Please summarize the following chunk of text in a concise manner,"
            "highlighting the most critical information. Do not omit any key details:\n\n"
            "{document}"
        )
        chunk_chain = chunk_prompt | llm | parser
        chunk_summary = chunk_chain.invoke({"document": chunk})
        chunk_summaries.append(chunk_summary)
        progress_bar.progress((i + 1) / len(chunks))
    return chunk_summaries

def generate_final_summary(summaries, llm, parser):
    combined_summaries = "\n".join(summaries)
    final_prompt = ChatPromptTemplate.from_template(
        "You are an expert at summarizer tasked with creating a final summary from summarized chunks."
        "Combine the key points from the provided summaries into a cohesive and comprehensive summary."
        "The final summary should be concise but detailed enough to capture the main ideas and in a simple english version:\n\n"
        "{document}"
    )
    final_chain = final_prompt | llm | parser
    final_summary = final_chain.invoke({"document": combined_summaries})
    return final_summary

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(model="llama3-8b-8192")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**You:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")

def main():
    st.title("Summarizer and Chat App")
    st.divider()
    st.markdown("## Start Summarizing and Chatting with your documents here.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    uploaded_file = st.file_uploader("Upload a file (Text, PDF, Docx, or CSV)", type=["pdf", "txt", "docx", "csv"])

    if "chunks" not in st.session_state:
        st.session_state.chunks = None

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_file_path = tmp_file.name

                loader = get_document_loader(temp_file_path, uploaded_file.type)

                if loader is None:
                    st.error("Unsupported file type!")
                    st.stop()

                doc = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                st.session_state.chunks = text_splitter.split_documents(doc)
                st.success("File loaded and processed successfully!")

                with st.spinner("Creating vector store..."):
                    vectorstore = get_vectorstore(st.session_state.chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)


            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()
            finally:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)


    if st.button("🧠 Summarize", disabled=st.session_state.chunks is None):
        llm = ChatGroq(model="llama3-8b-8192")
        parser = StrOutputParser()
        
        container = st.container()
        
        with st.spinner("Summarizing Chunks..."):
            try:
                chunk_summaries = summarize_chunks(st.session_state.chunks, llm, parser)
            except Exception as e:
                st.error(f"Error summarizing chunks: {e}")
                st.stop()

        with st.spinner("Generating Final Summary..."):
            try:
                final_summary = generate_final_summary(chunk_summaries, llm, parser)
                container.write(final_summary)
                st.download_button(
                    label="Download Final Summary",
                    data=final_summary,
                    file_name="final_summary.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating final summary: {e}")
                st.stop()
    
    if st.session_state.conversation:
        st.divider()
        st.markdown("## Chat with your document")
        user_question = st.text_input("Ask a question about your document:")
        if user_question:
            handle_userinput(user_question)


if __name__ == "__main__":
    main()

