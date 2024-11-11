import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings  # Adjusted for Llama model
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp  # Adjusted for Llama model

# Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking question", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = LlamaCppEmbeddings()  # Adjusted for Llama model, no API key needed

    # Create vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Perform similarity search and define LLM
    if user_question:
        match = vector_store.similarity_search(user_question)

        # Define the Llama LLM
        llm = LlamaCpp(
            model_path="path/to/llama-model.bin",  # Set the path to your Llama model file
            temperature=0,
            max_tokens=1000
        )

        # Load QA chain and get response
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
