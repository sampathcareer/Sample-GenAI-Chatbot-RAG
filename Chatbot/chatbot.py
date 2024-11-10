import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = ""  # OpenAI key

## Upload PDF files
st.header("My First Chatbot")


with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking question", type="pdf")


## Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)

# Break it into chunks  --> The purpose of breaking it into chunks is that there are small, small sections, right ? The OpenAI services get small tokens that it can read it, and it does not have to work on that entire file at the same time. So breaking it into small chunks helps make sure that you are working on a small section, and it is able to better understand that section, work on it better.

# So to break it into chunks, we are goging to use something that Lang Chain offers

## Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",   # What pattern, What identifier you want me to break the text on. (It could be a new line or it could be a tab or it could be any special characters, any normal characters that you want.) we are mentioning to break it on a new line.
        chunk_size=1000, # How big of a chunk you want to create?. we want to create chunks of 1000 characters
        chunk_overlap=150, ## For the relation. The algo to do is, bring me the last 150 characters from the previous chunk into this one also. So now your next chunk would not start from the next set of characters. it would also include those 150 characters and then the new ones. so that meaning, the context, the realtion is now retained.
        length_function=len   ## Length of the object
    )
    chunks = text_splitter.split_text(text) # It will break the whole text into small small chunks based on the above rule set that you given as parameter to RecursiveCharacterTextSplitter() function
    # st.write(chunks)


    ## Generating embeddings (To this we have to use the open AI service) 
    '''
    These are nothing but ready made LLMS, Just like OpenAI, Llama..etc
    '''
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    ## Creating vector store - FAISS(VECTOR STORE service that created by facebook). Facebook AI semantic search
    vector_store = FAISS.from_texts(chunks,embeddings) #It will accespt 2 parameters. one is chunk and another one is embeddings. ("a 123,b 2344" a is chunk and the number is chunk)like that it will store

    ''' 
    In the above 2 lines of code, there are 3 important things are happenings here, below are
    - embeddings (OpenAI)
    - initializing FAISS
    - store chunks & embeddings
    '''

    # get user question
    user_question = st.text_input("Type you question here")


    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)  ## First the question will convert to embedings, then it will do the semantic(similarity) search between the question and content of the vector store. Now the match would contain the list of all chunks that are present relevent to my question
        # st.write(match)

        # define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,   ## It means we are telling to our LLM, to not be not random. we are asking it to keep response very specific to the question. If the value will be higher like 0.4,0.8 or 2. you are asking the llm is okay for you to be random. And in that case, the LLM can go off the hook and start giving you lengthy answers. Things that are not related to what you actually need.
            max_tokens=1000,  ## It defines the limit of the response.
            # model_name = "gpt-3.5-turbo"  # Model name
            model_name = "GPT-2"
        )

    # output results
    '''
    Chain (chain of events) : (lot of action put together) sequence of the events. A chain of events.  do several thing together.
    chain -> take_the_question, get_relevant_document, pass_it_to_the_LLM, generate the output
    '''
    chain = load_qa_chain(llm, chain_type="stuff") 
    response = chain.run(input_documents = match, question = user_question)
    st.write(response)




