import streamlit as st
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
 
st.title("RAG System")
st.header("Ask me anything related to 'Leave No Context Behind' Paper")

# reading google api key
f = open(r"Keys\api_key.txt")
google_api_key = f.read()

# create the chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=google_api_key, model="gemini-1.5-pro-latest")

# reate the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")

# set up a connection with the Chroma for retrieval
connection = Chroma(persist_directory="Chroma_db", embedding_function=embedding_model)

# converting CHROMA connection to retriever object
retriever = connection.as_retriever(search_kwargs={"k": 5})

# user input
user_query = st.text_input("Enter your query")

# chat templates
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from user. Your answer should be based on the specific context."),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext: {Context}\nQuestion: {question}\nAnswer:")
])

# output parser for chatbot response
output_parser = StrOutputParser()

# function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"Context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

if st.button("Generate Answer"):
    if user_query:
        response = rag_chain.invoke(user_query)
        st.write(response)