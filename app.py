from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

# get environment variables
from dotenv import load_dotenv
load_dotenv()

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps users understand documents."),
        ("user", "Question: {input}")
    ]
)

# Streamlit app
st.title("Ollama + LangChain Document QA -- For Learning Purposes")
st.write("Using Ollama model gemma:2b to answer questions")

input = st.text_input("Enter your question about the documents:")

## calling ollama gemma:2b model
llm = Ollama(model="gemma:2b")

## creating output parser
output_parser = StrOutputParser()

## creating chain pipeline
chain = prompt | llm | output_parser

## running the chain
if input:
    st.write(chain.invoke({"input": input}))