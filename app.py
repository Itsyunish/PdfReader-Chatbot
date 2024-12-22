import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,  allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def validate_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email)


def validate_phone(phone):
    pattern = r"^\+?\d{10,15}$"
    return re.match(pattern, phone)


def call_me():
    st.header("User Information Form")
    st.text("Fill in your details here: ")
    with st.form("Call_me_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        submitted = st.form_submit_button("Submit")
            
            
        if submitted:
            if not name.strip():
                st.error("Name field is required.")
            elif not email.strip():
                st.error("Email field is required.")
            elif not validate_email(email):
                st.error("Invalid email address.")
            elif not phone.strip():
                st.error("Phone field is required.")
            elif not validate_phone(phone):
                st.error("Invalid phone number. Please enter a valid number with 10-15 digits.")
            else:
                st.success(f"Form submitted successfully! We'll call you soon, {name}.")
                        
                    
def appointment():
    st.header("Appointment form")
    st.text("Fill in your details and date for appointment: ")

    with st.form("appointment_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        appointment_date = st.date_input("Appointment Date", value="today", min_value="today")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not name.strip():
                st.error("Name field is required.")
            elif not email.strip():
                st.error("Email field is required.")
            elif not validate_email(email):
                st.error("Invalid email address.")
            elif not phone.strip():
                st.error("Phone field is required.")
            elif not validate_phone(phone):
                st.error("Invalid phone number. Please enter a valid number with 10-15 digits.")
            else:
                st.success(f"Appointment booked successfully for {name} on {appointment_date}.")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)
        
    if user_question == "form":
        call_me()
    elif user_question == "book appointment":
        appointment()
          
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        submit_btn = st.button("Submit & Process")
        st.write("Note: Type 'form' to get the user form if you want us to contact you. For booking an appointmnet type 'book appointment' and select a valid date. ")

        if submit_btn:
            if not pdf_docs:
                st.error("Please upload PDF files before clicking 'Submit & Process'.")
                return
            
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")
                

if __name__ == "__main__":
    main()



