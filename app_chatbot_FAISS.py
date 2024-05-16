import os
import openai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema.messages import SystemMessage
from PIL import Image
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from pathlib2 import Path
from dotenv import find_dotenv,load_dotenv
import time
import logging
import sys
import re
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from langchain.chains import RetrievalQA
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

env_path = Path(__file__).parent/'.env'
load_dotenv(dotenv_path=env_path)

# Secrets Authentication
VAULT_URL = os.getenv('VAULT_URL')

@st.cache_resource
def get_secrets(vault_url: str) -> dict:
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    secrets = {}
    secrets["api_key"] = client.get_secret("open-api-key").value
    secrets["api_base"] = client.get_secret("azure-openai-api-base").value
    secrets["api_version"] = client.get_secret("azure-openai-api-version").value
    secrets["azure_chat_openai_deployment_name"] = client.get_secret("azure-chat-openai-deployment-name").value
    secrets["azure_chat_openai_model_name"] = client.get_secret("azure-chat-openai-model-name").value
    secrets["azure-embedding-openai-deployment-name"] = client.get_secret("azure-embedding-openai-deployment-name").value
    secrets["azure-embedding-openai-model-name"] = client.get_secret("azure-embedding-openai-model-name").value
    
    return secrets

# Get your secrets!
secrets = get_secrets(VAULT_URL)

os.environ["OPENAI_API_TYPE"] = "Azure"
os.environ["OPENAI_API_VERSION"] = secrets["api_version"]
os.environ["OPENAI_API_BASE"] = secrets["api_base"]
os.environ["OPENAI_API_KEY"] = secrets["api_key"]

#ADA Embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment=secrets["azure-embedding-openai-deployment-name"], model=secrets["azure-embedding-openai-model-name"], openai_api_base=secrets["api_base"],openai_api_type="azure",
)
#ChatPrompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
                You are a helpful assistant that is an expert in aswering the questions.
                IMPORTANT : Go step-by-step and answer in detail. 
                """
            )
        ),
        HumanMessagePromptTemplate.from_template("{pageContent,metadata}"),
    ]
)

#Vectorstore path
db_path = Path(__file__).parent/'FAISS_Vectorstore/'
filepath=Path(__file__).parent

VECTORDB_URL=os.getenv('VECTORDB_URL')
CONTAINER=os.getenv('CONTAINER')

#Method to download blob to a file
def download_blob_to_file(blob_service_client: BlobServiceClient, container_name,blob):
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob)
        with open(file=db_path/blob, mode="wb") as sample_blob:
            download_stream = blob_client.download_blob()
            sample_blob.write(download_stream.readall())
        sample_blob.close()
        
        
#Check if vectorestore exists else create it
if not os.path.isdir(db_path): 
    os.makedirs(db_path)
    #Blob way of fetching VectorDB
    blob_service_client = BlobServiceClient(
            account_url=VECTORDB_URL,
            credential=DefaultAzureCredential()) 
    #Download the Vectorstore
    download_blob_to_file(blob_service_client, CONTAINER,'index.pkl')        
    download_blob_to_file(blob_service_client, CONTAINER,'index.faiss') 

#Load the vectorstore
new_db = FAISS.load_local(db_path, embeddings)
#Retriever with document sources
retriever= new_db.as_retriever()

#Function to add logo
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

# Sidebar contents
with st.sidebar:
    logo_path= Path(__file__).parent/'static/company_logo.png'
    my_logo = add_logo(logo_path=str(logo_path), width=200, height=113)
#Image on sidebar
    st.sidebar.image(my_logo)
    st.markdown(
        """
    ## About
    This is a Generative AI based chatbot ⚡ using :[OpenAI GPT4](https://openai.com/gpt-4) 
    """
    )
    st.title("Instructions")
    st.write("\nClear the chat history by clicking on the 'Reset Chat' button. \n")
    reset_chat = st.button("Reset Chat")
    st.markdown(
        """
    ## Example prompts:
    - What are the Perceptions around STIs and HIV prevention?
    - How we might increase confidence and decrease vulnerability of HCPs?
    - In primary care and infectious disease,what we should be offering?
    """
    )
    add_vertical_space(1)
    st.markdown("""
        ##### ~ by Rakesh Raushan
    """)
    if reset_chat:
        st.session_state.messages = []
        st.session_state.cost = 0
        st.session_state.tokens = 0
st.title("Apretude Evidence Navigator 👩‍⚕️")

def seconds_to_time(total_seconds):
    # Ensure that the input is a non-negative number
    if total_seconds < 0:
        raise ValueError("Input must be a non-negative number of seconds")

    # Calculate hours, minutes, and seconds
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60

    # Format the time as HH:MM:SS
    time_format = "{:02d}:{:02d}:{:06.3f}".format(hours, minutes, seconds)

    return time_format


def gpt_streaming_call(prompt, llm1,placeholder_response):
    assistant_response = ""
    try:
        template = """ You are a helpful assistant that is expert in providing answers in detail with specific examples.
You will also include the medical terms, studies,trial data,references etc. to support your answers.
IMPORTANT:Include source reference documents.       
        
Question: {question}
        {summaries}
        Answer:
        Sources:    """ 
        
        
        PROMPT = PromptTemplate(template=template, input_variables=['question','summaries'])
        query=prompt
        response = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,chain_type="stuff",chain_type_kwargs={"prompt": PROMPT},retriever=retriever, return_source_documents=True)({"question":query}) #,return_only_outputs=True
        #response = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)({"question":query}) #,return_only_outputs=True
        messages=st.session_state.messages
        if response != None:
            #response['answer'] = re.sub(r'/dbfs/FileStore/Company/([^,]+\.pdf)', r'\1', response['answer'])
            #response['sources'] = re.sub(r'/dbfs/FileStore/Company/([^,]+\.pdf)', r'\1', response['sources'])

            assistant_response = assistant_response + response['answer'] + response['sources']
            #assistant_response = assistant_response + str(response)
            with placeholder_response.chat_message("assistant", avatar="🤖"):
                st.info(assistant_response)

    except openai.BadRequestError as e:
        if e.code == "content_filter":
            with st.chat_message("assistant", avatar="🤖"):
                st.error(
                    "**Unable to Display**: Your message has violated our policy. Please adhere to community guidelines and refrain from inappropriate behavior or language."
                )

    except ValueError as e:
        assistant_response = str(e)
        if not assistant_response.startswith("Could not parse LLM output: `"):
            raise e
        assistant_response = response.removeprefix(
            "Could not parse LLM output: `"
        ).removesuffix("`")

        message = st.empty()
        with placeholder_response.chat_message("assistant", avatar="🤖"):
            st.info(assistant_response)

    return assistant_response

#Custom CallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        #self.text=re.sub(r'/dbfs/FileStore/Company/([^,]+\.pdf)', r'\1', self.text)
        self.text+=token        
        self.container.markdown(self.text)

#Empty chatbox    
starter_message = st.empty()

with starter_message.chat_message("user", avatar="🤖"):
    st.write(f"What can I help you with today?")


if "messages" not in st.session_state:
    st.session_state.messages = []


if len(st.session_state.messages) > 0:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="🧑‍💻"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"], avatar="🤖"):
                st.info(message["content"])


if queries:= st.chat_input("Enter your question here:"):
    start = time.time()
    starter_message.empty()
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(f"{queries}")
        prefix='Go step-by-step,'
        suffix='.Explain in 10 paragraphs'
        #query = prefix + queries + suffix
        query=queries
        
    st.session_state.messages.append({"role": "user", "content": queries}) 

    with st.spinner("Wait for it..."):
        placeholder_response = st.empty()
        #Custom CallbackHandler initilization
        stream_handler = StreamHandler(placeholder_response)
        
        llm = AzureChatOpenAI(
            deployment_name=secrets["azure_chat_openai_deployment_name"],
            openai_api_type="azure",
            openai_api_key=secrets["api_key"],
            openai_api_base=secrets["api_base"],
            openai_api_version=secrets["api_version"],
            temperature=0,
            streaming=True,
            callbacks=[stream_handler]
        )
        qa_chain = create_qa_with_sources_chain(llm)
        final_qa_chain = StuffDocumentsChain(
                            llm_chain=qa_chain,
                            document_variable_name="context",
                            document_prompt=prompt_template,
                                              )
        #response = ""
        response = gpt_streaming_call(query, final_qa_chain, placeholder_response)
        st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
    
        end = time.time()
        expander = st.expander("Time Taken")
        expander.info(f"Time taken: {seconds_to_time(end - start)}") 

                
                  
        
        
        




