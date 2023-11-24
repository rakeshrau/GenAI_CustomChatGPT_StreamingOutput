import os
import openai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pathlib2 import Path
from dotenv import find_dotenv,load_dotenv

env_path = Path(__file__).parent/'env'
load_dotenv(dotenv_path=env_path)

# Secrets Authentication
VAULT_URL = os.getenv('VAULT_URL')

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
os.environ["OPENAI_DEPLOYMENT_NAME"]=secrets["azure_chat_openai_deployment_name"]
deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"]
openai.api_type="azure"
openai.api_key =os.environ["OPENAI_API_KEY"]
openai.api_base=os.environ["OPENAI_API_BASE"]
openai.api_version=os.environ["OPENAI_API_VERSION"]


#Page layout
st.set_page_config(
    page_title="Executive Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Function to add logo
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

# Sidebar contents
with st.sidebar:
    logo_path= Path(__file__).parent/'static/GSK_logo.png'
    my_logo = add_logo(logo_path=str(logo_path), width=200, height=113)
#Image on sidebar
    st.sidebar.image(my_logo)
    st.markdown(
        """
    ## About
    This is an LLM chatbot ⚡ using:
    - [OpenAI GPT4](https://openai.com/gpt-4) 
     
    
    
    by Digital Fuel-Decision Science & AI team
    """
    )
st.title("Executive Assistant 🤖")

form = st.form(key="my_form")
form.header("🔎")
query = form.text_area("Enter your prompt here:")
submittted = form.form_submit_button("Submit")

if submittted:
    res_box = st.empty()
    try:
        report=[]
        for resp in openai.ChatCompletion.create(deployment_id=deployment_name,messages= [{"role": "user", "content":query}],stream=True):
            report.append(resp['choices'][0]['delta'].get('content', ''))
            response = "".join(report).strip()
            res_box.markdown(f'{response}') 
      
    except openai.error.InvalidRequestError as e:
            if e.error.code == "content_filter":
                response = "Filtered Content"
                res_box.markdown(f'{response}') 

    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix(
            "Could not parse LLM output: `"
        ).removesuffix("`")
        





