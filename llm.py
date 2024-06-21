from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import openai
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOpenAI
from prompt_templates import memory_prompt_template
import yaml

# Setup LLM. Fetch base_url from LM Studio
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Build a conversational chain
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device':'cpu'})

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm = llm, prompt = chat_prompt, memory = memory)

def load_normal_chain(chat_history):
    return ChatChain(chat_history)

class ChatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input=user_input, history = self.memory.chat_memory.messages,stop = ["Human: "])
    
