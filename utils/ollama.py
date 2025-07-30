from langchain_ollama import ChatOllama, OllamaEmbeddings
from utils import config as config


class OllamaLLM:
    def __init__(self):
        self.llm_model = config.OLLAMA_MODEL
        self.embedding_model= config.OLLAMA_EMBEDDING_MODEL

    def get_llm(self):
        llm=ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=self.llm_model,
            temperature=0,
        )
        return llm

    def get_embedding(self):
        embedding=OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=self.embedding_model,
        )
        return embedding