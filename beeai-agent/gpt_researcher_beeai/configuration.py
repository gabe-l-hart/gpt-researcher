import os

from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    retriever: str = "duckduckgo"
    llm_api_base: str = "http://localhost:11434"
    llm_api_key: str = "ollama"
    llm_model: str = "ollama:granite3.3:8b"
    llm_model_fast: str | None = "ollama:granite3.3:8b"
    llm_model_smart: str | None = "ollama:granite3.3:8b"
    llm_model_strategic: str | None = "ollama:granite3.3:8b"
    embedding: str | None = "ollama:granite3.3:8b"


def load_env():
    config = Configuration()
    os.environ["RETRIEVER"] = config.retriever
    os.environ["OPENAI_BASE_URL"] = config.llm_api_base
    os.environ["OPENAI_API_KEY"] = config.llm_api_key
    if any(
        model.startswith("ollama:")
        for model in [
            config.llm_model,
            config.llm_model_fast,
            config.llm_model_smart,
            config.llm_model_strategic,
            config.embedding or ""
        ]
    ):
        os.environ["OLLAMA_BASE_URL"] = config.llm_api_base
        os.environ["OPENAI_BASE_URL"] = config.llm_api_base.rstrip("/") + "/v1"

    os.environ["FAST_LLM"] = f"{config.llm_model_fast or config.llm_model}"
    os.environ["SMART_LLM"] = f"{config.llm_model_smart or config.llm_model}"
    os.environ["STRATEGIC_LLM"] = f"{config.llm_model_strategic or config.llm_model}"
    if "granite" in os.environ["SMART_LLM"].lower():
        os.environ["PROMPT_FAMILY"] = "granite"
    if config.embedding:
        os.environ["EMBEDDING"] = config.embedding
