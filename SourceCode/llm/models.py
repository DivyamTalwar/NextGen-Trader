import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, List
from pathlib import Path
from typing import Union
from graph.state import AgentState


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if this is a placeholder for a custom model."""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file"""
    with open(json_path, 'r') as f:
        models_data = json.load(f)
    
    models = []
    for model_data in models_data:
        provider_enum = ModelProvider(model_data["provider"])
        models.append(
            LLMModel(
                display_name=model_data["display_name"],
                model_name=model_data["model_name"],
                provider=provider_enum
            )
        )
    return models


current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model information by model_name"""
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    return next((model for model in all_models if model.model_name == model_name and model.provider == model_provider), None)


def get_model(model_name: str, model_provider: ModelProvider, api_key: str = None) -> Union[ChatOpenAI,ChatGroq,ChatOllama,ChatAnthropic,ChatDeepSeek,ChatGoogleGenerativeAI,]:
    if model_provider == ModelProvider.GROQ:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API key not found. Please provide it or set GROQ_API_KEY in your .env file.")
        return ChatGroq(model=model_name, api_key=key)
    elif model_provider == ModelProvider.OPENAI:
        key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if not key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY in your .env file.")
        return ChatOpenAI(model=model_name, api_key=key, base_url=base_url)
    elif model_provider == ModelProvider.ANTHROPIC:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key not found. Please provide it or set ANTHROPIC_API_KEY in your .env file.")
        return ChatAnthropic(model=model_name, api_key=key)
    elif model_provider == ModelProvider.DEEPSEEK:
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DeepSeek API key not found. Please provide it or set DEEPSEEK_API_KEY in your .env file.")
        return ChatDeepSeek(model=model_name, api_key=key)
    elif model_provider == ModelProvider.GEMINI:
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API key not found. Please provide it or set GOOGLE_API_KEY in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=key)
    elif model_provider == ModelProvider.OLLAMA:
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
        )


def get_agent_model_config(state: AgentState) -> Tuple[str, str, str | None]:
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Returns model_name, model_provider, and api_key.
    """
    metadata = state.get("metadata", {})
    
    # Prioritize user-provided API key and model
    user_api_key = metadata.get("api_key")
    if user_api_key:
        model_name = metadata.get("model_name")
        model_provider = metadata.get("model_provider")
        return model_name, model_provider, user_api_key

    # Fallback to pre-configured models
    model_name = metadata.get("model_name", "gpt-4o")
    model_provider = metadata.get("model_provider", "OpenAI")
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider, None
