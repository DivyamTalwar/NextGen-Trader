"""
Models Router - Endpoints for LLM model configuration and management
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.schemas import (
    ModelInfo,
    ModelListResponse,
    BaseResponse,
    ResponseStatus
)
from llm.models import (
    LLM_ORDER,
    OLLAMA_LLM_ORDER,
    get_model_info,
    ModelProvider,
    LLMModel
)
from utils.ollama import is_ollama_installed, ensure_ollama_and_model

router = APIRouter()


@router.get("/list")
async def list_available_models():
    """Get list of all available LLM models"""
    try:
        models = []
        
        # Load API models
        api_models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "llm", "api_models.json")
        with open(api_models_path, 'r') as f:
            api_models = json.load(f)
        
        # Load Ollama models
        ollama_models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "llm", "ollama_models.json")
        with open(ollama_models_path, 'r') as f:
            ollama_models = json.load(f)
        
        # Process API models (they're in a flat list)
        for model_data in api_models:
            models.append(ModelInfo(
                name=model_data.get("model_name", model_data.get("name", "")),
                provider=model_data.get("provider", "unknown").lower(),
                context_window=model_data.get("context_window", 4096),
                supports_vision=model_data.get("supports_vision", False),
                supports_tools=model_data.get("supports_tools", False),
                is_available=True  # Assume API models are available
            ))
        
        # Process Ollama models
        ollama_available = is_ollama_installed()
        for model_data in ollama_models:
            models.append(ModelInfo(
                name=model_data.get("model_name", model_data.get("name", "")),
                provider="ollama",
                context_window=model_data.get("context_window", 4096),
                supports_vision=model_data.get("supports_vision", False),
                supports_tools=model_data.get("supports_tools", False),
                is_available=ollama_available
            ))
        
        # Get default model
        default_model = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
        default_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "groq")
        
        response = ModelListResponse(
            models=models,
            default_model=default_model,
            default_provider=default_provider
        )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(models)} available models",
            data=response.dict()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )


@router.get("/model/{provider}/{model_name}")
async def get_model_details(provider: str, model_name: str):
    """Get detailed information about a specific model"""
    try:
        # Try to get model info
        model_info = get_model_info(model_name, provider)
        
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' from provider '{provider}' not found"
            )
        
        # Check if model is actually available
        is_available = True
        required_env_var = None
        
        if provider.lower() == "openai":
            required_env_var = "OPENAI_API_KEY"
            is_available = bool(os.getenv(required_env_var))
        elif provider.lower() == "anthropic":
            required_env_var = "ANTHROPIC_API_KEY"
            is_available = bool(os.getenv(required_env_var))
        elif provider.lower() == "groq":
            required_env_var = "GROQ_API_KEY"
            is_available = bool(os.getenv(required_env_var))
        elif provider.lower() == "google":
            required_env_var = "GOOGLE_API_KEY"
            is_available = bool(os.getenv(required_env_var))
        elif provider.lower() == "deepseek":
            required_env_var = "DEEPSEEK_API_KEY"
            is_available = bool(os.getenv(required_env_var))
        elif provider.lower() == "ollama":
            is_available = is_ollama_installed()
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved details for {model_name}",
            data={
                "model": model_info,
                "is_available": is_available,
                "required_env_var": required_env_var,
                "provider": provider,
                "name": model_name
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model details: {str(e)}"
        )


@router.get("/providers")
async def list_providers():
    """Get list of all LLM providers"""
    providers = [
        {
            "name": "openai",
            "display_name": "OpenAI",
            "models_count": len([m for m in LLM_ORDER if "gpt" in m[1].lower()]),
            "requires_api_key": True,
            "env_variable": "OPENAI_API_KEY",
            "is_configured": bool(os.getenv("OPENAI_API_KEY"))
        },
        {
            "name": "anthropic",
            "display_name": "Anthropic",
            "models_count": len([m for m in LLM_ORDER if "claude" in m[1].lower()]),
            "requires_api_key": True,
            "env_variable": "ANTHROPIC_API_KEY",
            "is_configured": bool(os.getenv("ANTHROPIC_API_KEY"))
        },
        {
            "name": "groq",
            "display_name": "Groq",
            "models_count": len([m for m in LLM_ORDER if "llama" in m[1].lower() or "mixtral" in m[1].lower()]),
            "requires_api_key": True,
            "env_variable": "GROQ_API_KEY",
            "is_configured": bool(os.getenv("GROQ_API_KEY"))
        },
        {
            "name": "google",
            "display_name": "Google",
            "models_count": len([m for m in LLM_ORDER if "gemini" in m[1].lower()]),
            "requires_api_key": True,
            "env_variable": "GOOGLE_API_KEY",
            "is_configured": bool(os.getenv("GOOGLE_API_KEY"))
        },
        {
            "name": "deepseek",
            "display_name": "DeepSeek",
            "models_count": len([m for m in LLM_ORDER if "deepseek" in m[1].lower()]),
            "requires_api_key": True,
            "env_variable": "DEEPSEEK_API_KEY",
            "is_configured": bool(os.getenv("DEEPSEEK_API_KEY"))
        },
        {
            "name": "ollama",
            "display_name": "Ollama (Local)",
            "models_count": len(OLLAMA_LLM_ORDER),
            "requires_api_key": False,
            "env_variable": None,
            "is_configured": is_ollama_installed()
        }
    ]
    
    configured_count = sum(1 for p in providers if p["is_configured"])
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Retrieved {len(providers)} providers ({configured_count} configured)",
        data={
            "providers": providers,
            "total": len(providers),
            "configured": configured_count
        }
    )


@router.post("/ollama/install/{model_name}")
async def install_ollama_model(model_name: str):
    """Install a specific Ollama model locally"""
    try:
        if not is_ollama_installed():
            raise HTTPException(
                status_code=400,
                detail="Ollama is not installed on this system"
            )
        
        # Check if model exists in our list
        ollama_models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "llm", "ollama_models.json")
        with open(ollama_models_path, 'r') as f:
            ollama_models = json.load(f)
        
        model_found = any(m.get("model_name", m.get("name", "")) == model_name for m in ollama_models)
        if not model_found:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in Ollama models list"
            )
        
        # Try to ensure the model is installed
        success = ensure_ollama_and_model(model_name)
        
        if success:
            return BaseResponse(
                status=ResponseStatus.SUCCESS,
                message=f"Model '{model_name}' installed successfully",
                data={"model": model_name, "status": "installed"}
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to install model '{model_name}'"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error installing Ollama model: {str(e)}"
        )


@router.get("/recommended")
async def get_recommended_models():
    """Get recommended models based on availability and performance"""
    recommendations = []
    
    # Check which providers are configured
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    has_ollama = is_ollama_installed()
    
    if has_groq:
        recommendations.append({
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
            "reason": "Fast, free, and high-quality open-source model",
            "use_case": "General purpose analysis with good speed",
            "priority": 1
        })
    
    if has_openai:
        recommendations.append({
            "model": "gpt-4o",
            "provider": "openai",
            "reason": "State-of-the-art model with excellent reasoning",
            "use_case": "Complex analysis requiring deep understanding",
            "priority": 2
        })
    
    if has_anthropic:
        recommendations.append({
            "model": "claude-3-opus-20240229",
            "provider": "anthropic",
            "reason": "Excellent for detailed analysis and long context",
            "use_case": "In-depth financial analysis with nuanced reasoning",
            "priority": 2
        })
    
    if has_google:
        recommendations.append({
            "model": "gemini-1.5-pro",
            "provider": "google",
            "reason": "Good balance of speed and quality",
            "use_case": "Multi-modal analysis if needed",
            "priority": 3
        })
    
    if has_ollama:
        recommendations.append({
            "model": "llama3.2:3b",
            "provider": "ollama",
            "reason": "Local model for privacy and no API costs",
            "use_case": "Quick local analysis without internet dependency",
            "priority": 4
        })
    
    if not recommendations:
        recommendations.append({
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
            "reason": "Recommended default - get free API key from groq.com",
            "use_case": "Best free option for getting started",
            "priority": 1
        })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    
    return BaseResponse(
        status=ResponseStatus.SUCCESS,
        message=f"Generated {len(recommendations)} model recommendations",
        data={
            "recommendations": recommendations,
            "configured_providers": {
                "groq": has_groq,
                "openai": has_openai,
                "anthropic": has_anthropic,
                "google": has_google,
                "ollama": has_ollama
            }
        }
    )


@router.get("/test/{provider}/{model_name}")
async def test_model(provider: str, model_name: str):
    """Test if a specific model is working"""
    try:
        from llm.models import get_model
        from langchain_core.messages import HumanMessage
        
        # Try to get the model
        model = get_model(model_name, provider)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' from '{provider}' not found"
            )
        
        # Try a simple test prompt
        test_prompt = "Reply with 'Model working' if you receive this message."
        
        try:
            response = model.invoke([HumanMessage(content=test_prompt)])
            
            return BaseResponse(
                status=ResponseStatus.SUCCESS,
                message="Model test successful",
                data={
                    "model": model_name,
                    "provider": provider,
                    "status": "working",
                    "response": response.content if hasattr(response, 'content') else str(response)
                }
            )
        except Exception as e:
            return BaseResponse(
                status=ResponseStatus.ERROR,
                message="Model test failed",
                data={
                    "model": model_name,
                    "provider": provider,
                    "status": "failed",
                    "error": str(e)
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing model: {str(e)}"
        )