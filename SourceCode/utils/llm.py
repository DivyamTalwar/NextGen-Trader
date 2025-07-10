"""Helper functions for LLM"""

import json
import time
from pydantic import BaseModel
from llm.models import get_model, get_model_info
from utils.progress import progress
from graph.state import AgentState


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider, api_key = get_agent_model_config(state, agent_name)
    else:
        model_name, model_provider, api_key = "gpt-4o", "OpenAI", None

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_key=api_key)

    structured_llm = llm
    if model_info and model_info.has_json_mode():
        structured_llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
    )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = structured_llm .invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            if "rate_limit_exceeded" in str(e):
                time.sleep(20) # Wait for 20 seconds before retrying

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
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
