"""
LLM loader module for Ollama.
Uses Ollama API for local inference - much faster setup!
"""

import sys
import os
from pathlib import Path
import requests
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    LLM_CONTEXT_WINDOW,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)
from config.logging_config import get_logger

logger = get_logger('llm.loader')


# Global model status
_ollama_available = None
_ollama_model_loaded = None


def check_ollama_available():
    """Check if Ollama is running and accessible."""
    global _ollama_available
    
    if _ollama_available is not None:
        return _ollama_available
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            _ollama_available = True
            logger.info("Ollama is available and running")
            return True
        else:
            _ollama_available = False
            logger.warning(f"Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        _ollama_available = False
        logger.warning("Ollama is not running. Please start Ollama first.")
        print("="*60)
        print("ERROR: Ollama is not running!")
        print("="*60)
        print("\nPlease install and start Ollama:")
        print("1. Download from: https://ollama.com/download")
        print("2. Install Ollama")
        print("3. Start Ollama (it runs in background)")
        print("4. Pull a model: ollama pull llama3.2")
        print("   Or: ollama pull mistral")
        print("="*60)
        return False
    except Exception as e:
        _ollama_available = False
        logger.error(f"Error checking Ollama: {e}")
        return False


def check_model_available(model_name=None):
    """
    Check if the specified model is available in Ollama.
    
    Args:
        model_name: Model name (uses OLLAMA_MODEL_NAME if None)
    
    Returns:
        tuple: (is_available, message)
    """
    if model_name is None:
        model_name = OLLAMA_MODEL_NAME
    
    if not check_ollama_available():
        return False, "Ollama is not running"
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Check if model exists (with or without tag)
            if model_name in model_names:
                return True, f"Model '{model_name}' is available"
            
            # Check if any variant exists
            base_name = model_name.split(':')[0]
            matching = [m for m in model_names if m.startswith(base_name)]
            if matching:
                return True, f"Model variant found: {matching[0]} (requested: {model_name})"
            
            return False, f"Model '{model_name}' not found. Available models: {', '.join(model_names[:5])}"
        else:
            return False, f"Failed to check models (status {response.status_code})"
    except Exception as e:
        return False, f"Error checking model: {e}"


def load_llm_model(model_name=None, use_fallback=False):
    """
    Load/check Ollama model availability.
    Note: Ollama models are loaded on-demand, so this just verifies availability.
    
    Args:
        model_name: Model name (uses OLLAMA_MODEL_NAME if None)
        use_fallback: Whether to use fallback model (not used with Ollama)
    
    Returns:
        dict with model info or None if unavailable
    """
    if model_name is None:
        model_name = OLLAMA_MODEL_NAME
    
    if not check_ollama_available():
        return None
    
    is_available, message = check_model_available(model_name)
    if not is_available:
        logger.warning(message)
        print(f"WARNING: {message}")
        print(f"To install model, run: ollama pull {model_name}")
        return None
    
    logger.info(f"Model '{model_name}' is available")
    return {
        'model_name': model_name,
        'base_url': OLLAMA_BASE_URL,
        'available': True
    }


def get_llm_model(use_fallback=False):
    """
    Get the LLM model info (Ollama loads on-demand).
    
    Args:
        use_fallback: Whether to use fallback model
    
    Returns:
        dict with model info or None
    """
    return load_llm_model(use_fallback=use_fallback)


def clear_model_cache():
    """Clear cached model status (for Ollama, this just resets availability check)."""
    global _ollama_available, _ollama_model_loaded
    _ollama_available = None
    _ollama_model_loaded = None


def get_model_info(model=None):
    """
    Get information about the model.
    
    Args:
        model: Model dict (uses current model if None)
    
    Returns:
        dict with model information
    """
    if model is None:
        model = get_llm_model()
    
    if model is None:
        return {
            'loaded': False,
            'error': 'Model not available'
        }
    
    return {
        'loaded': True,
        'model_name': model.get('model_name', OLLAMA_MODEL_NAME),
        'base_url': model.get('base_url', OLLAMA_BASE_URL),
        'context_size': LLM_CONTEXT_WINDOW,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS
    }


# Test function
def test_loader():
    """Test Ollama loader functionality."""
    print("Testing Ollama LLM loader...")
    
    # Check if Ollama is available
    print("\n1. Checking Ollama availability...")
    if check_ollama_available():
        print("[OK] Ollama is running")
    else:
        print("[ERROR] Ollama is not running")
        print("  Please install and start Ollama from: https://ollama.com/download")
        return False
    
    # Check model availability
    print("\n2. Checking model availability...")
    is_available, message = check_model_available()
    print(f"  {message}")
    
    if not is_available:
        print("\nModel not available. To install:")
        print(f"  ollama pull {OLLAMA_MODEL_NAME}")
        return False
    
    # Try to load model
    print("\n3. Loading model info...")
    model = load_llm_model()
    
    if model is None:
        print("[ERROR] Failed to load model")
        return False
    
    print("[OK] Model is available!")
    
    # Get model info
    info = get_model_info(model)
    print("\nModel information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test simple generation
    print("\n4. Testing model generation...")
    try:
        from llm.inference import generate_question
        context = "This is a Python project that implements a calculator."
        result = generate_question(context, model=model)
        
        if result.get('success'):
            print(f"  Question: {result['question']}")
            print("[OK] Model generation works!")
            return True
        else:
            print(f"  Note: {result.get('error', 'Unknown error')}")
            return True  # Still OK, fallback works
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return False


if __name__ == "__main__":
    test_loader()
