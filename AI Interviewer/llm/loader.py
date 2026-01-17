"""
LLM loader module for loading LLaMA models using llama.cpp.
Supports local CPU-based inference with llama-cpp-python.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    LLM_MODEL_PATH,
    LLM_FALLBACK_MODEL_PATH,
    LLM_CONTEXT_WINDOW,
    LLM_N_THREADS,
    LLM_N_GPU_LAYERS
)
from config.logging_config import get_logger

logger = get_logger('llm.loader')


# Global model cache
_llm_model = None
_llm_fallback_model = None


def load_llm_model(model_path=None, use_fallback=False):
    """
    Load LLaMA model using llama-cpp-python.
    
    Args:
        model_path: Path to model file (uses LLM_MODEL_PATH if None)
        use_fallback: Whether to use fallback model
    
    Returns:
        Llama model object or None if failed
    """
    global _llm_model, _llm_fallback_model
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("="*60)
        print("ERROR: llama-cpp-python not installed!")
        print("="*60)
        print("\nWindows Installation Instructions:")
        print("\n1. Install Visual Studio Build Tools:")
        print("   - Download: https://visualstudio.microsoft.com/downloads/")
        print("   - Install 'Build Tools for Visual Studio 2022'")
        print("   - Select 'Desktop development with C++' workload")
        print("   - Install (takes ~3-6GB, 10-30 minutes)")
        print("\n2. Then install llama-cpp-python:")
        print("   pip install llama-cpp-python")
        print("\n3. Alternative: Try pre-built wheel:")
        print("   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
        print("\nSee INSTALL_LLM.md for detailed instructions.")
        print("="*60)
        return None
    
    # Determine which model to load
    if use_fallback:
        if _llm_fallback_model is not None:
            return _llm_fallback_model
        
        model_file = model_path or LLM_FALLBACK_MODEL_PATH
        if model_file is None:
            print("ERROR: Fallback model path not configured")
            return None
    else:
        if _llm_model is not None:
            return _llm_model
        
        model_file = model_path or LLM_MODEL_PATH
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        print("Please download the LLaMA-3-8B-Instruct GGUF model and place it in the models/ directory")
        return None
    
        logger.info(f"Loading LLM model: {model_file}")
        logger.debug(f"  Context window: {LLM_CONTEXT_WINDOW}")
        logger.debug(f"  Threads: {LLM_N_THREADS}")
        logger.debug(f"  GPU layers: {LLM_N_GPU_LAYERS}")
    
    try:
        # Load model
        model = Llama(
            model_path=model_file,
            n_ctx=LLM_CONTEXT_WINDOW,
            n_threads=LLM_N_THREADS,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            verbose=False
        )
        
        # Cache model
        if use_fallback:
            _llm_fallback_model = model
        else:
            _llm_model = model
        
        logger.info("Model loaded successfully!")
        return model
    
    except Exception as e:
        logger.error(f"ERROR loading model: {e}")
        logger.error("Troubleshooting:")
        logger.error("1. Make sure the model file is a valid GGUF format")
        logger.error("2. Check if you have enough RAM (8B model needs ~8-10GB)")
        logger.error("3. Try reducing n_ctx (context window) if memory is limited")
        return None


def get_llm_model(use_fallback=False):
    """
    Get the loaded LLM model (loads if not already loaded).
    
    Args:
        use_fallback: Whether to use fallback model
    
    Returns:
        Llama model object or None
    """
    if use_fallback:
        if _llm_fallback_model is None:
            return load_llm_model(use_fallback=True)
        return _llm_fallback_model
    else:
        if _llm_model is None:
            return load_llm_model(use_fallback=False)
        return _llm_model


def clear_model_cache():
    """Clear cached models (useful for memory management)."""
    global _llm_model, _llm_fallback_model
    _llm_model = None
    _llm_fallback_model = None


def check_model_available(model_path=None):
    """
    Check if model file is available.
    
    Args:
        model_path: Path to model (uses LLM_MODEL_PATH if None)
    
    Returns:
        tuple: (is_available, message)
    """
    if model_path is None:
        model_path = LLM_MODEL_PATH
    
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    
    # Check file size (GGUF files are typically large)
    file_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)  # GB
    if file_size < 1:
        return False, f"Model file seems too small ({file_size:.2f} GB). Expected ~4-8GB for 8B model."
    
    return True, f"Model file found: {model_path} ({file_size:.2f} GB)"


def get_model_info(model=None):
    """
    Get information about the loaded model.
    
    Args:
        model: Llama model object (uses cached model if None)
    
    Returns:
        dict with model information
    """
    if model is None:
        model = get_llm_model()
    
    if model is None:
        return {
            'loaded': False,
            'error': 'Model not loaded'
        }
    
    try:
        # Get model context size
        n_ctx = getattr(model, 'n_ctx', LLM_CONTEXT_WINDOW)
        
        return {
            'loaded': True,
            'context_size': n_ctx,
            'threads': LLM_N_THREADS,
            'gpu_layers': LLM_N_GPU_LAYERS,
            'model_path': LLM_MODEL_PATH
        }
    except Exception as e:
        return {
            'loaded': False,
            'error': str(e)
        }


# Test function
def test_loader():
    """Test LLM loader functionality."""
    print("Testing LLM loader...")
    
    # Check if llama-cpp-python is installed
    try:
        import llama_cpp
        print("✓ llama-cpp-python is installed")
    except ImportError:
        print("✗ llama-cpp-python is not installed")
        print("  Install with: pip install llama-cpp-python")
        return False
    
    # Check model availability
    print("\nChecking model availability...")
    is_available, message = check_model_available()
    print(f"  {message}")
    
    if not is_available:
        print("\nModel not available. Please:")
        print("1. Download LLaMA-3-8B-Instruct GGUF model")
        print("2. Place it in the models/ directory")
        print("3. Update LLM_MODEL_PATH in config/settings.py if needed")
        return False
    
    # Try to load model
    print("\nLoading model...")
    model = load_llm_model()
    
    if model is None:
        print("✗ Failed to load model")
        return False
    
    print("✓ Model loaded successfully!")
    
    # Get model info
    info = get_model_info(model)
    print("\nModel information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test simple generation
    print("\nTesting model generation...")
    try:
        response = model(
            "Hello, how are you?",
            max_tokens=20,
            temperature=0.7,
            stop=["\n", "Human:", "User:"],
            echo=False
        )
        
        generated_text = response['choices'][0]['text'].strip()
        print(f"  Response: {generated_text}")
        print("✓ Model generation works!")
        
        return True
    
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


if __name__ == "__main__":
    test_loader()
