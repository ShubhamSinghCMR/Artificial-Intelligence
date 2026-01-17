"""
Configuration settings for AI Interviewer system.
All paths and settings are centralized here for easy modification.
"""

import os
from pathlib import Path
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# ============================================================================
# PATHS
# ============================================================================

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

def get_tesseract_path():
    """
    Get Tesseract OCR executable path.
    Checks environment variable first, then common locations.
    """
    # Check environment variable first
    env_path = os.getenv("TESSERACT_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path
    
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
    ]
    
    # Check if tesseract is in PATH
    import shutil
    if shutil.which("tesseract"):
        return "tesseract"
    
    # Check common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Default - user must set this
    return None

def get_llm_model_path():
    """Get path to LLaMA model file (GGUF format for llama.cpp)."""
    project_root = get_project_root()
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Default model name - user should download and place here
    return str(model_dir / "llama-3-8b-instruct.gguf")

def get_whisper_model_name():
    """Get Whisper model name (tiny, base, small, medium, large)."""
    # Check environment variable first
    return os.getenv("WHISPER_MODEL", "base")

def get_logs_dir():
    """Get directory for session logs."""
    project_root = get_project_root()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    return str(logs_dir)

# ============================================================================
# CAPTURE SETTINGS
# ============================================================================

# Screen Capture
SCREEN_CAPTURE_FPS = 2  # Capture frames every 0.5 seconds (2 fps)
# Screen capture region from env (format: "x,y,width,height" or empty for full screen)
_screen_region = os.getenv("SCREEN_CAPTURE_REGION", "").strip()
if _screen_region:
    try:
        SCREEN_CAPTURE_REGION = tuple(map(int, _screen_region.split(',')))
    except:
        SCREEN_CAPTURE_REGION = None
else:
    SCREEN_CAPTURE_REGION = None
SCREEN_CAPTURE_MONITOR = int(os.getenv("SCREEN_CAPTURE_MONITOR", "0"))  # Monitor index (0 = primary)

# Audio Capture
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))  # Whisper works best with 16kHz
AUDIO_CHUNK_SIZE = 4096  # Audio buffer size in frames
AUDIO_CHANNELS = 1  # Mono audio
AUDIO_FORMAT = "int16"  # 16-bit PCM
# Audio device index from env (None = default microphone)
_audio_device = os.getenv("AUDIO_DEVICE_INDEX", "").strip()
AUDIO_DEVICE_INDEX = int(_audio_device) if _audio_device and _audio_device.isdigit() else None

# ============================================================================
# PROCESSING INTERVALS
# ============================================================================

OCR_PROCESS_INTERVAL = float(os.getenv("OCR_PROCESS_INTERVAL", "3.0"))  # Run OCR every 3 seconds (not every frame)
STT_CHUNK_DURATION = float(os.getenv("STT_CHUNK_DURATION", "3.0"))  # Process audio in 3-second chunks (faster response)
FRAME_SAMPLE_INTERVAL = 2.0  # Sample frames every 2 seconds for change detection
CHANGE_DETECTION_THRESHOLD = 0.1  # Threshold for frame change (0.0-1.0)

# Silence Detection
SILENCE_THRESHOLD = 500  # Audio amplitude threshold for silence
SILENCE_DURATION = 1.5  # Seconds of silence before considering pause
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to process

# ============================================================================
# LLM SETTINGS (Ollama)
# ============================================================================

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Default Ollama API URL
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")  # Model name (options: llama3.2, mistral, llama3.1, etc.)
# To use a different model, change this and run: ollama pull <model_name>
# Popular models: llama3.2 (3B, fast), llama3.1 (8B, better quality), mistral (7B)

# LLM Generation Settings
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))  # Context window size (tokens)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))  # Creativity (0.0-1.0, lower = more deterministic)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))  # Max tokens per generation
LLM_TOP_P = 0.9  # Nucleus sampling
LLM_TOP_K = 40  # Top-k sampling

# Legacy settings (kept for compatibility, not used with Ollama)
LLM_MODEL_PATH = get_llm_model_path()  # Not used with Ollama
LLM_N_THREADS = 4  # Not used with Ollama
LLM_N_GPU_LAYERS = 0  # Not used with Ollama
LLM_FALLBACK_MODEL_PATH = None  # Not used with Ollama

# ============================================================================
# INTERVIEW SETTINGS
# ============================================================================

MIN_QUESTION_INTERVAL = float(os.getenv("MIN_QUESTION_INTERVAL", "30.0"))  # Minimum seconds between questions
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "10"))  # Maximum questions in one session
QUESTION_TIMEOUT = 60.0  # Seconds to wait for answer before next question
CONTEXT_WINDOW_SECONDS = 180  # Keep last 3 minutes of context

# Question Generation
INITIAL_QUESTION_DELAY = float(os.getenv("INITIAL_QUESTION_DELAY", "10.0"))  # Wait 10 seconds before first question
FOLLOW_UP_ENABLED = True  # Enable follow-up questions
MAX_FOLLOW_UPS = 2  # Maximum follow-ups per topic

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Scoring Criteria (weights sum to 1.0)
SCORE_WEIGHTS = {
    "technical_depth": 0.30,
    "clarity": 0.25,
    "originality": 0.20,
    "understanding": 0.25,
}

# Score Ranges
SCORE_MIN = 1.0
SCORE_MAX = 10.0
SCORE_PASSING = 6.0  # Minimum passing score

# Evaluation Thresholds
TECHNICAL_DEPTH_KEYWORDS = [
    "algorithm", "complexity", "optimization", "architecture", "design pattern",
    "scalability", "performance", "security", "testing", "deployment"
]

CLARITY_THRESHOLD = 0.6  # Minimum clarity score
UNDERSTANDING_THRESHOLD = 0.5  # Minimum understanding score

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Processing
ENABLE_FRAME_CACHING = True  # Cache OCR results for unchanged frames
MAX_CACHE_SIZE = 100  # Maximum cached frames
ENABLE_PARALLEL_PROCESSING = True  # Process OCR and STT in parallel

# LLM
LLM_BATCH_SIZE = 1  # Batch size for LLM inference
LLM_USE_CACHE = True  # Cache common question patterns

# Whisper
WHISPER_DEVICE = "cpu"  # "cpu" or "cuda" if available
WHISPER_COMPUTE_TYPE = "int8"  # "int8", "float16", "float32"

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_FILE_PATH = os.path.join(get_logs_dir(), "interviewer.log")
SESSION_LOG_PATH = os.path.join(get_logs_dir(), "session.json")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_settings():
    """Get all settings as a dictionary for easy inspection."""
    return {
        "paths": {
            "tesseract": get_tesseract_path(),
            "ollama_model": OLLAMA_MODEL_NAME,
            "ollama_url": OLLAMA_BASE_URL,
            "whisper_model": get_whisper_model_name(),
            "logs_dir": get_logs_dir(),
        },
        "capture": {
            "screen_fps": SCREEN_CAPTURE_FPS,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "audio_chunk_size": AUDIO_CHUNK_SIZE,
        },
        "processing": {
            "ocr_interval": OCR_PROCESS_INTERVAL,
            "stt_chunk_duration": STT_CHUNK_DURATION,
            "frame_sample_interval": FRAME_SAMPLE_INTERVAL,
        },
        "llm": {
            "ollama_model": OLLAMA_MODEL_NAME,
            "ollama_url": OLLAMA_BASE_URL,
            "context_window": LLM_CONTEXT_WINDOW,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        "interview": {
            "min_question_interval": MIN_QUESTION_INTERVAL,
            "max_questions": MAX_QUESTIONS_PER_SESSION,
            "context_window_seconds": CONTEXT_WINDOW_SECONDS,
        },
        "evaluation": {
            "score_weights": SCORE_WEIGHTS,
            "score_range": (SCORE_MIN, SCORE_MAX),
            "passing_score": SCORE_PASSING,
        },
    }

def validate_settings():
    """Validate that all required settings are configured."""
    errors = []
    
    # Check Tesseract path
    tesseract_path = get_tesseract_path()
    if not tesseract_path:
        errors.append("Tesseract OCR not found. Please install Tesseract or set path in settings.py")
    
    # Check LLM model path
    # Check Ollama availability (instead of model file)
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            errors.append("Ollama is not running. Please install and start Ollama from https://ollama.com/download")
        else:
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            if OLLAMA_MODEL_NAME not in model_names:
                # Check if any variant exists
                base_name = OLLAMA_MODEL_NAME.split(':')[0]
                matching = [m for m in model_names if m.startswith(base_name)]
                if not matching:
                    errors.append(f"Ollama model '{OLLAMA_MODEL_NAME}' not found. Run 'ollama pull {OLLAMA_MODEL_NAME}' to download it.")
    except requests.exceptions.ConnectionError:
        errors.append("Ollama is not running. Please install and start Ollama from https://ollama.com/download")
    except Exception as e:
        errors.append(f"Error checking Ollama: {e}")
    
    return errors
