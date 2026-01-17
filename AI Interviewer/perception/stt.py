"""
STT (Speech-to-Text) module for transcribing audio.
Uses OpenAI Whisper for local speech recognition.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import whisper
from config.settings import (
    get_whisper_model_name,
    AUDIO_SAMPLE_RATE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE
)


# Global model cache
_whisper_model = None


def load_whisper_model(model_name=None, device=None):
    """
    Load Whisper model (cached globally to avoid reloading).
    
    Args:
        model_name: Model name (uses get_whisper_model_name() if None)
        device: Device to use (uses WHISPER_DEVICE if None)
    
    Returns:
        Whisper model object
    """
    global _whisper_model
    
    if model_name is None:
        model_name = get_whisper_model_name()
    if device is None:
        device = WHISPER_DEVICE
    
    # Return cached model if already loaded
    if _whisper_model is not None:
        return _whisper_model
    
    try:
        print(f"Loading Whisper model: {model_name} (device: {device})...")
        _whisper_model = whisper.load_model(model_name, device=device)
        print(f"Whisper model loaded successfully!")
        return _whisper_model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None


def transcribe_audio(audio_data, model=None, sample_rate=None, language=None):
    """
    Transcribe audio data to text using Whisper.
    
    Args:
        audio_data: numpy array of audio data (int16 format)
        model: Whisper model (loads if None)
        sample_rate: Sample rate (uses AUDIO_SAMPLE_RATE if None)
        language: Language code (None = auto-detect)
    
    Returns:
        dict with keys:
            - 'text': Transcribed text
            - 'language': Detected language
            - 'segments': List of segments with timestamps
            - 'confidence': Average confidence (if available)
    """
    if audio_data is None or len(audio_data) == 0:
        return {
            'text': '',
            'language': None,
            'segments': [],
            'confidence': 0.0
        }
    
    try:
        # Load model if not provided
        if model is None:
            model = load_whisper_model()
            if model is None:
                return {
                    'text': '',
                    'language': None,
                    'segments': [],
                    'confidence': 0.0
                }
        
        if sample_rate is None:
            sample_rate = AUDIO_SAMPLE_RATE
        
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            # Convert int16 to float32 and normalize to [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Check audio level (for debugging)
        rms_level = np.sqrt(np.mean(audio_data**2))
        max_level = np.max(np.abs(audio_data))
        
        # Normalize audio if too quiet (boost volume)
        if max_level < 0.1:  # Audio is too quiet
            audio_data = audio_data * (0.5 / max_level) if max_level > 0 else audio_data
        
        # Transcribe with better settings
        result = model.transcribe(
            audio_data,
            language=language,
            task="transcribe",
            fp16=False,  # Always use FP32 on CPU
            verbose=False,  # Suppress verbose output
            condition_on_previous_text=True,  # Better context
            initial_prompt=None,  # Can add prompt for better accuracy
            temperature=0.0,  # More deterministic
            best_of=1,  # Faster
            beam_size=5  # Better accuracy
        )
        
        # Extract segments
        segments = []
        if 'segments' in result:
            for seg in result['segments']:
                segments.append({
                    'start': seg.get('start', 0.0),
                    'end': seg.get('end', 0.0),
                    'text': seg.get('text', '').strip()
                })
        
        # Calculate average confidence if available
        confidence = 0.0
        if segments:
            # Whisper doesn't provide confidence scores directly,
            # but we can estimate based on segment quality
            confidence = 85.0  # Placeholder - Whisper doesn't expose confidence
        
        return {
            'text': result.get('text', '').strip(),
            'language': result.get('language', None),
            'segments': segments,
            'confidence': confidence
        }
    
    except Exception as e:
        print(f"Error in transcription: {e}")
        return {
            'text': '',
            'language': None,
            'segments': [],
            'confidence': 0.0
        }


def transcribe_audio_simple(audio_data, model=None, sample_rate=None):
    """
    Simple transcription - returns just the text string.
    
    Args:
        audio_data: numpy array of audio data
        model: Whisper model (loads if None)
        sample_rate: Sample rate
    
    Returns:
        str: Transcribed text
    """
    result = transcribe_audio(audio_data, model=model, sample_rate=sample_rate)
    return result['text']


def transcribe_audio_file(filepath, model=None, language=None):
    """
    Transcribe audio from a file.
    
    Args:
        filepath: Path to audio file (WAV, MP3, etc.)
        model: Whisper model (loads if None)
        language: Language code (None = auto-detect)
    
    Returns:
        dict with transcription results
    """
    try:
        if model is None:
            model = load_whisper_model()
            if model is None:
                return {
                    'text': '',
                    'language': None,
                    'segments': [],
                    'confidence': 0.0
                }
        
        # Whisper can load audio files directly
        result = model.transcribe(
            filepath,
            language=language,
            task="transcribe",
            fp16=False if WHISPER_COMPUTE_TYPE == "float32" else True
        )
        
        segments = []
        if 'segments' in result:
            for seg in result['segments']:
                segments.append({
                    'start': seg.get('start', 0.0),
                    'end': seg.get('end', 0.0),
                    'text': seg.get('text', '').strip()
                })
        
        return {
            'text': result.get('text', '').strip(),
            'language': result.get('language', None),
            'segments': segments,
            'confidence': 85.0  # Placeholder
        }
    
    except Exception as e:
        print(f"Error transcribing file: {e}")
        return {
            'text': '',
            'language': None,
            'segments': [],
            'confidence': 0.0
        }


def get_whisper_available_models():
    """
    Get list of available Whisper models.
    
    Returns:
        list of model names
    """
    return ["tiny", "base", "small", "medium", "large"]


def clear_model_cache():
    """Clear the cached Whisper model (useful for memory management)."""
    global _whisper_model
    _whisper_model = None


def test_stt():
    """
    Test STT functionality.
    """
    print("Testing STT engine (Whisper)...")
    
    # Check if model can be loaded
    model_name = get_whisper_model_name()
    print(f"Model: {model_name}")
    print(f"Device: {WHISPER_DEVICE}")
    print(f"Compute type: {WHISPER_COMPUTE_TYPE}")
    
    model = load_whisper_model()
    if model is None:
        print("\nERROR: Failed to load Whisper model!")
        print("Make sure Whisper is installed: pip install openai-whisper")
        print("The model will be downloaded automatically on first use.")
        return False
    
    print("\nModel loaded successfully!")
    
    # Try to transcribe from a test audio file if it exists
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"\nTranscribing test file: {test_file}")
        result = transcribe_audio_file(test_file, model=model)
        
        print(f"\nTranscription Results:")
        print(f"  Language: {result['language']}")
        print(f"  Text: {result['text']}")
        print(f"  Segments: {len(result['segments'])}")
        
        if result['segments']:
            print("\nSegments:")
            for i, seg in enumerate(result['segments'][:5]):  # Show first 5
                print(f"  [{i+1}] {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
    else:
        print(f"\nTest file '{test_file}' not found.")
        print("You can create one by running: python capture/audio_capture.py")
        print("Then run this test again.")
    
    # Test with live audio capture
    print("\n" + "="*60)
    print("Testing live audio transcription...")
    print("(This will record 5 seconds of audio)")
    print("="*60)
    
    try:
        from capture.audio_capture import create_audio_stream, capture_audio_chunk
        
        p, stream = create_audio_stream()
        if stream is None:
            print("Failed to create audio stream")
            return False
        
        print("\nRecording 5 seconds... (please speak)")
        audio_data = capture_audio_chunk(stream, duration_seconds=5.0)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if audio_data is not None:
            # Check audio quality
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            max_amp = np.max(np.abs(audio_data))
            duration = len(audio_data) / AUDIO_SAMPLE_RATE
            print(f"\nAudio quality check:")
            print(f"  RMS level: {rms:.4f}")
            print(f"  Max amplitude: {max_amp}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Sample rate: {AUDIO_SAMPLE_RATE} Hz")
            
            # Save audio for verification
            from capture.audio_capture import save_audio_chunk
            save_audio_chunk(audio_data, "test_live_audio.wav", AUDIO_SAMPLE_RATE)
            print(f"  Audio saved to 'test_live_audio.wav' for verification")
            
            # Check if audio is too quiet
            if max_amp < 1000:  # Very quiet for int16
                print(f"  WARNING: Audio seems very quiet. Check microphone volume!")
            
            print("\nTranscribing...")
            result = transcribe_audio(audio_data, model=model)
            
            print(f"\nLive Transcription Results:")
            print(f"  Language: {result['language']}")
            print(f"  Text: {result['text']}")
            print(f"  Segments: {len(result['segments'])}")
            
            if result['segments']:
                print("\nDetailed segments:")
                for i, seg in enumerate(result['segments']):
                    print(f"  [{i+1}] {seg['start']:.2f}s - {seg['end']:.2f}s: '{seg['text']}'")
            
            return True
        else:
            print("Failed to capture audio")
            return False
    
    except Exception as e:
        print(f"Error in live test: {e}")
        return False


if __name__ == "__main__":
    test_stt()
