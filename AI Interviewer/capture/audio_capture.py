"""
Audio capture module for capturing microphone input.
Uses PyAudio for audio streaming and includes silence detection.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import pyaudio
from config.settings import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHUNK_SIZE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    AUDIO_DEVICE_INDEX,
    STT_CHUNK_DURATION,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION
)


def get_audio_format():
    """Convert format string to PyAudio format constant."""
    format_map = {
        "int16": pyaudio.paInt16,
        "int32": pyaudio.paInt32,
        "float32": pyaudio.paFloat32,
    }
    return format_map.get(AUDIO_FORMAT, pyaudio.paInt16)


def list_audio_devices():
    """
    List all available audio input devices.
    
    Returns:
        list of dicts with device info
    """
    p = pyaudio.PyAudio()
    devices = []
    
    try:
        device_count = p.get_device_count()
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
    finally:
        p.terminate()
    
    return devices


def create_audio_stream(device_index=None, chunk_size=None, sample_rate=None):
    """
    Create and start an audio input stream.
    
    Args:
        device_index: Audio device index (None = default)
        chunk_size: Buffer size in frames (uses AUDIO_CHUNK_SIZE if None)
        sample_rate: Sample rate (uses AUDIO_SAMPLE_RATE if None)
    
    Returns:
        tuple: (PyAudio instance, stream object) or (None, None) if failed
    """
    try:
        p = pyaudio.PyAudio()
        
        if chunk_size is None:
            chunk_size = AUDIO_CHUNK_SIZE
        if sample_rate is None:
            sample_rate = AUDIO_SAMPLE_RATE
        if device_index is None:
            device_index = AUDIO_DEVICE_INDEX
        
        stream = p.open(
            format=get_audio_format(),
            channels=AUDIO_CHANNELS,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
        
        return p, stream
    
    except Exception as e:
        print(f"Error creating audio stream: {e}")
        return None, None


def read_audio_chunk(stream, chunk_size=None):
    """
    Read a single audio chunk from the stream.
    
    Args:
        stream: PyAudio stream object
        chunk_size: Number of frames to read (uses AUDIO_CHUNK_SIZE if None)
    
    Returns:
        numpy array of audio data or None if failed
    """
    try:
        if chunk_size is None:
            chunk_size = AUDIO_CHUNK_SIZE
        
        data = stream.read(chunk_size, exception_on_overflow=False)
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        return audio_array
    
    except Exception as e:
        print(f"Error reading audio chunk: {e}")
        return None


def is_silence(audio_chunk, threshold=None):
    """
    Detect if audio chunk is silence.
    
    Args:
        audio_chunk: numpy array of audio data
        threshold: Amplitude threshold (uses SILENCE_THRESHOLD if None)
    
    Returns:
        bool: True if silence, False otherwise
    """
    if threshold is None:
        threshold = SILENCE_THRESHOLD
    
    if audio_chunk is None or len(audio_chunk) == 0:
        return True
    
    # Calculate RMS (Root Mean Square) amplitude
    rms = np.sqrt(np.mean(audio_chunk**2))
    
    return rms < threshold


def capture_audio_chunk(stream, duration_seconds=None):
    """
    Capture audio for specified duration.
    
    Args:
        stream: PyAudio stream object
        duration_seconds: Duration in seconds (uses STT_CHUNK_DURATION if None)
    
    Returns:
        numpy array of audio data or None if failed
    """
    if duration_seconds is None:
        duration_seconds = STT_CHUNK_DURATION
    
    sample_rate = stream._rate
    chunk_size = stream._frames_per_buffer
    total_frames = int(sample_rate * duration_seconds)
    
    chunks = []
    frames_read = 0
    
    while frames_read < total_frames:
        chunk = read_audio_chunk(stream, min(chunk_size, total_frames - frames_read))
        if chunk is not None:
            chunks.append(chunk)
            frames_read += len(chunk)
        else:
            break
    
    if chunks:
        return np.concatenate(chunks)
    return None


def audio_chunk_generator(stream, chunk_duration=None, max_chunks=None):
    """
    Generator that yields audio chunks of specified duration.
    
    Args:
        stream: PyAudio stream object
        chunk_duration: Duration in seconds (uses STT_CHUNK_DURATION if None)
        max_chunks: Maximum chunks to yield (None = infinite)
    
    Yields:
        tuple: (audio_data, timestamp) where audio_data is numpy array, timestamp is float
    """
    if chunk_duration is None:
        chunk_duration = STT_CHUNK_DURATION
    
    chunk_count = 0
    
    while True:
        if max_chunks is not None and chunk_count >= max_chunks:
            break
        
        timestamp = time.time()
        audio_data = capture_audio_chunk(stream, chunk_duration)
        
        if audio_data is not None:
            yield (audio_data, timestamp)
            chunk_count += 1
        else:
            break


def detect_speech_segments(stream, min_speech_duration=None, silence_duration=None):
    """
    Detect continuous speech segments (between silence periods).
    
    Args:
        stream: PyAudio stream object
        min_speech_duration: Minimum speech duration to return (uses MIN_SPEECH_DURATION if None)
        silence_duration: Silence duration to end segment (uses SILENCE_DURATION if None)
    
    Yields:
        tuple: (audio_data, timestamp) for each speech segment
    """
    if min_speech_duration is None:
        min_speech_duration = MIN_SPEECH_DURATION
    if silence_duration is None:
        silence_duration = SILENCE_DURATION
    
    sample_rate = stream._rate
    chunk_size = stream._frames_per_buffer
    silence_frames_threshold = int(sample_rate * silence_duration / chunk_size)
    min_speech_frames = int(sample_rate * min_speech_duration / chunk_size)
    
    speech_buffer = []
    silence_count = 0
    in_speech = False
    
    while True:
        chunk = read_audio_chunk(stream)
        if chunk is None:
            break
        
        timestamp = time.time()
        is_silent = is_silence(chunk)
        
        if is_silent:
            silence_count += 1
            if in_speech:
                # Check if we've had enough silence to end speech
                if silence_count >= silence_frames_threshold:
                    # End of speech segment
                    if len(speech_buffer) >= min_speech_frames:
                        audio_data = np.concatenate(speech_buffer)
                        yield (audio_data, timestamp)
                    speech_buffer = []
                    in_speech = False
                    silence_count = 0
        else:
            # Speech detected
            silence_count = 0
            in_speech = True
            speech_buffer.append(chunk)
            
            # If buffer gets too large, yield it
            if len(speech_buffer) > 1000:  # Prevent memory issues
                audio_data = np.concatenate(speech_buffer)
                yield (audio_data, timestamp)
                speech_buffer = []


def save_audio_chunk(audio_data, filepath, sample_rate=None):
    """
    Save audio chunk to WAV file.
    
    Args:
        audio_data: numpy array of audio data
        filepath: Path to save WAV file
        sample_rate: Sample rate (uses AUDIO_SAMPLE_RATE if None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import wave
        
        if sample_rate is None:
            sample_rate = AUDIO_SAMPLE_RATE
        
        # Ensure audio_data is int16
        if audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return True
    
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False


def get_audio_info(audio_data, sample_rate=None):
    """
    Get information about audio data.
    
    Args:
        audio_data: numpy array of audio data
        sample_rate: Sample rate (uses AUDIO_SAMPLE_RATE if None)
    
    Returns:
        dict with audio information
    """
    if audio_data is None:
        return None
    
    if sample_rate is None:
        sample_rate = AUDIO_SAMPLE_RATE
    
    duration = len(audio_data) / sample_rate
    
    return {
        "samples": len(audio_data),
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "dtype": str(audio_data.dtype),
        "rms_amplitude": float(np.sqrt(np.mean(audio_data**2))),
        "max_amplitude": int(np.max(np.abs(audio_data)))
    }


# Test function
def test_capture():
    """Test audio capture functionality."""
    print("Testing audio capture...")
    print("\nAvailable audio devices:")
    devices = list_audio_devices()
    for device in devices:
        print(f"  [{device['index']}] {device['name']} - {device['sample_rate']}Hz")
    
    print(f"\nCreating audio stream (sample_rate={AUDIO_SAMPLE_RATE}, chunk_size={AUDIO_CHUNK_SIZE})...")
    p, stream = create_audio_stream()
    
    if stream is None:
        print("Failed to create audio stream")
        return False
    
    print("Audio stream created successfully!")
    print("Recording 3 seconds of audio...")
    print("(Please speak into your microphone)")
    
    # Capture 3 seconds
    audio_data = capture_audio_chunk(stream, duration_seconds=3.0)
    
    if audio_data is not None:
        info = get_audio_info(audio_data)
        print(f"\nAudio captured successfully: {info}")
        
        # Check if silence
        is_silent = is_silence(audio_data)
        print(f"Is silence: {is_silent}")
        
        # Save test audio
        save_audio_chunk(audio_data, "test_audio.wav")
        print("Test audio saved as 'test_audio.wav'")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
    else:
        print("Failed to capture audio")
        stream.stop_stream()
        stream.close()
        p.terminate()
        return False


if __name__ == "__main__":
    test_capture()
