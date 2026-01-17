"""
Screen capture module for capturing screen frames.
Uses mss for efficient screen capture and OpenCV for image processing.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import cv2
from mss import mss
from config.settings import (
    SCREEN_CAPTURE_FPS,
    SCREEN_CAPTURE_REGION,
    SCREEN_CAPTURE_MONITOR,
    FRAME_SAMPLE_INTERVAL
)


def get_monitor_bounds(monitor_index=0):
    """
    Get screen bounds for specified monitor.
    
    Args:
        monitor_index: Monitor index (0 = primary)
    
    Returns:
        dict with 'top', 'left', 'width', 'height' keys
    """
    sct = mss()
    monitors = sct.monitors
    
    if monitor_index >= len(monitors):
        monitor_index = 0
    
    return monitors[monitor_index]


def capture_frame(region=None, monitor_index=0):
    """
    Capture a single screen frame.
    
    Args:
        region: Custom region (x, y, width, height) or None for full screen
        monitor_index: Monitor index (0 = primary)
    
    Returns:
        numpy array (BGR format, compatible with OpenCV) or None if failed
    """
    try:
        sct = mss()
        
        # Determine capture region
        if region is not None:
            # Custom region: (x, y, width, height)
            x, y, width, height = region
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
        else:
            # Use specified monitor
            monitor = get_monitor_bounds(monitor_index)
        
        # Capture screen
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array (BGRA format)
        img_array = np.array(screenshot)
        
        # Convert BGRA to BGR (remove alpha channel)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        
        return img_bgr
    
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None


def capture_frame_generator(region=None, monitor_index=0, fps=None, max_frames=None):
    """
    Generator that yields screen frames at specified FPS.
    
    Args:
        region: Custom region (x, y, width, height) or None for full screen
        monitor_index: Monitor index (0 = primary)
        fps: Frames per second (uses SCREEN_CAPTURE_FPS if None)
        max_frames: Maximum frames to capture (None = infinite)
    
    Yields:
        tuple: (frame, timestamp) where frame is numpy array, timestamp is float
    """
    if fps is None:
        fps = SCREEN_CAPTURE_FPS
    
    frame_interval = 1.0 / fps
    frame_count = 0
    
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        
        start_time = time.time()
        
        frame = capture_frame(region, monitor_index)
        if frame is not None:
            timestamp = time.time()
            yield (frame, timestamp)
            frame_count += 1
        
        # Sleep to maintain FPS
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


def sample_frames(frames, interval_seconds=None):
    """
    Sample frames at specified interval (for change detection).
    
    Args:
        frames: List of (frame, timestamp) tuples
        interval_seconds: Sampling interval (uses FRAME_SAMPLE_INTERVAL if None)
    
    Returns:
        List of sampled (frame, timestamp) tuples
    """
    if interval_seconds is None:
        interval_seconds = FRAME_SAMPLE_INTERVAL
    
    if not frames:
        return []
    
    sampled = []
    last_sample_time = frames[0][1] if frames else 0
    
    for frame, timestamp in frames:
        if timestamp - last_sample_time >= interval_seconds:
            sampled.append((frame, timestamp))
            last_sample_time = timestamp
    
    return sampled


def save_frame(frame, filepath):
    """
    Save a frame to disk.
    
    Args:
        frame: numpy array (BGR format)
        filepath: Path to save the frame
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cv2.imwrite(filepath, frame)
        return True
    except Exception as e:
        print(f"Error saving frame: {e}")
        return False


def get_frame_info(frame):
    """
    Get information about a frame.
    
    Args:
        frame: numpy array (BGR format)
    
    Returns:
        dict with frame information (width, height, channels, dtype)
    """
    if frame is None:
        return None
    
    height, width = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) > 2 else 1
    
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(frame.dtype),
        "size_bytes": frame.nbytes
    }


# Test function
def test_capture():
    """Test screen capture functionality."""
    print("Testing screen capture...")
    
    # Capture a single frame
    frame = capture_frame()
    if frame is not None:
        info = get_frame_info(frame)
        print(f"Frame captured successfully: {info}")
        
        # Save test frame
        save_frame(frame, "test_frame.png")
        print("Test frame saved as 'test_frame.png'")
        return True
    else:
        print("Failed to capture frame")
        return False


if __name__ == "__main__":
    test_capture()
