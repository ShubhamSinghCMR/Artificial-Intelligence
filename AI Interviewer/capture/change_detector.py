"""
Change detection module for comparing screen frames.
Detects significant changes to avoid unnecessary OCR processing.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from config.settings import CHANGE_DETECTION_THRESHOLD


def calculate_frame_hash(frame):
    """
    Calculate a simple hash of frame for fast comparison.
    
    Args:
        frame: numpy array (BGR format)
    
    Returns:
        int: Hash value
    """
    if frame is None:
        return 0
    
    # Resize to small size for faster hashing
    small = cv2.resize(frame, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Calculate hash (simple average)
    return int(np.mean(gray))


def calculate_pixel_diff(frame1, frame2):
    """
    Calculate pixel difference between two frames.
    
    Args:
        frame1: numpy array (BGR format)
        frame2: numpy array (BGR format)
    
    Returns:
        float: Difference ratio (0.0-1.0, higher = more different)
    """
    if frame1 is None or frame2 is None:
        return 1.0
    
    # Resize to same size if different
    if frame1.shape != frame2.shape:
        h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Calculate percentage of changed pixels
    total_pixels = diff.size
    changed_pixels = np.count_nonzero(diff > 30)  # Threshold for "changed" pixel
    change_ratio = changed_pixels / total_pixels
    
    return change_ratio


def calculate_histogram_diff(frame1, frame2):
    """
    Calculate histogram difference between two frames.
    More robust to small movements but detects content changes.
    
    Args:
        frame1: numpy array (BGR format)
        frame2: numpy array (BGR format)
    
    Returns:
        float: Difference ratio (0.0-1.0, higher = more different)
    """
    if frame1 is None or frame2 is None:
        return 1.0
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Compare histograms using correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Convert to difference ratio (1.0 - correlation)
    diff_ratio = 1.0 - correlation
    
    return diff_ratio


def calculate_structural_similarity(frame1, frame2):
    """
    Calculate Structural Similarity Index (SSIM) between two frames.
    More sophisticated method that considers structural information.
    
    Args:
        frame1: numpy array (BGR format)
        frame2: numpy array (BGR format)
    
    Returns:
        float: Similarity score (0.0-1.0, higher = more similar)
    """
    if frame1 is None or frame2 is None:
        return 0.0
    
    # Resize to same size if different
    if frame1.shape != frame2.shape:
        h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    
    # Resize to smaller size for faster computation
    gray1 = cv2.resize(gray1, (256, 256))
    gray2 = cv2.resize(gray2, (256, 256))
    
    # Simple SSIM calculation (simplified version)
    # Full SSIM would require scikit-image, but we'll use a simpler approach
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return max(0.0, min(1.0, ssim))


def has_changed(frame1, frame2, threshold=None, method="pixel_diff"):
    """
    Check if there's a significant change between two frames.
    
    Args:
        frame1: numpy array (BGR format) - previous frame
        frame2: numpy array (BGR format) - current frame
        threshold: Change threshold (uses CHANGE_DETECTION_THRESHOLD if None)
        method: Comparison method - "hash", "pixel_diff", "histogram", or "ssim"
    
    Returns:
        bool: True if significant change detected, False otherwise
    """
    if threshold is None:
        threshold = CHANGE_DETECTION_THRESHOLD
    
    if frame1 is None or frame2 is None:
        return True  # Consider it changed if one frame is missing
    
    if method == "hash":
        # Fast hash comparison
        hash1 = calculate_frame_hash(frame1)
        hash2 = calculate_frame_hash(frame2)
        return hash1 != hash2
    
    elif method == "pixel_diff":
        # Pixel difference method
        diff_ratio = calculate_pixel_diff(frame1, frame2)
        return diff_ratio > threshold
    
    elif method == "histogram":
        # Histogram comparison
        diff_ratio = calculate_histogram_diff(frame1, frame2)
        return diff_ratio > threshold
    
    elif method == "ssim":
        # Structural similarity (inverted - lower similarity = more change)
        similarity = calculate_structural_similarity(frame1, frame2)
        diff_ratio = 1.0 - similarity
        return diff_ratio > threshold
    
    else:
        # Default to pixel_diff
        diff_ratio = calculate_pixel_diff(frame1, frame2)
        return diff_ratio > threshold


def get_change_score(frame1, frame2, method="pixel_diff"):
    """
    Get a change score between two frames (0.0 = identical, 1.0 = completely different).
    
    Args:
        frame1: numpy array (BGR format) - previous frame
        frame2: numpy array (BGR format) - current frame
        method: Comparison method - "pixel_diff", "histogram", or "ssim"
    
    Returns:
        float: Change score (0.0-1.0)
    """
    if frame1 is None or frame2 is None:
        return 1.0
    
    if method == "pixel_diff":
        return calculate_pixel_diff(frame1, frame2)
    
    elif method == "histogram":
        return calculate_histogram_diff(frame1, frame2)
    
    elif method == "ssim":
        similarity = calculate_structural_similarity(frame1, frame2)
        return 1.0 - similarity
    
    else:
        return calculate_pixel_diff(frame1, frame2)


class FrameChangeDetector:
    """
    Simple stateful change detector that remembers the last frame.
    """
    
    def __init__(self, threshold=None, method="pixel_diff"):
        """
        Initialize change detector.
        
        Args:
            threshold: Change threshold (uses CHANGE_DETECTION_THRESHOLD if None)
            method: Comparison method
        """
        self.last_frame = None
        self.threshold = threshold if threshold is not None else CHANGE_DETECTION_THRESHOLD
        self.method = method
    
    def update(self, frame):
        """
        Update with new frame and check if changed.
        
        Args:
            frame: numpy array (BGR format)
        
        Returns:
            bool: True if changed, False otherwise
        """
        if self.last_frame is None:
            self.last_frame = frame
            return True  # First frame is always considered "changed"
        
        changed = has_changed(self.last_frame, frame, self.threshold, self.method)
        
        if changed:
            self.last_frame = frame
        
        return changed
    
    def get_score(self, frame):
        """
        Get change score without updating last frame.
        
        Args:
            frame: numpy array (BGR format)
        
        Returns:
            float: Change score (0.0-1.0)
        """
        if self.last_frame is None:
            return 1.0
        
        return get_change_score(self.last_frame, frame, self.method)
    
    def reset(self):
        """Reset the detector (forget last frame)."""
        self.last_frame = None


# Test function
def test_change_detector():
    """Test change detection functionality."""
    print("Testing change detector...")
    
    from capture.screen_capture import capture_frame
    
    # Capture two frames
    print("Capturing first frame...")
    frame1 = capture_frame()
    
    if frame1 is None:
        print("Failed to capture frame")
        return False
    
    print("Frame 1 captured. Please change something on screen...")
    import time
    time.sleep(3)
    
    print("Capturing second frame...")
    frame2 = capture_frame()
    
    if frame2 is None:
        print("Failed to capture frame")
        return False
    
    # Test different methods
    print("\nTesting different change detection methods:")
    
    methods = ["hash", "pixel_diff", "histogram", "ssim"]
    for method in methods:
        changed = has_changed(frame1, frame2, method=method)
        score = get_change_score(frame1, frame2, method=method)
        print(f"  {method:15s}: changed={changed}, score={score:.4f}")
    
    # Test stateful detector
    print("\nTesting stateful detector:")
    detector = FrameChangeDetector(method="pixel_diff")
    
    changed1 = detector.update(frame1)
    print(f"  First frame: changed={changed1}")
    
    changed2 = detector.update(frame1)  # Same frame
    print(f"  Same frame: changed={changed2}")
    
    changed3 = detector.update(frame2)  # Different frame
    print(f"  Different frame: changed={changed3}")
    
    return True


if __name__ == "__main__":
    test_change_detector()
