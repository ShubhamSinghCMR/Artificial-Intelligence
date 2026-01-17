"""
OCR (Optical Character Recognition) module for extracting text from screen frames.
Uses Tesseract OCR via pytesseract.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pytesseract
from config.settings import get_tesseract_path


# Configure pytesseract with Tesseract path
tesseract_path = get_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def preprocess_image(frame, enhance_contrast=True, denoise=True, resize_factor=None):
    """
    Preprocess image for better OCR results.
    
    Args:
        frame: numpy array (BGR format)
        enhance_contrast: Apply contrast enhancement
        denoise: Apply denoising
        resize_factor: Resize factor (None = no resize, 2.0 = double size)
    
    Returns:
        numpy array: Preprocessed grayscale image
    """
    if frame is None:
        return None
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Resize if needed (larger images often give better OCR results)
    if resize_factor and resize_factor != 1.0:
        height, width = gray.shape
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Threshold to binary (optional - can help with some images)
    # Uncomment if needed:
    # _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return gray


def extract_text(frame, preprocess=True, lang='eng', config='--psm 6'):
    """
    Extract text from image using OCR.
    
    Args:
        frame: numpy array (BGR format)
        preprocess: Apply preprocessing (recommended)
        lang: Language code (default: 'eng')
        config: Tesseract config string (default: '--psm 6' for uniform block of text)
    
    Returns:
        dict with keys:
            - 'text': Extracted text string
            - 'confidence': Average confidence (0-100)
            - 'word_count': Number of words detected
            - 'raw_data': Raw Tesseract data dict
    """
    if frame is None:
        return {
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'raw_data': None
        }
    
    try:
        # Preprocess if requested
        if preprocess:
            processed = preprocess_image(frame, enhance_contrast=True, denoise=True)
        else:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                processed = frame
        
        # Extract text with confidence data
        data = pytesseract.image_to_data(processed, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        
        # Extract text and calculate confidence
        text_parts = []
        confidences = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 0:  # Ignore empty text and negative confidence
                text_parts.append(text)
                confidences.append(conf)
        
        # Combine text
        full_text = ' '.join(text_parts)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Count words
        word_count = len(full_text.split()) if full_text else 0
        
        return {
            'text': full_text,
            'confidence': float(avg_confidence),
            'word_count': word_count,
            'raw_data': data
        }
    
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'raw_data': None
        }


def extract_text_simple(frame, preprocess=True):
    """
    Simple text extraction - returns just the text string.
    
    Args:
        frame: numpy array (BGR format)
        preprocess: Apply preprocessing
    
    Returns:
        str: Extracted text
    """
    result = extract_text(frame, preprocess=preprocess)
    return result['text']


def extract_text_with_boxes(frame, preprocess=True, lang='eng', min_confidence=30):
    """
    Extract text with bounding box information.
    
    Args:
        frame: numpy array (BGR format)
        preprocess: Apply preprocessing
        lang: Language code
        min_confidence: Minimum confidence threshold for words
    
    Returns:
        list of dicts with keys: 'text', 'confidence', 'x', 'y', 'width', 'height'
    """
    if frame is None:
        return []
    
    try:
        # Preprocess if requested
        if preprocess:
            processed = preprocess_image(frame, enhance_contrast=True, denoise=True)
        else:
            if len(frame.shape) == 3:
                processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                processed = frame
        
        # Extract data with bounding boxes
        data = pytesseract.image_to_data(processed, lang=lang, output_type=pytesseract.Output.DICT)
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf >= min_confidence:
                boxes.append({
                    'text': text,
                    'confidence': conf,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                })
        
        return boxes
    
    except Exception as e:
        print(f"Error extracting text with boxes: {e}")
        return []


def extract_code_text(frame, preprocess=True):
    """
    Extract text optimized for code snippets.
    Uses different Tesseract config for code.
    
    Args:
        frame: numpy array (BGR format)
        preprocess: Apply preprocessing
    
    Returns:
        dict with text extraction results
    """
    # PSM 6: uniform block of text (good for code)
    # Add digits config for better number recognition
    config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz{}[]()=+-*/<>.,;:!?@#$%^&_|\\/"\' \n\t'
    
    return extract_text(frame, preprocess=preprocess, config=config)


def get_ocr_languages():
    """
    Get list of available OCR languages.
    
    Returns:
        list of language codes
    """
    try:
        langs = pytesseract.get_languages()
        return langs
    except Exception as e:
        print(f"Error getting languages: {e}")
        return ['eng']  # Default to English


def test_ocr():
    """
    Test OCR functionality.
    """
    print("Testing OCR engine...")
    
    # Check Tesseract path
    tesseract_path = get_tesseract_path()
    if not tesseract_path:
        print("\n" + "="*60)
        print("ERROR: Tesseract OCR not found!")
        print("="*60)
        print("\nTo install Tesseract OCR on Windows:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   (Direct link: https://digi.bib.uni-mannheim.de/tesseract/")
        print("2. Run the installer (tesseract-ocr-w64-setup-*.exe)")
        print("3. During installation, note the installation path")
        print("   (Default: C:\\Program Files\\Tesseract-OCR\\)")
        print("4. After installation, the path should be auto-detected")
        print("\nAlternatively, you can manually set the path in config/settings.py")
        print("by modifying the get_tesseract_path() function.")
        print("\nQuick test: After installation, restart and run this test again.")
        print("="*60 + "\n")
        return False
    
    print(f"Tesseract path: {tesseract_path}")
    
    # Check available languages
    langs = get_ocr_languages()
    print(f"Available languages: {', '.join(langs)}")
    
    # Try to capture a frame and extract text
    print("\nCapturing screen frame for OCR test...")
    from capture.screen_capture import capture_frame
    
    frame = capture_frame()
    if frame is None:
        print("Failed to capture frame")
        return False
    
    print("Frame captured. Extracting text...")
    
    # Test basic extraction
    result = extract_text(frame, preprocess=True)
    print(f"\nExtraction Results:")
    print(f"  Text length: {len(result['text'])} characters")
    print(f"  Word count: {result['word_count']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"\nExtracted text (first 500 chars):")
    print(result['text'][:500])
    
    # Test with boxes
    boxes = extract_text_with_boxes(frame, min_confidence=30)
    print(f"\nDetected {len(boxes)} text boxes (confidence >= 30%)")
    
    if boxes:
        print("Sample boxes (first 5):")
        for i, box in enumerate(boxes[:5]):
            print(f"  [{i+1}] '{box['text']}' - conf: {box['confidence']:.1f}%")
    
    return True


if __name__ == "__main__":
    test_ocr()
