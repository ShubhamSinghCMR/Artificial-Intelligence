"""
AI Interviewer - Main Entry Point
Automated interviewer for project presentations.
"""

import sys
import os
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from context.state import create_state, get_session_stats
from interview.engine import InterviewEngine
from evaluation.scorer import get_final_score
from evaluation.feedback import generate_final_feedback, format_feedback_text
from config.settings import (
    validate_settings,
    SESSION_LOG_PATH,
    LOG_TO_FILE
)
from config.logging_config import setup_logging, get_logger

# Set up logging
logger = setup_logging()


def print_banner():
    """Print startup banner."""
    logger.info("="*60)
    logger.info("AI-Driven Automated Interviewer")
    logger.info("For Project Presentations")
    logger.info("="*60)


def validate_environment():
    """Validate that all required components are available."""
    logger.info("Validating environment...")
    
    errors = validate_settings()
    
    # Check critical components
    try:
        import cv2
        logger.info("  ✓ OpenCV available")
    except ImportError:
        errors.append("OpenCV not installed: pip install opencv-python")
        logger.warning("OpenCV not available")
    
    try:
        import pytesseract
        logger.info("  ✓ Tesseract OCR available")
    except ImportError:
        errors.append("pytesseract not installed: pip install pytesseract")
        logger.warning("pytesseract not available")
    
    try:
        import pyaudio
        logger.info("  ✓ PyAudio available")
    except ImportError:
        errors.append("PyAudio not installed: pip install pyaudio")
        logger.warning("PyAudio not available")
    
    try:
        import whisper
        logger.info("  ✓ Whisper available")
    except ImportError:
        errors.append("Whisper not installed: pip install openai-whisper")
        logger.warning("Whisper not available")
    
    try:
        from llama_cpp import Llama
        logger.info("  ✓ llama-cpp-python available")
    except ImportError:
        logger.warning("llama-cpp-python not installed (LLM will use fallbacks)")
        logger.warning("Install: pip install llama-cpp-python")
        logger.warning("See INSTALL_LLM.md for details")
    
    if errors:
        logger.warning("="*60)
        logger.warning("WARNINGS/ERRORS:")
        logger.warning("="*60)
        for error in errors:
            logger.warning(f"  ⚠ {error}")
        logger.warning("="*60)
        
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            logger.info("Exiting...")
            return False
    
    logger.info("  ✓ Environment validated")
    return True


def save_session_log(state, feedback=None):
    """Save session log to file."""
    logger.info("Saving session log...")
    try:
        session_data = {
            'session_start': state.get('session_start_time'),
            'session_end': state.get('session_end_time'),
            'duration_seconds': state.get('session_end_time', 0) - state.get('session_start_time', 0),
            'stats': get_session_stats(state),
            'questions': state.get('questions', []),
            'answers': state.get('answers', []),
            'topics_discussed': state.get('topics_discussed', []),
            'keywords_mentioned': list(state.get('keywords_mentioned', set())),
        }
        
        if feedback:
            session_data['final_scores'] = {
                'overall': feedback.get('overall_score', 0),
                'technical_depth': feedback.get('detailed_scores', {}).get('technical_depth', {}).get('score', 0),
                'clarity': feedback.get('detailed_scores', {}).get('clarity', {}).get('score', 0),
                'originality': feedback.get('detailed_scores', {}).get('originality', {}).get('score', 0),
                'understanding': feedback.get('detailed_scores', {}).get('understanding', {}).get('score', 0),
                'passing': feedback.get('passing', False)
            }
            session_data['feedback'] = feedback
        
        # Save to JSON
        with open(SESSION_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session log saved to: {SESSION_LOG_PATH}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save session log: {e}")
        return False


def print_final_report(state):
    """Print final interview report."""
    logger.info("="*60)
    logger.info("FINAL INTERVIEW REPORT")
    logger.info("="*60)
    
    # Get final scores
    final_scores = get_final_score(state)
    
    print("SCORES:")
    print("-"*60)
    print(f"  Technical Depth: {final_scores['technical_depth']:.1f}/10.0")
    print(f"  Clarity:         {final_scores['clarity']:.1f}/10.0")
    print(f"  Originality:     {final_scores['originality']:.1f}/10.0")
    print(f"  Understanding:   {final_scores['understanding']:.1f}/10.0")
    print(f"  Overall Score:   {final_scores['overall']:.1f}/10.0")
    print(f"  Status:          {'PASS' if final_scores['passing'] else 'NEEDS IMPROVEMENT'}")
    print("")
    
    # Get detailed feedback
    feedback = generate_final_feedback(state)
    feedback_text = format_feedback_text(feedback)
    
    print(feedback_text)
    
    # Save session log
    if LOG_TO_FILE:
        save_session_log(state, feedback)
    
    return feedback


def main():
    """Main entry point."""
    print_banner()
    
    # Validate environment
    if not validate_environment():
        return
    
    # Create state
    logger.info("Initializing interview session...")
    state = create_state()
    
    # Create and run engine
    try:
        engine = InterviewEngine(state)
        
        logger.info("="*60)
        logger.info("INTERVIEW SESSION STARTING")
        logger.info("="*60)
        logger.info("Instructions:")
        logger.info("  - Start presenting your project")
        logger.info("  - The system will capture screen and audio")
        logger.info("  - Questions will be asked automatically")
        logger.info("  - Answer the questions when asked")
        logger.info("  - Press Ctrl+C to end the session")
        logger.info("="*60)
        
        # Run interview
        engine.run()
        
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
    finally:
        # Generate final report
        if state.get('question_count', 0) > 0:
            print_final_report(state)
        else:
            logger.info("Session ended without questions asked.")
            if LOG_TO_FILE:
                save_session_log(state)
        
        logger.info("="*60)
        logger.info("Thank you for using AI Interviewer!")
        logger.info("="*60)


if __name__ == "__main__":
    main()
