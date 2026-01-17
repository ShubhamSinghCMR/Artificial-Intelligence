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


def print_banner():
    """Print startup banner."""
    print("\n" + "="*60)
    print("AI-Driven Automated Interviewer")
    print("For Project Presentations")
    print("="*60 + "\n")


def validate_environment():
    """Validate that all required components are available."""
    print("Validating environment...")
    
    errors = validate_settings()
    
    # Check critical components
    try:
        import cv2
        print("  ✓ OpenCV available")
    except ImportError:
        errors.append("OpenCV not installed: pip install opencv-python")
    
    try:
        import pytesseract
        print("  ✓ Tesseract OCR available")
    except ImportError:
        errors.append("pytesseract not installed: pip install pytesseract")
    
    try:
        import pyaudio
        print("  ✓ PyAudio available")
    except ImportError:
        errors.append("PyAudio not installed: pip install pyaudio")
    
    try:
        import whisper
        print("  ✓ Whisper available")
    except ImportError:
        errors.append("Whisper not installed: pip install openai-whisper")
    
    try:
        from llama_cpp import Llama
        print("  ✓ llama-cpp-python available")
    except ImportError:
        print("  ⚠ llama-cpp-python not installed (LLM will use fallbacks)")
        print("    Install: pip install llama-cpp-python")
        print("    See INSTALL_LLM.md for details")
    
    if errors:
        print("\n" + "="*60)
        print("WARNINGS/ERRORS:")
        print("="*60)
        for error in errors:
            print(f"  ⚠ {error}")
        print("="*60 + "\n")
        
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Exiting...")
            return False
    
    print("  ✓ Environment validated\n")
    return True


def save_session_log(state, feedback=None):
    """Save session log to file."""
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
        
        print(f"  ✓ Session log saved to: {SESSION_LOG_PATH}")
        return True
    
    except Exception as e:
        print(f"  ⚠ Failed to save session log: {e}")
        return False


def print_final_report(state):
    """Print final interview report."""
    print("\n" + "="*60)
    print("FINAL INTERVIEW REPORT")
    print("="*60 + "\n")
    
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
    print("Initializing interview session...")
    state = create_state()
    
    # Create and run engine
    try:
        engine = InterviewEngine(state)
        
        print("="*60)
        print("INTERVIEW SESSION STARTING")
        print("="*60)
        print("\nInstructions:")
        print("  - Start presenting your project")
        print("  - The system will capture screen and audio")
        print("  - Questions will be asked automatically")
        print("  - Answer the questions when asked")
        print("  - Press Ctrl+C to end the session")
        print("\n" + "="*60 + "\n")
        
        # Run interview
        engine.run()
        
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Generate final report
        if state.get('question_count', 0) > 0:
            print_final_report(state)
        else:
            print("\nSession ended without questions asked.")
            if LOG_TO_FILE:
                save_session_log(state)
        
        print("\n" + "="*60)
        print("Thank you for using AI Interviewer!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
