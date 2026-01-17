"""
Interview engine - main orchestration module.
Coordinates capture, perception, fusion, and question generation.
"""

import sys
import os
from pathlib import Path
import time
import threading
from queue import Queue

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from context.state import create_state, add_context_entry, end_session, get_session_stats
from context.fusion import fuse_ocr_and_transcript, get_context_for_llm
from capture.screen_capture import capture_frame, FrameChangeDetector
from capture.audio_capture import create_audio_stream, capture_audio_chunk, audio_chunk_generator
from perception.ocr import extract_text
from perception.stt import transcribe_audio
from perception.content_parser import parse_ocr_content, parse_transcript_content
from interview.trigger_logic import should_ask_initial_question, should_ask_question, should_ask_followup
from interview.question_generator import QuestionGenerator
from config.settings import (
    OCR_PROCESS_INTERVAL,
    STT_CHUNK_DURATION,
    MIN_QUESTION_INTERVAL
)


class InterviewEngine:
    """
    Main interview engine that orchestrates all components.
    """
    
    def __init__(self, state=None):
        """
        Initialize interview engine.
        
        Args:
            state: State dictionary (creates new if None)
        """
        self.state = state if state else create_state()
        self.is_running = False
        self.question_generator = QuestionGenerator(self.state)
        
        # Capture components
        self.audio_stream = None
        self.audio_p = None
        self.change_detector = FrameChangeDetector()
        
        # Processing queues
        self.frame_queue = Queue()
        self.audio_queue = Queue()
        
        # Threads
        self.capture_thread = None
        self.processing_thread = None
        
        # Timing
        self.last_ocr_time = 0
        self.last_stt_time = 0
    
    def start(self):
        """Start the interview session."""
        print("="*60)
        print("Starting AI Interviewer")
        print("="*60)
        
        # Initialize audio capture
        print("\nInitializing audio capture...")
        self.audio_p, self.audio_stream = create_audio_stream()
        if self.audio_stream is None:
            print("ERROR: Failed to initialize audio capture")
            return False
        
        print("✓ Audio capture initialized")
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("\n✓ Interview engine started")
        print("Press Ctrl+C to stop\n")
        
        return True
    
    def stop(self):
        """Stop the interview session."""
        print("\n" + "="*60)
        print("Stopping AI Interviewer")
        print("="*60)
        
        self.is_running = False
        
        # Stop audio
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_p:
            self.audio_p.terminate()
        
        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # End session
        end_session(self.state)
        
        # Print stats
        stats = get_session_stats(self.state)
        print("\nSession Statistics:")
        print(f"  Duration: {stats['duration_minutes']:.1f} minutes")
        print(f"  Questions asked: {stats['question_count']}")
        print(f"  Answers received: {stats['answer_count']}")
        print(f"  Topics discussed: {stats['topics_discussed']}")
        
        print("\n✓ Interview session ended")
    
    def _capture_loop(self):
        """Main capture loop (runs in separate thread)."""
        while self.is_running:
            try:
                # Capture screen frame
                frame = capture_frame()
                if frame is not None:
                    # Check for changes
                    if self.change_detector.update(frame):
                        self.frame_queue.put((frame, time.time()))
                
                # Capture audio chunk
                if self.audio_stream:
                    audio_data = capture_audio_chunk(self.audio_stream, duration_seconds=STT_CHUNK_DURATION)
                    if audio_data is not None:
                        self.audio_queue.put((audio_data, time.time()))
                
                # Small sleep to prevent CPU overload
                time.sleep(0.1)
            
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(1.0)
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Process frames (OCR)
                if not self.frame_queue.empty():
                    frame, frame_time = self.frame_queue.get_nowait()
                    
                    # Only process if enough time has passed
                    if current_time - self.last_ocr_time >= OCR_PROCESS_INTERVAL:
                        self._process_frame(frame, frame_time)
                        self.last_ocr_time = current_time
                
                # Process audio (STT)
                if not self.audio_queue.empty():
                    audio_data, audio_time = self.audio_queue.get_nowait()
                    
                    # Only process if enough time has passed
                    if current_time - self.last_stt_time >= STT_CHUNK_DURATION:
                        self._process_audio(audio_data, audio_time)
                        self.last_stt_time = current_time
                
                # Check for questions
                self._check_and_ask_questions()
                
                # Small sleep
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def _process_frame(self, frame, timestamp):
        """Process a screen frame with OCR."""
        try:
            # Extract text
            ocr_result = extract_text(frame, preprocess=True)
            
            if ocr_result and ocr_result.get('text'):
                # Parse content
                ocr_content = parse_ocr_content(ocr_result)
                
                # Fuse with transcript if available
                latest_transcript = self.state.get('latest_transcript')
                if latest_transcript:
                    fused = fuse_ocr_and_transcript(ocr_result, None, self.state)
                else:
                    fused = fuse_ocr_and_transcript(ocr_result, None, self.state)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def _process_audio(self, audio_data, timestamp):
        """Process audio chunk with STT."""
        try:
            # Transcribe
            transcript_result = transcribe_audio(audio_data)
            
            if transcript_result and transcript_result.get('text'):
                # Parse content
                transcript_content = parse_transcript_content(transcript_result)
                
                # Fuse with OCR if available
                latest_ocr = self.state.get('latest_screen_content')
                if latest_ocr:
                    ocr_result = {'text': latest_ocr.get('text', ''), 'confidence': latest_ocr.get('confidence', 0)}
                    fused = fuse_ocr_and_transcript(ocr_result, transcript_result, self.state)
                else:
                    fused = fuse_ocr_and_transcript(None, transcript_result, self.state)
                
                # Print transcript (for debugging)
                text = transcript_result.get('text', '').strip()
                if text:
                    print(f"\n[STUDENT]: {text}")
        
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def _check_and_ask_questions(self):
        """Check triggers and ask questions if appropriate."""
        try:
            current_time = time.time()
            
            # Get recent context
            from context.state import get_recent_context
            recent_context = get_recent_context(self.state, seconds=60)
            
            # Check for initial question
            if self.state.get('question_count', 0) == 0:
                if should_ask_initial_question(self.state, current_time):
                    self._ask_initial_question()
                return
            
            # Check for regular question
            trigger = should_ask_question(self.state, current_time, recent_context)
            if trigger['should_ask']:
                self._ask_question(trigger['trigger_type'])
            
            # Check for follow-up
            followup = should_ask_followup(self.state)
            if followup['should_ask']:
                self._ask_followup_question(followup['question_id'])
        
        except Exception as e:
            print(f"Error checking questions: {e}")
    
    def _ask_initial_question(self):
        """Ask the initial question."""
        try:
            context = get_context_for_llm(self.state, seconds=60)
            question_info = self.question_generator.generate_initial_question(context)
            
            if question_info and question_info.get('question'):
                question = question_info['question']
                print(f"\n{'='*60}")
                print(f"[INTERVIEWER]: {question}")
                print(f"{'='*60}\n")
                
                # Add to state
                add_context_entry(self.state, 'question', question, {
                    'question_id': self.state.get('question_count', 0),
                    'is_initial': True
                })
        
        except Exception as e:
            print(f"Error asking initial question: {e}")
    
    def _ask_question(self, trigger_type=None):
        """Ask a question based on trigger."""
        try:
            context = get_context_for_llm(self.state, seconds=60)
            question_info = self.question_generator.generate_question_from_context(
                context=context,
                trigger_type=trigger_type
            )
            
            if question_info and question_info.get('question'):
                question = question_info['question']
                print(f"\n{'='*60}")
                print(f"[INTERVIEWER]: {question}")
                print(f"  (Trigger: {trigger_type or 'time-based'})")
                print(f"{'='*60}\n")
                
                # Add to state
                add_context_entry(self.state, 'question', question, {
                    'question_id': self.state.get('question_count', 0),
                    'trigger_type': trigger_type
                })
        
        except Exception as e:
            print(f"Error asking question: {e}")
    
    def _ask_followup_question(self, question_id):
        """Ask a follow-up question."""
        try:
            question_info = self.question_generator.generate_followup_question(question_id)
            
            if question_info and question_info.get('question'):
                question = question_info['question']
                print(f"\n{'='*60}")
                print(f"[INTERVIEWER] (Follow-up): {question}")
                print(f"{'='*60}\n")
                
                # Add to state
                add_context_entry(self.state, 'question', question, {
                    'question_id': self.state.get('question_count', 0),
                    'parent_question_id': question_id,
                    'is_followup': True
                })
        
        except Exception as e:
            print(f"Error asking follow-up: {e}")
    
    def run(self):
        """Run the interview (blocking)."""
        if not self.start():
            return
        
        try:
            # Main loop
            while self.is_running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self.stop()


def run_interview(state=None):
    """
    Run interview session (convenience function).
    
    Args:
        state: Optional state dictionary
    """
    engine = InterviewEngine(state)
    engine.run()


# Test function
def test_engine():
    """Test interview engine (short test)."""
    print("Testing interview engine...")
    print("This will run for 10 seconds as a test.")
    
    engine = InterviewEngine()
    
    if not engine.start():
        print("Failed to start engine")
        return False
    
    try:
        # Run for 10 seconds
        time.sleep(10)
    finally:
        engine.stop()
    
    return True


if __name__ == "__main__":
    # Run full interview
    run_interview()
