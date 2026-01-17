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
from capture.screen_capture import capture_frame
from capture.change_detector import FrameChangeDetector
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
from config.logging_config import get_logger

logger = get_logger('interview.engine')


class InterviewEngine:
    """
    Main interview engine that orchestrates all components.
    Implements turn-based interview flow:
    1. LISTENING: User describes project, system listens and stores
    2. WAITING_FOR_ANSWER: Question asked, waiting for user to answer
    3. PROCEEDING: Answer received, saying "proceed"
    """
    
    # Interview states
    STATE_LISTENING = "listening"  # User is describing, system listening
    STATE_WAITING_FOR_ANSWER = "waiting_for_answer"  # Question asked, waiting for answer
    STATE_PROCEEDING = "proceeding"  # Answer received, saying proceed
    STATE_ENDED = "ended"  # Interview ended
    
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
        
        # Turn-based interview state
        self.interview_state = self.STATE_LISTENING
        self.current_question_id = None
        
        # Speech accumulation
        self.speech_buffer = []  # Accumulate speech while listening
        self.answer_buffer = []  # Accumulate speech for answer
        self.last_speech_time = None  # Track when user last spoke
        self.silence_start_time = None  # Track when silence started
        self.question_asked_flag = False  # Prevent asking multiple questions
        self.consecutive_empty_chunks = 0  # Track consecutive empty STT chunks
        
        # Settings
        self.silence_duration_for_question = 10.0  # Seconds of silence before asking question
        self.silence_duration_for_answer = 10.0  # Seconds of silence before considering answer complete
        self.min_speech_length = 10  # Minimum characters before asking question
    
    def start(self):
        """Start the interview session."""
        # Initialize audio capture
        self.audio_p, self.audio_stream = create_audio_stream()
        if self.audio_stream is None:
            logger.error("Failed to initialize audio capture")
            return False
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("\n" + "="*60)
        print("DEMONSTRATION")
        print("="*60 + "\n")
        print("[INTERVIEWER]: Please begin speaking. Describe your project.\n")
        
        return True
    
    def stop(self):
        """Stop the interview session."""
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
                logger.error(f"Error in capture loop: {e}", exc_info=True)
                time.sleep(1.0)
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)."""
        while self.is_running and self.interview_state != self.STATE_ENDED:
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
                
                # State machine logic
                if self.interview_state == self.STATE_LISTENING:
                    self._handle_listening_state(current_time)
                elif self.interview_state == self.STATE_WAITING_FOR_ANSWER:
                    self._handle_waiting_for_answer_state(current_time)
                elif self.interview_state == self.STATE_PROCEEDING:
                    self._handle_proceeding_state()
                
                # Small sleep
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(1.0)
    
    def _process_frame(self, frame, timestamp):
        """Process a screen frame with OCR."""
        try:
            # Extract text from screen
            ocr_result = extract_text(frame, preprocess=True)
            
            if ocr_result and ocr_result.get('text'):
                # Parse content
                ocr_content = parse_ocr_content(ocr_result)
                
                # Store latest screen content in state for question generation
                self.state['latest_screen_content'] = {
                    'text': ocr_result.get('text', ''),
                    'confidence': ocr_result.get('confidence', 0),
                    'timestamp': timestamp
                }
                
                # Fuse with transcript if available
                latest_transcript = self.state.get('latest_transcript')
                if latest_transcript:
                    fused = fuse_ocr_and_transcript(ocr_result, None, self.state)
                else:
                    fused = fuse_ocr_and_transcript(ocr_result, None, self.state)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
    
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
                
                # Log transcript
                text = transcript_result.get('text', '').strip()
                if text:
                    # Print candidate transcription
                    print(f"[CANDIDATE]: {text}")
                    
                    # Check for "thank you" to end interview
                    text_lower = text.lower()
                    if any(phrase in text_lower for phrase in ['thank you', 'thanks', 'thankyou', 'thank u']):
                        print("\n[INTERVIEWER]: Thank you! The interview is now complete.\n")
                        self.interview_state = self.STATE_ENDED
                        self.is_running = False
                        return
                    
                    # Update last speech time (use current processing time, not audio timestamp)
                    self.last_speech_time = time.time()  # Use current time, not audio timestamp
                    self.silence_start_time = None
                    self.question_asked_flag = False  # Reset flag when new speech detected
                    self.consecutive_empty_chunks = 0  # Reset empty chunk counter
                    
                    # Accumulate speech based on state
                    if self.interview_state == self.STATE_LISTENING:
                        self.speech_buffer.append(text)
                    elif self.interview_state == self.STATE_WAITING_FOR_ANSWER:
                        self.answer_buffer.append(text)
                else:
                    # No speech in this chunk - increment counter
                    self.consecutive_empty_chunks += 1
                    
                    # If we've had multiple empty chunks, update silence start time
                    if self.consecutive_empty_chunks >= 2:  # 2 chunks = ~6 seconds of no speech
                        if self.silence_start_time is None and self.last_speech_time is not None:
                            self.silence_start_time = time.time()
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
    
    def _handle_listening_state(self, current_time):
        """Handle LISTENING state: accumulate speech, detect silence, ask question."""
        try:
            # Only check for silence if we have received speech and haven't asked question yet
            if self.last_speech_time is not None and not self.question_asked_flag:
                # Use silence_start_time if we've detected empty chunks, otherwise use last_speech_time
                if self.silence_start_time is not None:
                    # We've had consecutive empty chunks, use silence start time
                    time_since_silence = current_time - self.silence_start_time
                    time_since_speech = time_since_silence
                else:
                    # Calculate time since last speech was received
                    time_since_speech = current_time - self.last_speech_time
                
                # Wait for full silence duration (10 seconds)
                if time_since_speech >= self.silence_duration_for_question:
                    # Check if we have enough speech to ask a question
                    accumulated_text = " ".join(self.speech_buffer).strip()
                    
                    if len(accumulated_text) >= self.min_speech_length:
                        # Set flag to prevent asking multiple questions
                        self.question_asked_flag = True
                        
                        # Print candidate transcription
                        print(f"\n[CANDIDATE TRANSCRIPTION]: {accumulated_text}\n")
                        print("Candidate stopped.\n")
                        
                        # Ask question based on accumulated speech and screen content
                        self._ask_question_from_speech(accumulated_text)
                        
                        # Clear speech buffer
                        self.speech_buffer = []
                        self.last_speech_time = None
                        self.silence_start_time = None
                        self.consecutive_empty_chunks = 0
                        
                        # Switch to waiting for answer
                        self.interview_state = self.STATE_WAITING_FOR_ANSWER
                        self.answer_buffer = []
            elif self.last_speech_time is None:
                # No speech yet, initialize silence tracking
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
        
        except Exception as e:
            logger.error(f"Error in listening state: {e}", exc_info=True)
    
    def _handle_waiting_for_answer_state(self, current_time):
        """Handle WAITING_FOR_ANSWER state: wait for answer, detect completion."""
        try:
            # Check if user has stopped speaking (answer complete)
            if self.last_speech_time is not None:
                # Calculate time since last speech
                time_since_speech = current_time - self.last_speech_time
                
                # If silence detected for long enough (10 seconds), answer is complete
                if time_since_speech >= self.silence_duration_for_answer:
                    # Process the answer
                    answer_text = " ".join(self.answer_buffer).strip()
                    
                    if len(answer_text) >= 5:  # Minimum answer length
                        print(f"\n[CANDIDATE ANSWER]: {answer_text}\n")
                        print("Candidate stopped.\n")
                        
                        self._process_answer(answer_text)
                        
                        # Clear answer buffer
                        self.answer_buffer = []
                        self.last_speech_time = None
                        
                        # Switch to proceeding state
                        self.interview_state = self.STATE_PROCEEDING
        
        except Exception as e:
            logger.error(f"Error in waiting for answer state: {e}", exc_info=True)
    
    def _handle_proceeding_state(self):
        """Handle PROCEEDING state: say 'proceed' and return to listening."""
        try:
            print("[INTERVIEWER]: Please proceed with your presentation.\n")
            
            # Switch back to listening
            self.interview_state = self.STATE_LISTENING
            self.speech_buffer = []
            self.last_speech_time = None
            self.silence_start_time = None
            self.question_asked_flag = False  # Reset flag for next question
            self.consecutive_empty_chunks = 0  # Reset empty chunk counter
        
        except Exception as e:
            logger.error(f"Error in proceeding state: {e}", exc_info=True)
    
    def _ask_question_from_speech(self, speech_text):
        """Ask a question based on the accumulated speech and screen content."""
        try:
            # Get context including both speech and screen content (returns a string)
            context_str = get_context_for_llm(self.state, seconds=60)
            
            # Get latest screen content (OCR)
            latest_ocr = self.state.get('latest_screen_content', {})
            screen_text = ""
            if latest_ocr and isinstance(latest_ocr, dict):
                screen_text = latest_ocr.get('text', '')
            
            # Build comprehensive context with both speech and screen
            context_text = f"Student has described: {speech_text}\n\n"
            if screen_text:
                context_text += f"Screen content visible: {screen_text}\n\n"
            if context_str:
                context_text += context_str
            
            # Generate question (pass context as string)
            question_info = self.question_generator.generate_question_from_context(
                context=context_text,
                trigger_type='speech_complete'
            )
            
            if question_info and question_info.get('question'):
                question = question_info['question']
            else:
                # Fallback question
                question = "Can you tell me more about that?"
            
            # Print question
            print("Question:")
            print(f"[INTERVIEWER]: {question}\n")
            print("Now you answer.\n")
            
            # Add to state
            question_id = self.state.get('question_count', 0)
            add_context_entry(self.state, 'question', question, {
                'question_id': question_id,
                'trigger_type': 'speech_complete'
            })
            
            self.current_question_id = question_id
        
        except Exception as e:
            logger.error(f"Error asking question from speech: {e}", exc_info=True)
    
    # Old methods kept for compatibility but not used in new turn-based flow
    def _ask_initial_question(self):
        """Ask the initial question (not used in turn-based flow)."""
        pass
    
    def _ask_question(self, trigger_type=None):
        """Ask a question based on trigger (not used in turn-based flow)."""
        pass
    
    def _ask_followup_question(self, question_id):
        """Ask a follow-up question (not used in turn-based flow)."""
        pass
    
    def _check_for_answer(self, current_time):
        """Check if answer has been received."""
        try:
            # Check timeout
            if self.answer_start_time and (current_time - self.answer_start_time) > self.answer_timeout:
                logger.warning("Answer timeout. Moving to next question.")
                self.waiting_for_answer = False
                self.answer_buffer = []
                return
            
            # If we have accumulated answer text, process it
            if self.answer_buffer:
                # Wait a bit for more speech (in case student is still talking)
                time_since_last = current_time - self.last_stt_time
                
                # If no new speech for 3 seconds, consider answer complete
                if time_since_last > 3.0:
                    answer_text = " ".join(self.answer_buffer).strip()
                    
                    if len(answer_text) > 10:  # Minimum answer length
                        self._process_answer(answer_text)
                        self.answer_buffer = []
                        self.waiting_for_answer = False
        
        except Exception as e:
            print(f"Error checking for answer: {e}")
    
    def _process_answer(self, answer_text):
        """Process and store student answer."""
        try:
            from context.state import add_context_entry
            from evaluation.scorer import score_answer
            from context.fusion import get_context_for_llm
            
            # Get context for scoring
            context = get_context_for_llm(self.state, seconds=60)
            
            # Get the question
            questions = self.state.get('questions', [])
            question = None
            for q in questions:
                if q.get('question_id') == self.current_question_id:
                    question = q.get('question', '')
                    break
            
            if question:
                # Score the answer
                scores = score_answer(question, answer_text, context, use_llm=False)
                
                # Add to state
                add_context_entry(self.state, 'answer', answer_text, {
                    'question_id': self.current_question_id,
                    'scores': scores
                })
            else:
                # Add answer without scoring if question not found
                add_context_entry(self.state, 'answer', answer_text, {
                    'question_id': self.current_question_id
                })
            
            # Switch to proceeding state to say "proceed"
            self.interview_state = self.STATE_PROCEEDING
        
        except Exception as e:
            logger.error(f"Error processing answer: {e}", exc_info=True)
    
    def _capture_manual_answer(self):
        """Capture answer via manual input (fallback)."""
        try:
            print("\n[SYSTEM]: Waiting for answer...")
            print("(Type your answer and press Enter, or speak your answer)")
            print("(Press Enter with empty line to skip)\n")
            
            # Try to get input (non-blocking would be better, but this works)
            import select
            import sys
            
            # Simple input for now (can be enhanced with threading)
            answer = input("[YOUR ANSWER]: ").strip()
            
            if answer:
                self._process_answer(answer)
                self.waiting_for_answer = False
                return True
            
            return False
        
        except Exception as e:
            print(f"Error capturing manual answer: {e}")
            return False
    
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
