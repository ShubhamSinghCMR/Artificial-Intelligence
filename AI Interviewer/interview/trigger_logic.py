"""
Trigger logic module for determining when to ask questions.
Detects natural pause points, topic changes, and optimal timing.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    MIN_QUESTION_INTERVAL,
    INITIAL_QUESTION_DELAY,
    QUESTION_TIMEOUT,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION
)


def should_ask_initial_question(state, current_time=None):
    """
    Check if it's time to ask the first question.
    
    Args:
        state: State dictionary
        current_time: Current timestamp (uses time.time() if None)
    
    Returns:
        bool: True if should ask initial question
    """
    if current_time is None:
        current_time = time.time()
    
    # Check if already asked questions
    if state.get('question_count', 0) > 0:
        return False
    
    # Check if session just started
    session_start = state.get('session_start_time', current_time)
    elapsed = current_time - session_start
    
    # Wait for initial delay
    return elapsed >= INITIAL_QUESTION_DELAY


def should_ask_question(state, current_time=None, recent_context=None):
    """
    Check if it's time to ask a new question.
    
    Args:
        state: State dictionary
        current_time: Current timestamp
        recent_context: Recent context entries (optional, for topic change detection)
    
    Returns:
        dict with keys: 'should_ask', 'reason', 'trigger_type'
    """
    if current_time is None:
        current_time = time.time()
    
    # Check if max questions reached
    if state.get('question_count', 0) >= state.get('max_questions', 10):
        return {
            'should_ask': False,
            'reason': 'Maximum questions reached',
            'trigger_type': None
        }
    
    # Check minimum interval
    last_question_time = state.get('last_question_time', 0)
    time_since_last = current_time - last_question_time
    
    if time_since_last < MIN_QUESTION_INTERVAL:
        return {
            'should_ask': False,
            'reason': f'Too soon since last question ({time_since_last:.1f}s < {MIN_QUESTION_INTERVAL}s)',
            'trigger_type': None
        }
    
    # Check if waiting for answer
    last_answer_time = state.get('last_answer_time', 0)
    last_question_time = state.get('last_question_time', 0)
    
    # If question was asked but no answer yet, wait
    if last_question_time > last_answer_time:
        time_waiting = current_time - last_question_time
        if time_waiting < QUESTION_TIMEOUT:
            return {
                'should_ask': False,
                'reason': f'Waiting for answer ({time_waiting:.1f}s < {QUESTION_TIMEOUT}s)',
                'trigger_type': None
            }
    
    # Check for natural pause (silence detection)
    if recent_context:
        pause_detected = detect_natural_pause(recent_context)
        if pause_detected:
            return {
                'should_ask': True,
                'reason': 'Natural pause detected',
                'trigger_type': 'pause'
            }
    
    # Check for topic change
    if recent_context:
        topic_change = detect_topic_change(state, recent_context)
        if topic_change:
            return {
                'should_ask': True,
                'reason': 'Topic change detected',
                'trigger_type': 'topic_change'
            }
    
    # Time-based trigger (if enough time has passed)
    if time_since_last >= MIN_QUESTION_INTERVAL * 1.5:  # 1.5x minimum interval
        return {
            'should_ask': True,
            'reason': f'Time-based trigger ({time_since_last:.1f}s since last question)',
            'trigger_type': 'time_based'
        }
    
    return {
        'should_ask': False,
        'reason': 'No trigger conditions met',
        'trigger_type': None
    }


def should_ask_followup(state, question_id=None, current_time=None):
    """
    Check if should ask a follow-up question.
    
    Args:
        state: State dictionary
        question_id: ID of question to follow up on (None = latest)
        current_time: Current timestamp
    
    Returns:
        dict with keys: 'should_ask', 'reason', 'question_id'
    """
    if current_time is None:
        current_time = time.time()
    
    # Check if follow-ups are enabled
    if not state.get('follow_up_enabled', True):
        return {
            'should_ask': False,
            'reason': 'Follow-ups disabled',
            'question_id': None
        }
    
    # Get question
    questions = state.get('questions', [])
    if not questions:
        return {
            'should_ask': False,
            'reason': 'No questions to follow up on',
            'question_id': None
        }
    
    if question_id is None:
        question = questions[-1]
        question_id = question.get('question_id', len(questions) - 1)
    else:
        question = None
        for q in questions:
            if q.get('question_id') == question_id:
                question = q
                break
        
        if question is None:
            return {
                'should_ask': False,
                'reason': 'Question not found',
                'question_id': None
            }
    
    # Check if answer exists for this question
    answers = state.get('answers', [])
    question_answers = [a for a in answers if a.get('question_id') == question_id]
    
    if not question_answers:
        return {
            'should_ask': False,
            'reason': 'No answer yet for this question',
            'question_id': question_id
        }
    
    # Check follow-up count for this question
    followup_count = len([q for q in questions if q.get('metadata', {}).get('is_followup') and 
                          q.get('metadata', {}).get('parent_question_id') == question_id])
    
    max_followups = state.get('max_follow_ups', 2)
    if followup_count >= max_followups:
        return {
            'should_ask': False,
            'reason': f'Maximum follow-ups reached ({followup_count}/{max_followups})',
            'question_id': question_id
        }
    
    # Check minimum interval
    last_question_time = state.get('last_question_time', 0)
    time_since_last = current_time - last_question_time
    
    if time_since_last < MIN_QUESTION_INTERVAL:
        return {
            'should_ask': False,
            'reason': f'Too soon since last question',
            'question_id': question_id
        }
    
    # Check if answer was recent enough
    if question_answers:
        last_answer = question_answers[-1]
        answer_time = last_answer.get('timestamp', 0)
        time_since_answer = current_time - answer_time
        
        # Wait a bit after answer before follow-up
        if time_since_answer < 5.0:  # Wait 5 seconds after answer
            return {
                'should_ask': False,
                'reason': 'Too soon after answer',
                'question_id': question_id
            }
        
        # Don't wait too long (answer might be stale)
        if time_since_answer > 120.0:  # 2 minutes
            return {
                'should_ask': False,
                'reason': 'Answer too old for follow-up',
                'question_id': question_id
            }
    
    return {
        'should_ask': True,
        'reason': 'Follow-up conditions met',
        'question_id': question_id
    }


def detect_natural_pause(recent_context):
    """
    Detect if there's a natural pause in speech.
    
    Args:
        recent_context: List of recent context entries
    
    Returns:
        bool: True if natural pause detected
    """
    if not recent_context:
        return False
    
    # Look for transcript entries
    transcript_entries = [e for e in recent_context if e.get('type') == 'transcript']
    
    if not transcript_entries:
        return False
    
    # Check if last transcript entry is old (indicating silence)
    if transcript_entries:
        last_transcript = transcript_entries[-1]
        last_time = last_transcript.get('timestamp', 0)
        current_time = time.time()
        
        time_since_speech = current_time - last_time
        return time_since_speech >= SILENCE_DURATION
    
    return False


def detect_topic_change(state, recent_context):
    """
    Detect if topic has changed significantly.
    
    Args:
        state: State dictionary
        recent_context: Recent context entries
    
    Returns:
        bool: True if topic change detected
    """
    # Get recent topics
    recent_topics = []
    for entry in recent_context:
        content = entry.get('content', {})
        if isinstance(content, dict):
            topics = content.get('topics', [])
            for topic in topics:
                topic_name = topic.get('topic', '') if isinstance(topic, dict) else str(topic)
                if topic_name:
                    recent_topics.append(topic_name.lower())
    
    # Get previously discussed topics
    discussed_topics = [t.lower() for t in state.get('topics_discussed', [])]
    
    # Check if new topics appeared
    if recent_topics:
        new_topics = [t for t in recent_topics if t not in discussed_topics]
        if len(new_topics) > 0:
            return True
    
    # Check keyword changes
    recent_keywords = set()
    for entry in recent_context:
        content = entry.get('content', {})
        if isinstance(content, dict):
            keywords = content.get('keywords', {})
            tech = keywords.get('technical', [])
            general = keywords.get('general', [])
            recent_keywords.update([k.lower() for k in tech + general])
    
    mentioned_keywords = state.get('keywords_mentioned', set())
    if isinstance(mentioned_keywords, list):
        mentioned_keywords = set([k.lower() for k in mentioned_keywords])
    
    new_keywords = recent_keywords - mentioned_keywords
    if len(new_keywords) >= 3:  # Significant number of new keywords
        return True
    
    return False


def get_optimal_question_timing(state, recent_context=None):
    """
    Get optimal timing information for next question.
    
    Args:
        state: State dictionary
        recent_context: Recent context entries
    
    Returns:
        dict with timing information
    """
    current_time = time.time()
    
    last_question_time = state.get('last_question_time', 0)
    time_since_last = current_time - last_question_time
    
    timing = {
        'current_time': current_time,
        'time_since_last_question': time_since_last,
        'can_ask_now': time_since_last >= MIN_QUESTION_INTERVAL,
        'time_until_can_ask': max(0, MIN_QUESTION_INTERVAL - time_since_last),
        'has_pause': False,
        'has_topic_change': False
    }
    
    if recent_context:
        timing['has_pause'] = detect_natural_pause(recent_context)
        timing['has_topic_change'] = detect_topic_change(state, recent_context)
    
    return timing


# Test function
def test_trigger_logic():
    """Test trigger logic functionality."""
    print("Testing trigger logic...")
    
    from context.state import create_state, add_context_entry
    
    # Create state
    state = create_state()
    state['session_start_time'] = time.time() - 15  # 15 seconds ago
    
    print("\n1. Testing initial question trigger:")
    should_ask = should_ask_initial_question(state)
    print(f"  Should ask initial: {should_ask}")
    
    print("\n2. Testing question trigger:")
    # Add some context
    add_context_entry(state, 'transcript', {'text': 'This is my project'})
    time.sleep(0.1)
    
    trigger = should_ask_question(state, recent_context=[])
    print(f"  Should ask: {trigger['should_ask']}")
    print(f"  Reason: {trigger['reason']}")
    print(f"  Type: {trigger['trigger_type']}")
    
    print("\n3. Testing follow-up trigger:")
    # Add question and answer
    add_context_entry(state, 'question', 'What is your project?')
    time.sleep(0.1)
    add_context_entry(state, 'answer', 'It is a web app', {'question_id': 0})
    time.sleep(0.1)
    
    followup = should_ask_followup(state)
    print(f"  Should ask follow-up: {followup['should_ask']}")
    print(f"  Reason: {followup['reason']}")
    
    print("\n4. Testing timing information:")
    timing = get_optimal_question_timing(state)
    print(f"  Time since last: {timing['time_since_last_question']:.1f}s")
    print(f"  Can ask now: {timing['can_ask_now']}")
    print(f"  Has pause: {timing['has_pause']}")
    
    return True


if __name__ == "__main__":
    test_trigger_logic()
