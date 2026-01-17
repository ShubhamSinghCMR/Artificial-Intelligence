"""
State management module for interview session.
Maintains context buffer, timestamps, and interview history.
"""

import sys
import os
from pathlib import Path
import time
from collections import deque

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import CONTEXT_WINDOW_SECONDS


def create_state():
    """
    Create initial interview state.
    
    Returns:
        dict with initial state structure
    """
    current_time = time.time()
    
    return {
        # Session control
        "is_active": True,
        "session_start_time": current_time,
        "session_end_time": None,
        
        # Context buffer (sliding window)
        "context_buffer": deque(maxlen=1000),  # Max 1000 entries
        "context_window_seconds": CONTEXT_WINDOW_SECONDS,
        
        # Current inputs (latest)
        "latest_screen_content": None,
        "latest_transcript": None,
        "latest_combined_content": None,
        
        # Accumulated content
        "all_screen_content": [],
        "all_transcripts": [],
        
        # Interview history
        "questions": [],  # List of {question, timestamp, context}
        "answers": [],     # List of {answer, timestamp, question_id}
        
        # Control flags
        "last_question_time": 0,
        "last_answer_time": 0,
        "question_count": 0,
        "answer_count": 0,
        
        # Evaluation
        "scores": {},
        "feedback": None,
        
        # Metadata
        "topics_discussed": [],
        "keywords_mentioned": set(),
        "code_snippets_seen": [],
    }


def add_context_entry(state, entry_type, content, metadata=None):
    """
    Add an entry to the context buffer.
    
    Args:
        state: State dictionary
        entry_type: Type of entry ('screen', 'transcript', 'combined', 'question', 'answer')
        content: Content data
        metadata: Optional metadata dict
    
    Returns:
        None (modifies state in place)
    """
    if metadata is None:
        metadata = {}
    
    entry = {
        'type': entry_type,
        'content': content,
        'timestamp': time.time(),
        'metadata': metadata
    }
    
    state['context_buffer'].append(entry)
    
    # Update latest content based on type
    if entry_type == 'screen':
        state['latest_screen_content'] = content
        state['all_screen_content'].append(entry)
    elif entry_type == 'transcript':
        state['latest_transcript'] = content
        state['all_transcripts'].append(entry)
    elif entry_type == 'combined':
        state['latest_combined_content'] = content
    elif entry_type == 'question':
        state['questions'].append({
            'question': content,
            'timestamp': entry['timestamp'],
            'question_id': len(state['questions']),
            'metadata': metadata
        })
        state['last_question_time'] = entry['timestamp']
        state['question_count'] += 1
    elif entry_type == 'answer':
        state['answers'].append({
            'answer': content,
            'timestamp': entry['timestamp'],
            'answer_id': len(state['answers']),
            'question_id': metadata.get('question_id', -1),
            'metadata': metadata
        })
        state['last_answer_time'] = entry['timestamp']
        state['answer_count'] += 1


def get_recent_context(state, seconds=None, max_entries=None):
    """
    Get recent context entries within time window.
    
    Args:
        state: State dictionary
        seconds: Time window in seconds (uses CONTEXT_WINDOW_SECONDS if None)
        max_entries: Maximum number of entries to return (None = all)
    
    Returns:
        list of context entries
    """
    if seconds is None:
        seconds = state.get('context_window_seconds', CONTEXT_WINDOW_SECONDS)
    
    current_time = time.time()
    cutoff_time = current_time - seconds
    
    recent = []
    for entry in state['context_buffer']:
        if entry['timestamp'] >= cutoff_time:
            recent.append(entry)
    
    # Sort by timestamp (oldest first)
    recent.sort(key=lambda x: x['timestamp'])
    
    if max_entries:
        recent = recent[-max_entries:]  # Get most recent N entries
    
    return recent


def get_context_summary(state, seconds=None):
    """
    Get a summary of recent context.
    
    Args:
        state: State dictionary
        seconds: Time window in seconds
    
    Returns:
        dict with summary information
    """
    recent = get_recent_context(state, seconds)
    
    summary = {
        'total_entries': len(recent),
        'time_window_seconds': seconds or state.get('context_window_seconds', CONTEXT_WINDOW_SECONDS),
        'entry_types': {},
        'latest_screen': None,
        'latest_transcript': None,
        'recent_questions': [],
        'recent_answers': []
    }
    
    for entry in recent:
        entry_type = entry['type']
        summary['entry_types'][entry_type] = summary['entry_types'].get(entry_type, 0) + 1
        
        if entry_type == 'screen' and summary['latest_screen'] is None:
            summary['latest_screen'] = entry
        elif entry_type == 'transcript' and summary['latest_transcript'] is None:
            summary['latest_transcript'] = entry
        elif entry_type == 'question':
            summary['recent_questions'].append(entry)
        elif entry_type == 'answer':
            summary['recent_answers'].append(entry)
    
    return summary


def get_combined_recent_text(state, seconds=None):
    """
    Get combined text from recent screen and transcript entries.
    
    Args:
        state: State dictionary
        seconds: Time window in seconds
    
    Returns:
        str: Combined text
    """
    recent = get_recent_context(state, seconds)
    
    screen_texts = []
    transcript_texts = []
    
    for entry in recent:
        if entry['type'] == 'screen':
            content = entry.get('content', {})
            if isinstance(content, dict):
                screen_texts.append(content.get('text', ''))
            elif isinstance(content, str):
                screen_texts.append(content)
        elif entry['type'] == 'transcript':
            content = entry.get('content', {})
            if isinstance(content, dict):
                transcript_texts.append(content.get('text', ''))
            elif isinstance(content, str):
                transcript_texts.append(content)
    
    combined = ""
    if screen_texts:
        combined += "[SCREEN CONTENT]\n" + "\n".join(screen_texts) + "\n\n"
    if transcript_texts:
        combined += "[SPEECH]\n" + "\n".join(transcript_texts) + "\n"
    
    return combined.strip()


def update_topics_and_keywords(state, combined_content):
    """
    Update topics and keywords from combined content.
    
    Args:
        state: State dictionary
        combined_content: Combined content dict from content_parser
    
    Returns:
        None (modifies state in place)
    """
    # Update topics
    topics = combined_content.get('topics', [])
    for topic in topics:
        topic_name = topic.get('topic', '')
        if topic_name and topic_name not in state['topics_discussed']:
            state['topics_discussed'].append(topic_name)
    
    # Update keywords
    keywords = combined_content.get('keywords', {})
    technical = keywords.get('technical', [])
    general = keywords.get('general', [])
    
    for keyword in technical + general:
        state['keywords_mentioned'].add(keyword.lower())
    
    # Update code snippets
    code_snippets = combined_content.get('code_snippets', [])
    for snippet in code_snippets:
        # Avoid duplicates
        code_text = snippet.get('code', '')
        if code_text and code_text not in [s.get('code', '') for s in state['code_snippets_seen']]:
            state['code_snippets_seen'].append(snippet)


def get_question_context(state, question_id=None):
    """
    Get context relevant to a specific question or latest question.
    
    Args:
        state: State dictionary
        question_id: Question ID (None = latest question)
    
    Returns:
        dict with question context
    """
    if question_id is None:
        if not state['questions']:
            return None
        question = state['questions'][-1]
        question_id = question['question_id']
    else:
        # Find question by ID
        question = None
        for q in state['questions']:
            if q['question_id'] == question_id:
                question = q
                break
        
        if question is None:
            return None
    
    # Get context around question time
    question_time = question['timestamp']
    context_window = 60  # 1 minute before and after
    
    relevant_context = get_recent_context(
        state, 
        seconds=context_window * 2
    )
    
    # Filter to entries around question time
    filtered_context = [
        entry for entry in relevant_context
        if abs(entry['timestamp'] - question_time) <= context_window
    ]
    
    return {
        'question': question,
        'context_entries': filtered_context,
        'context_text': get_combined_recent_text(state, seconds=context_window * 2)
    }


def end_session(state):
    """
    End the interview session.
    
    Args:
        state: State dictionary
    
    Returns:
        None (modifies state in place)
    """
    state['is_active'] = False
    state['session_end_time'] = time.time()
    
    # Calculate session duration
    duration = state['session_end_time'] - state['session_start_time']
    state['session_duration'] = duration


def get_session_stats(state):
    """
    Get session statistics.
    
    Args:
        state: State dictionary
    
    Returns:
        dict with session statistics
    """
    duration = 0
    if state.get('session_end_time'):
        duration = state['session_end_time'] - state['session_start_time']
    elif state['is_active']:
        duration = time.time() - state['session_start_time']
    
    return {
        'is_active': state['is_active'],
        'duration_seconds': duration,
        'duration_minutes': duration / 60,
        'question_count': state['question_count'],
        'answer_count': state['answer_count'],
        'topics_discussed': len(state['topics_discussed']),
        'keywords_mentioned': len(state['keywords_mentioned']),
        'code_snippets_seen': len(state['code_snippets_seen']),
        'context_entries': len(state['context_buffer'])
    }


# Test function
def test_state():
    """Test state management functionality."""
    print("Testing state management...")
    
    # Create state
    state = create_state()
    print(f"State created: is_active={state['is_active']}")
    
    # Add some context entries
    print("\nAdding context entries...")
    add_context_entry(state, 'screen', {'text': 'Hello world', 'confidence': 85.0})
    add_context_entry(state, 'transcript', {'text': 'This is my project', 'language': 'en'})
    add_context_entry(state, 'question', 'What is your project about?')
    add_context_entry(state, 'answer', 'It is a web application', {'question_id': 0})
    
    # Get recent context
    print("\nRecent context:")
    recent = get_recent_context(state, seconds=60)
    print(f"  Entries: {len(recent)}")
    for entry in recent:
        print(f"    [{entry['type']}] {entry.get('content', {})}")
    
    # Get context summary
    print("\nContext summary:")
    summary = get_context_summary(state)
    print(f"  Total entries: {summary['total_entries']}")
    print(f"  Entry types: {summary['entry_types']}")
    print(f"  Recent questions: {len(summary['recent_questions'])}")
    
    # Get combined text
    print("\nCombined recent text:")
    combined = get_combined_recent_text(state)
    print(combined[:200] + "..." if len(combined) > 200 else combined)
    
    # Session stats
    print("\nSession stats:")
    stats = get_session_stats(state)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


if __name__ == "__main__":
    test_state()
