"""
Context fusion module for combining OCR and STT content.
Time-aligns screen content with speech and creates unified context.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from perception.content_parser import (
    parse_ocr_content,
    parse_transcript_content,
    combine_content
)
from context.state import (
    add_context_entry,
    update_topics_and_keywords,
    get_recent_context
)


def fuse_ocr_and_transcript(ocr_result, transcript_result, state=None):
    """
    Fuse OCR and transcript results into unified context.
    
    Args:
        ocr_result: OCR result dict from perception.ocr.extract_text()
        transcript_result: Transcript result dict from perception.stt.transcribe_audio()
        state: Optional state dictionary (if provided, updates state)
    
    Returns:
        dict with fused content
    """
    # Parse both contents
    ocr_content = parse_ocr_content(ocr_result) if ocr_result else None
    transcript_content = parse_transcript_content(transcript_result) if transcript_result else None
    
    # Handle cases where one or both are missing
    if ocr_content is None and transcript_content is None:
        return None
    
    if ocr_content is None:
        ocr_content = {
            'text': '',
            'code_snippets': [],
            'keywords': {'technical': [], 'general': [], 'counts': {}},
            'topics': [],
            'word_count': 0,
            'confidence': 0.0
        }
    
    if transcript_content is None:
        transcript_content = {
            'text': '',
            'code_snippets': [],
            'keywords': {'technical': [], 'general': [], 'counts': {}},
            'topics': [],
            'word_count': 0,
            'language': None,
            'segments': []
        }
    
    # Combine contents
    combined = combine_content(ocr_content, transcript_content)
    
    # Add timestamp
    combined['timestamp'] = time.time()
    combined['ocr_timestamp'] = time.time()  # Can be improved with actual timestamps
    combined['transcript_timestamp'] = time.time()
    
    # Update state if provided
    if state is not None:
        # Add to context buffer
        if ocr_result:
            add_context_entry(state, 'screen', ocr_content)
        if transcript_result:
            add_context_entry(state, 'transcript', transcript_content)
        
        add_context_entry(state, 'combined', combined)
        
        # Update topics and keywords
        update_topics_and_keywords(state, combined)
    
    return combined


def fuse_recent_context(state, seconds=None):
    """
    Fuse recent context entries from state buffer.
    
    Args:
        state: State dictionary
        seconds: Time window in seconds (None = use context window)
    
    Returns:
        dict with fused recent context
    """
    recent = get_recent_context(state, seconds=seconds)
    
    if not recent:
        return {
            'text': '',
            'code_snippets': [],
            'keywords': {'technical': [], 'general': [], 'counts': {}},
            'topics': [],
            'timestamp': time.time()
        }
    
    # Separate screen and transcript entries
    screen_entries = [e for e in recent if e['type'] == 'screen']
    transcript_entries = [e for e in recent if e['type'] == 'transcript']
    
    # Get latest combined if available
    combined_entries = [e for e in recent if e['type'] == 'combined']
    
    if combined_entries:
        # Use most recent combined entry
        return combined_entries[-1]['content']
    
    # Otherwise, combine from individual entries
    # Aggregate OCR content
    ocr_texts = []
    ocr_code = []
    ocr_keywords = {'technical': set(), 'general': set(), 'counts': {}}
    ocr_topics = []
    
    for entry in screen_entries:
        content = entry.get('content', {})
        if isinstance(content, dict):
            if content.get('text'):
                ocr_texts.append(content['text'])
            ocr_code.extend(content.get('code_snippets', []))
            kw = content.get('keywords', {})
            ocr_keywords['technical'].update(kw.get('technical', []))
            ocr_keywords['general'].update(kw.get('general', []))
            ocr_topics.extend(content.get('topics', []))
    
    # Aggregate transcript content
    trans_texts = []
    trans_code = []
    trans_keywords = {'technical': set(), 'general': set(), 'counts': {}}
    trans_topics = []
    
    for entry in transcript_entries:
        content = entry.get('content', {})
        if isinstance(content, dict):
            if content.get('text'):
                trans_texts.append(content['text'])
            trans_code.extend(content.get('code_snippets', []))
            kw = content.get('keywords', {})
            trans_keywords['technical'].update(kw.get('technical', []))
            trans_keywords['general'].update(kw.get('general', []))
            trans_topics.extend(content.get('topics', []))
    
    # Create combined structure
    combined_text = ""
    if ocr_texts:
        combined_text += "[SCREEN CONTENT]\n" + "\n".join(ocr_texts) + "\n\n"
    if trans_texts:
        combined_text += "[SPEECH]\n" + "\n".join(trans_texts) + "\n"
    
    # Merge keywords
    all_technical = list(ocr_keywords['technical'] | trans_keywords['technical'])
    all_general = list(ocr_keywords['general'] | trans_keywords['general'])
    
    # Merge code snippets (simple deduplication)
    all_code = ocr_code + trans_code
    unique_code = []
    seen_codes = set()
    for code in all_code:
        code_str = code.get('code', '') if isinstance(code, dict) else str(code)
        if code_str and code_str not in seen_codes:
            seen_codes.add(code_str)
            unique_code.append(code)
    
    # Merge topics (deduplicate)
    all_topics = ocr_topics + trans_topics
    unique_topics = []
    seen_topics = set()
    for topic in all_topics:
        topic_name = topic.get('topic', '') if isinstance(topic, dict) else str(topic)
        if topic_name and topic_name.lower() not in seen_topics:
            seen_topics.add(topic_name.lower())
            unique_topics.append(topic)
    
    return {
        'text': combined_text.strip(),
        'code_snippets': unique_code[:20],  # Limit to 20 most recent
        'keywords': {
            'technical': all_technical,
            'general': all_general,
            'counts': {}
        },
        'topics': unique_topics[:10],  # Limit to 10 most recent
        'timestamp': time.time(),
        'has_screen_content': len(ocr_texts) > 0,
        'has_speech_content': len(trans_texts) > 0
    }


def get_context_for_llm(state, seconds=None, include_questions=False):
    """
    Get formatted context for LLM input.
    
    Args:
        state: State dictionary
        seconds: Time window in seconds
        include_questions: Whether to include recent questions/answers
    
    Returns:
        str: Formatted context string for LLM
    """
    # Get fused recent context
    fused = fuse_recent_context(state, seconds=seconds)
    
    # Build context string
    context_parts = []
    
    # Add screen content
    if fused.get('has_screen_content'):
        context_parts.append("=== SCREEN CONTENT ===")
        screen_text = fused.get('text', '').split('[SCREEN CONTENT]')
        if len(screen_text) > 1:
            context_parts.append(screen_text[1].split('[SPEECH]')[0].strip())
    
    # Add speech content
    if fused.get('has_speech_content'):
        context_parts.append("\n=== STUDENT SPEECH ===")
        speech_text = fused.get('text', '').split('[SPEECH]')
        if len(speech_text) > 1:
            context_parts.append(speech_text[1].strip())
    
    # Add code snippets
    code_snippets = fused.get('code_snippets', [])
    if code_snippets:
        context_parts.append("\n=== CODE SNIPPETS ===")
        for i, snippet in enumerate(code_snippets[:5], 1):  # Top 5
            code = snippet.get('code', '') if isinstance(snippet, dict) else str(snippet)
            lang = snippet.get('language', '') if isinstance(snippet, dict) else ''
            context_parts.append(f"\n[{i}] ({lang}):\n{code[:200]}...")  # Limit length
    
    # Add topics
    topics = fused.get('topics', [])
    if topics:
        context_parts.append("\n=== TOPICS DISCUSSED ===")
        topic_names = [t.get('topic', '') if isinstance(t, dict) else str(t) for t in topics[:5]]
        context_parts.append(", ".join(topic_names))
    
    # Add keywords
    keywords = fused.get('keywords', {})
    tech_keywords = keywords.get('technical', [])
    if tech_keywords:
        context_parts.append(f"\n=== TECHNICAL TERMS ===")
        context_parts.append(", ".join(tech_keywords[:10]))
    
    # Add questions/answers if requested
    if include_questions:
        questions = state.get('questions', [])
        answers = state.get('answers', [])
        
        if questions:
            context_parts.append("\n=== RECENT QUESTIONS ===")
            for q in questions[-3:]:  # Last 3 questions
                context_parts.append(f"Q: {q.get('question', '')}")
        
        if answers:
            context_parts.append("\n=== RECENT ANSWERS ===")
            for a in answers[-3:]:  # Last 3 answers
                context_parts.append(f"A: {a.get('answer', '')}")
    
    return "\n".join(context_parts)


def time_align_content(screen_content, transcript_content, tolerance_seconds=5.0):
    """
    Time-align screen content with transcript based on timestamps.
    
    Args:
        screen_content: Screen content dict with timestamp
        transcript_content: Transcript content dict with timestamp
        tolerance_seconds: Time tolerance for alignment
    
    Returns:
        bool: True if content is time-aligned, False otherwise
    """
    if not screen_content or not transcript_content:
        return False
    
    screen_time = screen_content.get('timestamp', 0)
    transcript_time = transcript_content.get('timestamp', 0)
    
    time_diff = abs(screen_time - transcript_time)
    
    return time_diff <= tolerance_seconds


# Test function
def test_fusion():
    """Test context fusion functionality."""
    print("Testing context fusion...")
    
    # Mock OCR and transcript results
    ocr_result = {
        'text': 'def hello(): print("Hello World")',
        'confidence': 85.0
    }
    
    transcript_result = {
        'text': 'This is my Python project that prints hello world',
        'language': 'en',
        'segments': []
    }
    
    # Test fusion
    print("\n1. Testing basic fusion:")
    fused = fuse_ocr_and_transcript(ocr_result, transcript_result)
    print(f"  Combined text length: {len(fused.get('text', ''))}")
    print(f"  Code snippets: {len(fused.get('code_snippets', []))}")
    print(f"  Technical keywords: {fused.get('keywords', {}).get('technical', [])}")
    
    # Test with state
    print("\n2. Testing fusion with state:")
    from context.state import create_state
    state = create_state()
    
    fused_with_state = fuse_ocr_and_transcript(ocr_result, transcript_result, state)
    print(f"  State updated: {state.get('latest_combined_content') is not None}")
    print(f"  Topics in state: {len(state.get('topics_discussed', []))}")
    
    # Test recent context fusion
    print("\n3. Testing recent context fusion:")
    recent_fused = fuse_recent_context(state)
    print(f"  Recent context text length: {len(recent_fused.get('text', ''))}")
    
    # Test LLM context formatting
    print("\n4. Testing LLM context formatting:")
    llm_context = get_context_for_llm(state, include_questions=True)
    print(f"  LLM context length: {len(llm_context)}")
    print(f"  First 300 chars:\n{llm_context[:300]}...")
    
    return True


if __name__ == "__main__":
    test_fusion()
