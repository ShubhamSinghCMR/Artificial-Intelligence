"""
Content parser module for analyzing and structuring extracted text.
Identifies code snippets, keywords, topics, and presentation structure.
"""

import sys
import os
from pathlib import Path
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import TECHNICAL_DEPTH_KEYWORDS


def extract_code_snippets(text):
    """
    Extract code snippets from text.
    
    Args:
        text: Text string from OCR or transcript
    
    Returns:
        list of dicts with keys: 'code', 'language', 'start_pos', 'end_pos'
    """
    code_snippets = []
    
    # Common code patterns
    code_patterns = [
        # Function definitions
        (r'(def\s+\w+\s*\([^)]*\)\s*:.*?)(?=\n\S|\n\n|$)', 'python'),
        # Class definitions
        (r'(class\s+\w+.*?:.*?)(?=\n\S|\n\n|$)', 'python'),
        # Variable assignments with operators
        (r'(\w+\s*[=+\-*/]+\s*[^;\n]+)', 'generic'),
        # Code blocks with braces
        (r'(\{[^}]{10,}\})', 'generic'),
        # Code blocks with brackets
        (r'(\[[^\]]{10,}\])', 'generic'),
        # Import statements
        (r'(import\s+[\w\s,\.]+|from\s+[\w\.]+\s+import\s+[\w\s,]+)', 'python'),
        # HTML/XML tags
        (r'(<[^>]{5,}>)', 'html'),
        # SQL queries
        (r'(SELECT\s+.*?FROM\s+.*?(?:WHERE\s+.*?)?(?:;|$))', 'sql'),
        # JSON-like structures
        (r'(\{[^{}]*"[^"]+"\s*:\s*[^,}]+[^}]*\})', 'json'),
    ]
    
    for pattern, language in code_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        for match in matches:
            code = match.group(1).strip()
            # Filter out very short matches (likely false positives)
            if len(code) > 10:
                code_snippets.append({
                    'code': code,
                    'language': language,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
    
    # Remove duplicates and overlapping matches
    code_snippets = _deduplicate_code_snippets(code_snippets)
    
    return code_snippets


def extract_keywords(text, custom_keywords=None):
    """
    Extract keywords from text.
    
    Args:
        text: Text string
        custom_keywords: List of custom keywords to look for (uses TECHNICAL_DEPTH_KEYWORDS if None)
    
    Returns:
        dict with keys: 'technical', 'general', 'counts'
    """
    if custom_keywords is None:
        custom_keywords = TECHNICAL_DEPTH_KEYWORDS
    
    text_lower = text.lower()
    
    # Technical keywords
    technical_found = []
    technical_counts = {}
    
    for keyword in custom_keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            technical_found.append(keyword)
            technical_counts[keyword] = len(matches)
    
    # Common technical terms
    common_tech_terms = [
        'api', 'database', 'server', 'client', 'framework', 'library',
        'function', 'method', 'variable', 'class', 'object', 'module',
        'interface', 'component', 'system', 'application', 'software',
        'data', 'model', 'view', 'controller', 'backend', 'frontend',
        'javascript', 'python', 'java', 'html', 'css', 'sql', 'json',
        'http', 'https', 'rest', 'graphql', 'docker', 'kubernetes'
    ]
    
    general_keywords = []
    general_counts = {}
    
    for term in common_tech_terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            general_keywords.append(term)
            general_counts[term] = len(matches)
    
    return {
        'technical': technical_found,
        'general': general_keywords,
        'counts': {**technical_counts, **general_counts}
    }


def identify_topics(text):
    """
    Identify main topics/sections in the text.
    
    Args:
        text: Text string
    
    Returns:
        list of dicts with keys: 'topic', 'confidence', 'context'
    """
    topics = []
    
    # Look for section headers (common patterns)
    header_patterns = [
        r'^#+\s+(.+)$',  # Markdown headers
        r'^(\d+\.\s+[A-Z][^\n]+)$',  # Numbered headers
        r'^([A-Z][A-Z\s]{5,})$',  # ALL CAPS headers
        r'^(Introduction|Overview|Features|Architecture|Implementation|Conclusion|Summary)[:\.]?',
    ]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check for header patterns
        for pattern in header_patterns:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                topic = match.group(1) if match.groups() else line_stripped
                # Get context (next few lines)
                context = '\n'.join(lines[i+1:i+4])[:200] if i+1 < len(lines) else ""
                topics.append({
                    'topic': topic.strip(),
                    'confidence': 0.8,
                    'context': context
                })
                break
    
    # If no headers found, try to identify topics from keywords
    if not topics:
        keywords = extract_keywords(text)
        if keywords['technical']:
            for tech_term in keywords['technical'][:5]:  # Top 5
                topics.append({
                    'topic': tech_term.title(),
                    'confidence': 0.6,
                    'context': ""
                })
    
    return topics


def parse_ocr_content(ocr_result):
    """
    Parse OCR extraction result into structured content.
    
    Args:
        ocr_result: dict from perception.ocr.extract_text()
    
    Returns:
        dict with structured content information
    """
    text = ocr_result.get('text', '')
    confidence = ocr_result.get('confidence', 0.0)
    
    if not text:
        return {
            'text': '',
            'code_snippets': [],
            'keywords': {'technical': [], 'general': [], 'counts': {}},
            'topics': [],
            'word_count': 0,
            'confidence': confidence
        }
    
    code_snippets = extract_code_snippets(text)
    keywords = extract_keywords(text)
    topics = identify_topics(text)
    
    return {
        'text': text,
        'code_snippets': code_snippets,
        'keywords': keywords,
        'topics': topics,
        'word_count': len(text.split()),
        'confidence': confidence
    }


def parse_transcript_content(transcript_result):
    """
    Parse STT transcription result into structured content.
    
    Args:
        transcript_result: dict from perception.stt.transcribe_audio()
    
    Returns:
        dict with structured content information
    """
    text = transcript_result.get('text', '')
    segments = transcript_result.get('segments', [])
    language = transcript_result.get('language', None)
    
    if not text:
        return {
            'text': '',
            'code_snippets': [],
            'keywords': {'technical': [], 'general': [], 'counts': {}},
            'topics': [],
            'segments': [],
            'word_count': 0,
            'language': language
        }
    
    code_snippets = extract_code_snippets(text)
    keywords = extract_keywords(text)
    topics = identify_topics(text)
    
    return {
        'text': text,
        'code_snippets': code_snippets,
        'keywords': keywords,
        'topics': topics,
        'segments': segments,
        'word_count': len(text.split()),
        'language': language
    }


def combine_content(ocr_content, transcript_content):
    """
    Combine OCR and transcript content into unified structure.
    
    Args:
        ocr_content: Parsed OCR content dict
        transcript_content: Parsed transcript content dict
    
    Returns:
        dict with combined content
    """
    # Merge text
    combined_text = ""
    if ocr_content.get('text'):
        combined_text += f"[SCREEN] {ocr_content['text']}\n"
    if transcript_content.get('text'):
        combined_text += f"[SPEECH] {transcript_content['text']}\n"
    
    # Merge code snippets (deduplicate)
    all_code = ocr_content.get('code_snippets', []) + transcript_content.get('code_snippets', [])
    unique_code = _deduplicate_code_snippets(all_code)
    
    # Merge keywords
    ocr_keywords = ocr_content.get('keywords', {'technical': [], 'general': [], 'counts': {}})
    trans_keywords = transcript_content.get('keywords', {'technical': [], 'general': [], 'counts': {}})
    
    combined_technical = list(set(ocr_keywords.get('technical', []) + trans_keywords.get('technical', [])))
    combined_general = list(set(ocr_keywords.get('general', []) + trans_keywords.get('general', [])))
    
    # Merge counts
    combined_counts = {}
    for key, count in {**ocr_keywords.get('counts', {}), **trans_keywords.get('counts', {})}.items():
        combined_counts[key] = combined_counts.get(key, 0) + count
    
    # Merge topics
    all_topics = ocr_content.get('topics', []) + transcript_content.get('topics', [])
    unique_topics = _deduplicate_topics(all_topics)
    
    return {
        'text': combined_text.strip(),
        'code_snippets': unique_code,
        'keywords': {
            'technical': combined_technical,
            'general': combined_general,
            'counts': combined_counts
        },
        'topics': unique_topics,
        'word_count': ocr_content.get('word_count', 0) + transcript_content.get('word_count', 0),
        'ocr_confidence': ocr_content.get('confidence', 0.0),
        'has_screen_content': bool(ocr_content.get('text')),
        'has_speech_content': bool(transcript_content.get('text')),
        'segments': transcript_content.get('segments', [])
    }


def _deduplicate_code_snippets(code_snippets):
    """Remove duplicate and overlapping code snippets."""
    if not code_snippets:
        return []
    
    # Sort by start position
    sorted_snippets = sorted(code_snippets, key=lambda x: x['start_pos'])
    
    unique = []
    for snippet in sorted_snippets:
        # Check if this snippet overlaps with any existing one
        is_duplicate = False
        for existing in unique:
            # Check if codes are similar (simple string comparison)
            if _code_similarity(snippet['code'], existing['code']) > 0.8:
                is_duplicate = True
                break
            # Check if positions overlap significantly
            if (snippet['start_pos'] < existing['end_pos'] and 
                snippet['end_pos'] > existing['start_pos']):
                # Keep the longer one
                if len(snippet['code']) > len(existing['code']):
                    unique.remove(existing)
                    unique.append(snippet)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(snippet)
    
    return unique


def _deduplicate_topics(topics):
    """Remove duplicate topics."""
    if not topics:
        return []
    
    unique = []
    seen = set()
    
    for topic in topics:
        topic_key = topic['topic'].lower().strip()
        if topic_key not in seen:
            seen.add(topic_key)
            unique.append(topic)
    
    return unique


def _code_similarity(code1, code2):
    """Calculate similarity between two code snippets (0.0-1.0)."""
    if not code1 or not code2:
        return 0.0
    
    # Simple similarity: check if one contains the other
    code1_lower = code1.lower().strip()
    code2_lower = code2.lower().strip()
    
    if code1_lower == code2_lower:
        return 1.0
    
    if code1_lower in code2_lower or code2_lower in code1_lower:
        return 0.9
    
    # Calculate character overlap
    set1 = set(code1_lower)
    set2 = set(code2_lower)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def get_content_summary(combined_content):
    """
    Get a human-readable summary of the content.
    
    Args:
        combined_content: Combined content dict from combine_content()
    
    Returns:
        str: Summary text
    """
    parts = []
    
    if combined_content.get('has_screen_content'):
        parts.append("Screen content detected")
    if combined_content.get('has_speech_content'):
        parts.append("Speech content detected")
    
    word_count = combined_content.get('word_count', 0)
    if word_count > 0:
        parts.append(f"{word_count} words")
    
    code_count = len(combined_content.get('code_snippets', []))
    if code_count > 0:
        parts.append(f"{code_count} code snippets")
    
    tech_keywords = combined_content.get('keywords', {}).get('technical', [])
    if tech_keywords:
        parts.append(f"Technical terms: {', '.join(tech_keywords[:5])}")
    
    topics = combined_content.get('topics', [])
    if topics:
        topic_names = [t['topic'] for t in topics[:3]]
        parts.append(f"Topics: {', '.join(topic_names)}")
    
    return " | ".join(parts) if parts else "No content detected"


# Test function
def test_content_parser():
    """Test content parser functionality."""
    print("Testing content parser...")
    
    # Test OCR content
    print("\n1. Testing OCR content parsing:")
    ocr_result = {
        'text': '''
        # My Project
        
        def calculate_sum(a, b):
            return a + b
        
        This project uses Python and implements algorithms for optimization.
        The architecture follows MVC pattern.
        ''',
        'confidence': 85.0
    }
    
    ocr_content = parse_ocr_content(ocr_result)
    print(f"  Code snippets: {len(ocr_content['code_snippets'])}")
    print(f"  Technical keywords: {ocr_content['keywords']['technical']}")
    print(f"  Topics: {[t['topic'] for t in ocr_content['topics']]}")
    
    # Test transcript content
    print("\n2. Testing transcript content parsing:")
    transcript_result = {
        'text': 'I am building a web application using Python and FastAPI. The system uses a database for storage.',
        'segments': [],
        'language': 'en'
    }
    
    transcript_content = parse_transcript_content(transcript_result)
    print(f"  Technical keywords: {transcript_content['keywords']['technical']}")
    print(f"  General keywords: {transcript_content['keywords']['general']}")
    
    # Test combining
    print("\n3. Testing content combination:")
    combined = combine_content(ocr_content, transcript_content)
    summary = get_content_summary(combined)
    print(f"  Summary: {summary}")
    print(f"  Total code snippets: {len(combined['code_snippets'])}")
    print(f"  Combined technical keywords: {combined['keywords']['technical']}")
    
    return True


if __name__ == "__main__":
    test_content_parser()
