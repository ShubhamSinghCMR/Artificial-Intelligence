"""
Scorer module for evaluating student answers.
Scores on technical depth, clarity, originality, and understanding.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    SCORE_WEIGHTS,
    SCORE_MIN,
    SCORE_MAX,
    SCORE_PASSING,
    TECHNICAL_DEPTH_KEYWORDS
)
from llm.inference import analyze_answer


def score_answer(question, answer, context=None, use_llm=True):
    """
    Score a student's answer.
    
    Args:
        question: Question that was asked
        answer: Student's answer
        context: Presentation context (optional)
        use_llm: Whether to use LLM for analysis (falls back to heuristics if False or unavailable)
    
    Returns:
        dict with scores and analysis
    """
    if not answer or len(answer.strip()) < 5:
        return {
            'technical_depth': 1.0,
            'clarity': 1.0,
            'originality': 1.0,
            'understanding': 1.0,
            'overall': 1.0,
            'method': 'heuristic (answer too short)'
        }
    
    # Try LLM analysis if requested
    if use_llm:
        try:
            analysis = analyze_answer(question, answer, context or "")
            if analysis.get('success'):
                scores = {
                    'technical_depth': analysis.get('technical_depth', 5.0),
                    'clarity': analysis.get('clarity', 5.0),
                    'understanding': analysis.get('understanding', 5.0),
                    'originality': _score_originality(answer, context),
                    'analysis_text': analysis.get('analysis_text', ''),
                    'method': 'llm'
                }
                scores['overall'] = calculate_overall_score(scores)
                return scores
        except Exception as e:
            print(f"LLM analysis failed, using heuristics: {e}")
    
    # Fallback to heuristic scoring
    return score_answer_heuristic(question, answer, context)


def score_answer_heuristic(question, answer, context=None):
    """
    Score answer using heuristic methods (no LLM required).
    
    Args:
        question: Question asked
        answer: Student's answer
        context: Optional context
    
    Returns:
        dict with scores
    """
    answer_lower = answer.lower()
    answer_words = answer.split()
    answer_length = len(answer_words)
    
    # Technical Depth (0-10)
    technical_score = _score_technical_depth(answer, context)
    
    # Clarity (0-10)
    clarity_score = _score_clarity(answer, answer_length)
    
    # Originality (0-10)
    originality_score = _score_originality(answer, context)
    
    # Understanding (0-10)
    understanding_score = _score_understanding(answer, question, answer_length)
    
    scores = {
        'technical_depth': technical_score,
        'clarity': clarity_score,
        'originality': originality_score,
        'understanding': understanding_score,
        'method': 'heuristic'
    }
    
    scores['overall'] = calculate_overall_score(scores)
    
    return scores


def _score_technical_depth(answer, context=None):
    """
    Score technical depth (0-10).
    
    Args:
        answer: Student's answer
        context: Optional context
    
    Returns:
        float: Technical depth score
    """
    answer_lower = answer.lower()
    score = 5.0  # Base score
    
    # Check for technical keywords
    tech_keywords_found = sum(1 for kw in TECHNICAL_DEPTH_KEYWORDS if kw.lower() in answer_lower)
    score += min(3.0, tech_keywords_found * 0.5)  # Up to +3 for keywords
    
    # Check for code mentions
    code_indicators = ['function', 'method', 'class', 'algorithm', 'data structure', 
                      'api', 'database', 'server', 'client', 'framework']
    code_mentions = sum(1 for indicator in code_indicators if indicator in answer_lower)
    score += min(2.0, code_mentions * 0.3)  # Up to +2 for code mentions
    
    # Check answer length (longer answers might indicate depth)
    word_count = len(answer.split())
    if word_count > 50:
        score += 1.0
    elif word_count < 10:
        score -= 1.0
    
    # Clamp to range
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _score_clarity(answer, answer_length):
    """
    Score clarity (0-10).
    
    Args:
        answer: Student's answer
        answer_length: Number of words
    
    Returns:
        float: Clarity score
    """
    score = 5.0  # Base score
    
    # Length-based (too short = unclear, too long = might be rambling)
    if 20 <= answer_length <= 100:
        score += 2.0  # Good length
    elif answer_length < 10:
        score -= 2.0  # Too short
    elif answer_length > 200:
        score -= 1.0  # Might be rambling
    
    # Check for structure indicators
    structure_words = ['first', 'then', 'next', 'finally', 'because', 'therefore', 
                      'however', 'for example', 'specifically']
    structure_count = sum(1 for word in structure_words if word in answer.lower())
    score += min(2.0, structure_count * 0.4)  # Up to +2 for structure
    
    # Check for repetition (reduces clarity)
    words = answer.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.5:
        score -= 1.0  # Too much repetition
    
    # Clamp to range
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _score_originality(answer, context=None):
    """
    Score originality (0-10).
    
    Args:
        answer: Student's answer
        context: Optional context
    
    Returns:
        float: Originality score
    """
    score = 5.0  # Base score
    
    answer_lower = answer.lower()
    
    # Check for generic phrases (reduces originality)
    generic_phrases = ['it is good', 'it works well', 'i think', 'i believe', 
                      'basically', 'kind of', 'sort of', 'you know']
    generic_count = sum(1 for phrase in generic_phrases if phrase in answer_lower)
    score -= min(2.0, generic_count * 0.5)  # Up to -2 for generic phrases
    
    # Check for specific examples (increases originality)
    example_indicators = ['for example', 'for instance', 'specifically', 'in my case',
                         'i used', 'i implemented', 'i created']
    example_count = sum(1 for indicator in example_indicators if indicator in answer_lower)
    score += min(2.0, example_count * 0.5)  # Up to +2 for examples
    
    # Check for personal experience
    personal_indicators = ['i', 'my', 'we', 'our', 'i found', 'i learned']
    personal_count = sum(1 for indicator in personal_indicators if indicator in answer_lower)
    if personal_count > 3:
        score += 1.0  # Personal experience
    
    # Clamp to range
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _score_understanding(answer, question, answer_length):
    """
    Score understanding (0-10).
    
    Args:
        answer: Student's answer
        question: Question asked
        answer_length: Number of words
    
    Returns:
        float: Understanding score
    """
    score = 5.0  # Base score
    
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    # Check if answer addresses the question
    question_keywords = set(question_lower.split())
    answer_keywords = set(answer_lower.split())
    overlap = len(question_keywords & answer_keywords)
    
    if overlap > 2:
        score += 2.0  # Answer is relevant
    elif overlap == 0:
        score -= 2.0  # Answer doesn't address question
    
    # Check for explanation depth
    explanation_words = ['because', 'since', 'due to', 'as a result', 'therefore',
                        'this means', 'which allows', 'enables']
    explanation_count = sum(1 for word in explanation_words if word in answer_lower)
    score += min(2.0, explanation_count * 0.5)  # Up to +2 for explanations
    
    # Check answer length (very short = might not understand)
    if answer_length < 10:
        score -= 1.5
    elif answer_length > 30:
        score += 1.0  # Longer answer might show understanding
    
    # Check for uncertainty (reduces understanding score)
    uncertainty_words = ['maybe', 'perhaps', 'i guess', 'i think', 'not sure',
                        'i don\'t know', 'probably']
    uncertainty_count = sum(1 for word in uncertainty_words if word in answer_lower)
    score -= min(1.5, uncertainty_count * 0.3)  # Up to -1.5 for uncertainty
    
    # Clamp to range
    return max(SCORE_MIN, min(SCORE_MAX, score))


def calculate_overall_score(scores):
    """
    Calculate weighted overall score.
    
    Args:
        scores: dict with individual scores
    
    Returns:
        float: Overall score (weighted average)
    """
    overall = 0.0
    
    for criterion, weight in SCORE_WEIGHTS.items():
        score = scores.get(criterion, 5.0)
        overall += score * weight
    
    # Round to 1 decimal place
    return round(overall, 1)


def aggregate_scores(answer_scores):
    """
    Aggregate scores from multiple answers.
    
    Args:
        answer_scores: List of score dicts
    
    Returns:
        dict with aggregated scores
    """
    if not answer_scores:
        return {
            'technical_depth': 0.0,
            'clarity': 0.0,
            'originality': 0.0,
            'understanding': 0.0,
            'overall': 0.0,
            'answer_count': 0
        }
    
    # Calculate averages
    aggregated = {
        'technical_depth': 0.0,
        'clarity': 0.0,
        'originality': 0.0,
        'understanding': 0.0,
        'answer_count': len(answer_scores)
    }
    
    for scores in answer_scores:
        aggregated['technical_depth'] += scores.get('technical_depth', 0)
        aggregated['clarity'] += scores.get('clarity', 0)
        aggregated['originality'] += scores.get('originality', 0)
        aggregated['understanding'] += scores.get('understanding', 0)
    
    # Average
    count = aggregated['answer_count']
    for key in ['technical_depth', 'clarity', 'originality', 'understanding']:
        aggregated[key] = round(aggregated[key] / count, 1)
    
    # Calculate overall
    aggregated['overall'] = calculate_overall_score(aggregated)
    
    return aggregated


def get_final_score(state):
    """
    Get final score for the entire interview session.
    
    Args:
        state: State dictionary
    
    Returns:
        dict with final scores
    """
    answers = state.get('answers', [])
    
    if not answers:
        return {
            'technical_depth': 0.0,
            'clarity': 0.0,
            'originality': 0.0,
            'understanding': 0.0,
            'overall': 0.0,
            'answer_count': 0,
            'passing': False
        }
    
    # Get scores for all answers
    answer_scores = []
    for answer_entry in answers:
        answer = answer_entry.get('answer', '')
        question_id = answer_entry.get('question_id', -1)
        
        # Find corresponding question
        questions = state.get('questions', [])
        question = None
        for q in questions:
            if q.get('question_id') == question_id:
                question = q.get('question', '')
                break
        
        if question and answer:
            scores = score_answer(question, answer, use_llm=False)  # Use heuristics for speed
            answer_scores.append(scores)
    
    # Aggregate
    final = aggregate_scores(answer_scores)
    final['passing'] = final['overall'] >= SCORE_PASSING
    
    return final


# Test function
def test_scorer():
    """Test scorer functionality."""
    print("Testing scorer...")
    
    # Test answer scoring
    question = "How does your algorithm work?"
    answer = "I implemented a binary search algorithm that has O(log n) time complexity. It works by dividing the search space in half at each step, which makes it very efficient for sorted arrays."
    
    print("\n1. Testing answer scoring:")
    scores = score_answer(question, answer, use_llm=False)
    print(f"  Technical Depth: {scores['technical_depth']:.1f}")
    print(f"  Clarity: {scores['clarity']:.1f}")
    print(f"  Originality: {scores['originality']:.1f}")
    print(f"  Understanding: {scores['understanding']:.1f}")
    print(f"  Overall: {scores['overall']:.1f}")
    print(f"  Method: {scores['method']}")
    
    # Test aggregation
    print("\n2. Testing score aggregation:")
    answer_scores = [
        {'technical_depth': 8.0, 'clarity': 7.0, 'originality': 6.0, 'understanding': 8.0},
        {'technical_depth': 7.0, 'clarity': 8.0, 'originality': 7.0, 'understanding': 7.0},
        {'technical_depth': 6.0, 'clarity': 6.0, 'originality': 8.0, 'understanding': 6.0}
    ]
    
    aggregated = aggregate_scores(answer_scores)
    print(f"  Aggregated Technical Depth: {aggregated['technical_depth']:.1f}")
    print(f"  Aggregated Clarity: {aggregated['clarity']:.1f}")
    print(f"  Aggregated Overall: {aggregated['overall']:.1f}")
    print(f"  Answer Count: {aggregated['answer_count']}")
    
    # Test short answer
    print("\n3. Testing short answer:")
    short_answer = "It works."
    short_scores = score_answer(question, short_answer, use_llm=False)
    print(f"  Overall: {short_scores['overall']:.1f}")
    print(f"  (Should be low due to short answer)")
    
    return True


if __name__ == "__main__":
    test_scorer()
