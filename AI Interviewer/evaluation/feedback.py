"""
Feedback generator module for creating detailed feedback from scores.
Provides constructive feedback highlighting strengths and areas for improvement.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import SCORE_PASSING
from evaluation.scorer import get_final_score


def generate_feedback(scores, state=None):
    """
    Generate detailed feedback from scores.
    
    Args:
        scores: Score dict (from scorer)
        state: Optional state dictionary for additional context
    
    Returns:
        dict with feedback sections
    """
    overall = scores.get('overall', 0.0)
    passing = overall >= SCORE_PASSING
    
    feedback = {
        'overall_score': overall,
        'passing': passing,
        'summary': _generate_summary(overall, passing),
        'strengths': _identify_strengths(scores),
        'weaknesses': _identify_weaknesses(scores),
        'recommendations': _generate_recommendations(scores),
        'detailed_scores': {
            'technical_depth': {
                'score': scores.get('technical_depth', 0.0),
                'feedback': _get_criterion_feedback('technical_depth', scores.get('technical_depth', 0.0))
            },
            'clarity': {
                'score': scores.get('clarity', 0.0),
                'feedback': _get_criterion_feedback('clarity', scores.get('clarity', 0.0))
            },
            'originality': {
                'score': scores.get('originality', 0.0),
                'feedback': _get_criterion_feedback('originality', scores.get('originality', 0.0))
            },
            'understanding': {
                'score': scores.get('understanding', 0.0),
                'feedback': _get_criterion_feedback('understanding', scores.get('understanding', 0.0))
            }
        }
    }
    
    return feedback


def generate_final_feedback(state):
    """
    Generate final feedback for the entire interview session.
    
    Args:
        state: State dictionary
    
    Returns:
        dict with comprehensive feedback
    """
    from evaluation.scorer import get_final_score
    
    final_scores = get_final_score(state)
    feedback = generate_feedback(final_scores, state)
    
    # Add session-specific information
    feedback['session_info'] = {
        'questions_asked': state.get('question_count', 0),
        'answers_received': state.get('answer_count', 0),
        'topics_discussed': len(state.get('topics_discussed', [])),
        'duration_minutes': (state.get('session_end_time', 0) - state.get('session_start_time', 0)) / 60
    }
    
    # Add question-answer pairs for context
    questions = state.get('questions', [])
    answers = state.get('answers', [])
    
    feedback['qa_pairs'] = []
    for answer in answers:
        question_id = answer.get('question_id')
        question_text = None
        for q in questions:
            if q.get('question_id') == question_id:
                question_text = q.get('question', '')
                break
        
        if question_text:
            feedback['qa_pairs'].append({
                'question': question_text,
                'answer': answer.get('answer', ''),
                'timestamp': answer.get('timestamp', 0)
            })
    
    return feedback


def format_feedback_text(feedback):
    """
    Format feedback as human-readable text.
    
    Args:
        feedback: Feedback dict from generate_feedback()
    
    Returns:
        str: Formatted feedback text
    """
    lines = []
    
    # Header
    lines.append("="*60)
    lines.append("INTERVIEW FEEDBACK")
    lines.append("="*60)
    lines.append("")
    
    # Overall score
    overall = feedback.get('overall_score', 0.0)
    passing = feedback.get('passing', False)
    status = "PASS" if passing else "NEEDS IMPROVEMENT"
    
    lines.append(f"Overall Score: {overall:.1f}/10.0")
    lines.append(f"Status: {status}")
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-"*60)
    lines.append(feedback.get('summary', ''))
    lines.append("")
    
    # Strengths
    strengths = feedback.get('strengths', [])
    if strengths:
        lines.append("STRENGTHS")
        lines.append("-"*60)
        for i, strength in enumerate(strengths, 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
    
    # Weaknesses
    weaknesses = feedback.get('weaknesses', [])
    if weaknesses:
        lines.append("AREAS FOR IMPROVEMENT")
        lines.append("-"*60)
        for i, weakness in enumerate(weaknesses, 1):
            lines.append(f"{i}. {weakness}")
        lines.append("")
    
    # Detailed scores
    lines.append("DETAILED SCORES")
    lines.append("-"*60)
    detailed = feedback.get('detailed_scores', {})
    for criterion, data in detailed.items():
        score = data.get('score', 0.0)
        criterion_name = criterion.replace('_', ' ').title()
        lines.append(f"{criterion_name}: {score:.1f}/10.0")
        lines.append(f"  {data.get('feedback', '')}")
        lines.append("")
    
    # Recommendations
    recommendations = feedback.get('recommendations', [])
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-"*60)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    # Session info
    session_info = feedback.get('session_info')
    if session_info:
        lines.append("SESSION INFORMATION")
        lines.append("-"*60)
        lines.append(f"Questions Asked: {session_info.get('questions_asked', 0)}")
        lines.append(f"Answers Received: {session_info.get('answers_received', 0)}")
        lines.append(f"Topics Discussed: {session_info.get('topics_discussed', 0)}")
        lines.append(f"Duration: {session_info.get('duration_minutes', 0):.1f} minutes")
        lines.append("")
    
    lines.append("="*60)
    
    return "\n".join(lines)


def _generate_summary(overall_score, passing):
    """Generate overall summary."""
    if overall_score >= 9.0:
        return "Excellent performance! You demonstrated strong technical understanding, clear communication, and creative thinking throughout the interview."
    elif overall_score >= 7.5:
        return "Good performance! You showed solid understanding and communication skills. There are some areas that could be strengthened further."
    elif overall_score >= 6.0:
        return "Satisfactory performance. You demonstrated basic understanding, but there are several areas that need improvement to reach a higher level."
    elif overall_score >= 4.0:
        return "Needs improvement. While you showed some understanding, significant work is needed in multiple areas to meet the expected standards."
    else:
        return "Significant improvement required. Focus on building fundamental understanding and communication skills."


def _identify_strengths(scores):
    """Identify strengths from scores."""
    strengths = []
    
    if scores.get('technical_depth', 0) >= 7.5:
        strengths.append("Strong technical depth - demonstrated good understanding of technical concepts and implementation details")
    
    if scores.get('clarity', 0) >= 7.5:
        strengths.append("Clear communication - explained concepts in a well-structured and understandable manner")
    
    if scores.get('originality', 0) >= 7.5:
        strengths.append("Creative thinking - provided original insights and specific examples from your experience")
    
    if scores.get('understanding', 0) >= 7.5:
        strengths.append("Good comprehension - showed solid understanding of the questions and provided relevant answers")
    
    # If no specific strengths, find the highest
    if not strengths:
        max_score = max(
            scores.get('technical_depth', 0),
            scores.get('clarity', 0),
            scores.get('originality', 0),
            scores.get('understanding', 0)
        )
        if max_score >= 6.0:
            if scores.get('technical_depth', 0) == max_score:
                strengths.append("Technical knowledge shows promise")
            elif scores.get('clarity', 0) == max_score:
                strengths.append("Communication skills are developing well")
            elif scores.get('originality', 0) == max_score:
                strengths.append("Creative thinking is evident")
            else:
                strengths.append("Understanding of concepts is improving")
    
    return strengths if strengths else ["Keep practicing and building your skills"]


def _identify_weaknesses(scores):
    """Identify weaknesses from scores."""
    weaknesses = []
    
    if scores.get('technical_depth', 0) < 6.0:
        weaknesses.append("Technical depth needs improvement - focus on explaining implementation details, algorithms, and technical decisions")
    
    if scores.get('clarity', 0) < 6.0:
        weaknesses.append("Clarity needs work - practice structuring your answers and explaining concepts more clearly")
    
    if scores.get('originality', 0) < 6.0:
        weaknesses.append("Originality could be enhanced - try to provide specific examples and personal insights rather than generic responses")
    
    if scores.get('understanding', 0) < 6.0:
        weaknesses.append("Understanding needs development - ensure your answers directly address the questions and demonstrate comprehension")
    
    return weaknesses


def _generate_recommendations(scores):
    """Generate recommendations for improvement."""
    recommendations = []
    
    if scores.get('technical_depth', 0) < 7.0:
        recommendations.append("Study technical concepts more deeply - learn about algorithms, data structures, design patterns, and system architecture")
        recommendations.append("Practice explaining your code and technical decisions in detail")
    
    if scores.get('clarity', 0) < 7.0:
        recommendations.append("Practice structuring your answers - use frameworks like 'First, I... Then, I... Finally, I...'")
        recommendations.append("Work on being concise yet complete - aim for 30-100 words per answer")
    
    if scores.get('originality', 0) < 7.0:
        recommendations.append("Include specific examples from your projects - mention actual code, features, or challenges you faced")
        recommendations.append("Share personal experiences and lessons learned")
    
    if scores.get('understanding', 0) < 7.0:
        recommendations.append("Listen carefully to questions and ensure your answers directly address what was asked")
        recommendations.append("Ask for clarification if you don't understand a question")
        recommendations.append("Practice explaining concepts in your own words")
    
    # General recommendations
    if not recommendations:
        recommendations.append("Continue practicing technical interviews")
        recommendations.append("Build more projects to gain hands-on experience")
    
    return recommendations


def _get_criterion_feedback(criterion, score):
    """Get feedback for a specific criterion."""
    feedback_templates = {
        'technical_depth': {
            (9.0, 10.0): "Excellent technical depth - demonstrated comprehensive understanding of technical concepts",
            (7.5, 9.0): "Good technical depth - showed solid understanding of technical aspects",
            (6.0, 7.5): "Adequate technical depth - basic understanding but could go deeper",
            (4.0, 6.0): "Limited technical depth - needs more focus on technical details",
            (0.0, 4.0): "Insufficient technical depth - significant improvement needed"
        },
        'clarity': {
            (9.0, 10.0): "Excellent clarity - communicated ideas very clearly and effectively",
            (7.5, 9.0): "Good clarity - explanations were clear and well-structured",
            (6.0, 7.5): "Adequate clarity - understandable but could be more organized",
            (4.0, 6.0): "Limited clarity - explanations need better structure",
            (0.0, 4.0): "Poor clarity - significant improvement in communication needed"
        },
        'originality': {
            (9.0, 10.0): "Highly original - provided unique insights and specific examples",
            (7.5, 9.0): "Good originality - included personal examples and insights",
            (6.0, 7.5): "Some originality - could include more specific examples",
            (4.0, 6.0): "Limited originality - answers were somewhat generic",
            (0.0, 4.0): "Lacks originality - answers were too generic, need specific examples"
        },
        'understanding': {
            (9.0, 10.0): "Excellent understanding - demonstrated deep comprehension of concepts",
            (7.5, 9.0): "Good understanding - showed solid grasp of the material",
            (6.0, 7.5): "Adequate understanding - basic comprehension but could be deeper",
            (4.0, 6.0): "Limited understanding - needs to develop better comprehension",
            (0.0, 4.0): "Poor understanding - significant improvement needed"
        }
    }
    
    templates = feedback_templates.get(criterion, {})
    
    for (min_score, max_score), feedback in templates.items():
        if min_score <= score < max_score:
            return feedback
    
    return "Score evaluation"


# Test function
def test_feedback():
    """Test feedback generation."""
    print("Testing feedback generator...")
    
    # Test with sample scores
    scores = {
        'technical_depth': 7.5,
        'clarity': 6.5,
        'originality': 8.0,
        'understanding': 7.0,
        'overall': 7.3
    }
    
    print("\n1. Testing feedback generation:")
    feedback = generate_feedback(scores)
    print(f"  Overall Score: {feedback['overall_score']:.1f}")
    print(f"  Passing: {feedback['passing']}")
    print(f"  Strengths: {len(feedback['strengths'])}")
    print(f"  Weaknesses: {len(feedback['weaknesses'])}")
    
    print("\n2. Testing formatted feedback:")
    formatted = format_feedback_text(feedback)
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    
    return True


if __name__ == "__main__":
    test_feedback()
