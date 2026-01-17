"""
LLM inference module for generating questions and analyzing answers.
Uses llama-cpp-python for local inference with graceful fallback.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.loader import get_llm_model
from config.settings import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_TOP_K
)
from config.prompts import (
    get_question_prompt,
    get_followup_prompt,
    get_analysis_prompt
)


def generate_question(context, model=None, prompt_template=None):
    """
    Generate a question based on presentation context.
    
    Args:
        context: Context string (from context fusion)
        model: LLM model (loads if None)
        prompt_template: Prompt template string (uses default if None)
    
    Returns:
        dict with keys: 'question', 'success', 'error'
    """
    if model is None:
        model = get_llm_model()
    
    if model is None:
        # Fallback to template-based question generation
        return _generate_question_fallback(context)
    
    # Get prompt template
    if prompt_template is None:
        # Detect if context has code
        has_code = any(keyword in context.lower() for keyword in ['def ', 'class ', 'function', 'import ', '{', '}'])
        focus = "code" if has_code else "general"
        prompt_template = get_question_prompt(context, focus=focus)

    try:
        # Format prompt
        prompt = prompt_template.format(context=context[:2000])  # Limit context length
        
        # Generate response
        response = model(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            stop=["\n\n", "Context:", "Question:", "Answer:"],
            echo=False
        )
        
        # Extract question
        question = response['choices'][0]['text'].strip()
        
        # Clean up question
        question = _clean_question(question)
        
        return {
            'question': question,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        print(f"Error generating question: {e}")
        return _generate_question_fallback(context)


def generate_followup_question(previous_question, answer, context, model=None):
    """
    Generate a follow-up question based on previous answer.
    
    Args:
        previous_question: Previous question string
        answer: Student's answer string
        context: Current context
        model: LLM model (loads if None)
    
    Returns:
        dict with keys: 'question', 'success', 'error'
    """
    if model is None:
        model = get_llm_model()
    
    if model is None:
        return _generate_followup_fallback(previous_question, answer, context)
    
    # Get follow-up prompt
    prompt_template = get_followup_prompt(
        previous_question=previous_question,
        answer=answer[:500],  # Limit answer length
        context=context[:1000],  # Limit context length
        clarification=False
    )

    try:
        prompt = prompt_template
        
        response = model(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            stop=["\n\n", "Context:", "Question:", "Answer:"],
            echo=False
        )
        
        question = response['choices'][0]['text'].strip()
        question = _clean_question(question)
        
        return {
            'question': question,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        print(f"Error generating follow-up: {e}")
        return _generate_followup_fallback(previous_question, answer, context)


def analyze_answer(question, answer, context, model=None):
    """
    Analyze student's answer and provide insights.
    
    Args:
        question: Question that was asked
        answer: Student's answer
        context: Presentation context
        model: LLM model (loads if None)
    
    Returns:
        dict with analysis: 'quality', 'technical_depth', 'clarity', 'understanding', 'feedback'
    """
    if model is None:
        model = get_llm_model()
    
    if model is None:
        return _analyze_answer_fallback(question, answer, context)
    
    # Get analysis prompt
    prompt_template = get_analysis_prompt(
        question=question,
        answer=answer[:1000],
        context=context[:1000],
        detailed=False
    )

    try:
        prompt = prompt_template
        
        response = model(
            prompt,
            max_tokens=300,  # Longer for analysis
            temperature=0.5,  # Lower temperature for more consistent analysis
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            stop=["\n\n\n"],
            echo=False
        )
        
        analysis_text = response['choices'][0]['text'].strip()
        
        # Parse analysis (simple extraction)
        analysis = _parse_analysis(analysis_text)
        
        return {
            'analysis_text': analysis_text,
            'technical_depth': analysis.get('technical_depth', 5.0),
            'clarity': analysis.get('clarity', 5.0),
            'understanding': analysis.get('understanding', 5.0),
            'feedback': analysis.get('feedback', ''),
            'success': True,
            'error': None
        }
    
    except Exception as e:
        print(f"Error analyzing answer: {e}")
        return _analyze_answer_fallback(question, answer, context)


def _clean_question(question):
    """Clean and format question text."""
    # Remove common prefixes
    prefixes = ["Question:", "Q:", "Q.", "?", "Here's a question:", "A question:"]
    for prefix in prefixes:
        if question.startswith(prefix):
            question = question[len(prefix):].strip()
    
    # Ensure it ends with question mark
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    
    # Remove extra whitespace
    question = ' '.join(question.split())
    
    return question


def _parse_analysis(analysis_text):
    """Parse analysis text into structured format."""
    analysis = {
        'technical_depth': 5.0,
        'clarity': 5.0,
        'understanding': 5.0,
        'feedback': analysis_text
    }
    
    # Simple keyword-based scoring
    text_lower = analysis_text.lower()
    
    # Technical depth indicators
    if any(word in text_lower for word in ['excellent', 'deep', 'thorough', 'comprehensive', 'strong']):
        analysis['technical_depth'] = 8.0
    elif any(word in text_lower for word in ['good', 'solid', 'adequate']):
        analysis['technical_depth'] = 6.5
    elif any(word in text_lower for word in ['basic', 'superficial', 'limited', 'weak']):
        analysis['technical_depth'] = 4.0
    
    # Clarity indicators
    if any(word in text_lower for word in ['clear', 'well-explained', 'concise', 'articulate']):
        analysis['clarity'] = 8.0
    elif any(word in text_lower for word in ['unclear', 'confusing', 'vague', 'unclear']):
        analysis['clarity'] = 4.0
    
    # Understanding indicators
    if any(word in text_lower for word in ['understands', 'grasps', 'comprehends', 'demonstrates']):
        analysis['understanding'] = 8.0
    elif any(word in text_lower for word in ['lacks', 'doesn\'t understand', 'confused']):
        analysis['understanding'] = 4.0
    
    return analysis


# Fallback functions (when LLM is not available)
def _generate_question_fallback(context):
    """Generate question using template-based fallback."""
    # Simple template-based questions
    templates = [
        "Can you explain how you implemented {feature}?",
        "What challenges did you face while building this?",
        "How does {component} work in your project?",
        "What technologies did you use and why?",
        "Can you walk me through your architecture?",
        "How did you handle {aspect} in your implementation?",
    ]
    
    import random
    template = random.choice(templates)
    
    # Try to extract keywords from context
    keywords = _extract_keywords_from_context(context)
    
    if keywords:
        feature = keywords[0] if keywords else "this feature"
        question = template.format(feature=feature, component=feature, aspect=feature)
    else:
        question = "Can you tell me more about your project?"
    
    return {
        'question': question,
        'success': False,
        'error': 'LLM not available, using template fallback'
    }


def _generate_followup_fallback(previous_question, answer, context):
    """Generate follow-up using template-based fallback."""
    templates = [
        "Can you elaborate on that?",
        "How did you approach that problem?",
        "What alternatives did you consider?",
        "Can you give me a specific example?",
        "Why did you choose that approach?",
    ]
    
    import random
    question = random.choice(templates)
    
    return {
        'question': question,
        'success': False,
        'error': 'LLM not available, using template fallback'
    }


def _analyze_answer_fallback(question, answer, context):
    """Analyze answer using simple heuristics."""
    # Simple scoring based on answer length and keywords
    answer_lower = answer.lower()
    
    # Technical keywords boost score
    tech_keywords = ['algorithm', 'architecture', 'design', 'implementation', 'optimization', 
                     'scalability', 'performance', 'security', 'database', 'api']
    tech_score = sum(1 for kw in tech_keywords if kw in answer_lower) * 0.5
    tech_score = min(10.0, 5.0 + tech_score)
    
    # Length-based clarity (longer answers might be clearer, but not always)
    length_score = min(10.0, 5.0 + (len(answer.split()) / 20))
    
    # Understanding (simple heuristic)
    understanding_score = (tech_score + length_score) / 2
    
    return {
        'analysis_text': f"Answer analyzed using fallback method. Technical depth: {tech_score:.1f}, Clarity: {length_score:.1f}",
        'technical_depth': tech_score,
        'clarity': length_score,
        'understanding': understanding_score,
        'feedback': 'Analysis performed using fallback method (LLM not available)',
        'success': False,
        'error': 'LLM not available, using heuristic fallback'
    }


def _extract_keywords_from_context(context):
    """Extract keywords from context for fallback question generation."""
    import re
    
    # Common technical terms
    tech_terms = [
        'api', 'database', 'server', 'client', 'framework', 'algorithm',
        'architecture', 'component', 'module', 'function', 'class', 'method'
    ]
    
    context_lower = context.lower()
    found = []
    
    for term in tech_terms:
        if term in context_lower:
            found.append(term)
    
    return found[:3]  # Return top 3


# Test function
def test_inference():
    """Test LLM inference functionality."""
    print("Testing LLM inference...")
    
    # Test context
    context = """
    [SCREEN CONTENT]
    def calculate_sum(a, b):
        return a + b
    
    [SPEECH]
    This is my Python project that implements a calculator using functions.
    """
    
    print("\n1. Testing question generation:")
    result = generate_question(context)
    print(f"  Question: {result['question']}")
    print(f"  Success: {result['success']}")
    if result.get('error'):
        print(f"  Note: {result['error']}")
    
    print("\n2. Testing follow-up question:")
    followup = generate_followup_question(
        "What is your project about?",
        "It's a calculator application",
        context
    )
    print(f"  Follow-up: {followup['question']}")
    print(f"  Success: {followup['success']}")
    
    print("\n3. Testing answer analysis:")
    analysis = analyze_answer(
        "How does your calculator work?",
        "It uses functions to perform mathematical operations like addition and subtraction",
        context
    )
    print(f"  Technical Depth: {analysis['technical_depth']:.1f}")
    print(f"  Clarity: {analysis['clarity']:.1f}")
    print(f"  Understanding: {analysis['understanding']:.1f}")
    print(f"  Feedback: {analysis['feedback'][:100]}...")
    
    return True


if __name__ == "__main__":
    test_inference()
