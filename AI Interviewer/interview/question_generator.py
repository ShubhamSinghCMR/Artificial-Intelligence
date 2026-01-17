"""
Question generator module for creating interview questions.
Uses LLM inference and maintains question queue.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.inference import generate_question, generate_followup_question
from context.fusion import get_context_for_llm
from interview.trigger_logic import should_ask_question, should_ask_followup


class QuestionGenerator:
    """
    Question generator that maintains state and generates questions.
    """
    
    def __init__(self, state):
        """
        Initialize question generator.
        
        Args:
            state: State dictionary
        """
        self.state = state
        self.question_queue = []
        self.asked_questions = set()  # Track asked questions to avoid duplicates
        self.model = None  # Will be loaded lazily
    
    def generate_initial_question(self, context=None):
        """
        Generate the first question for the interview.
        
        Args:
            context: Optional context string (fetches from state if None)
        
        Returns:
            dict with keys: 'question', 'success', 'error', 'question_id'
        """
        if context is None:
            context = get_context_for_llm(self.state, seconds=60)
        
        if not context or len(context.strip()) < 50:
            return {
                'question': "Can you tell me about your project?",
                'success': False,
                'error': 'Insufficient context, using fallback',
                'question_id': None
            }
        
        # Generate question
        result = generate_question(context, model=self.model)
        
        # Check for duplicates
        question = result['question']
        if self._is_duplicate(question):
            # Try generating again with different prompt
            result = generate_question(context + "\n\nGenerate a different question.", model=self.model)
            question = result['question']
        
        # Add to asked questions
        self.asked_questions.add(question.lower().strip())
        
        return {
            'question': question,
            'success': result['success'],
            'error': result.get('error'),
            'question_id': len(self.state.get('questions', []))
        }
    
    def generate_question_from_context(self, context=None, trigger_type=None):
        """
        Generate a question based on current context.
        
        Args:
            context: Optional context string or dict with 'formatted_context' key
            trigger_type: Type of trigger ('pause', 'topic_change', 'time_based')
        
        Returns:
            dict with question info
        """
        # Handle dict context (from new turn-based flow)
        if isinstance(context, dict):
            context = context.get('formatted_context', '')
        
        if context is None:
            context = get_context_for_llm(self.state, seconds=60)
        
        # Ensure context is a string
        if not isinstance(context, str):
            context = str(context) if context else ''
        
        if not context or len(context.strip()) < 50:
            return None
        
        # Adjust context based on trigger type
        if trigger_type == 'code':
            # Focus on code if code detected
            context = self._enhance_context_for_code(context)
        
        # Generate question
        result = generate_question(context, model=self.model)
        question = result['question']
        
        # Check for duplicates
        if self._is_duplicate(question):
            # Try once more
            result = generate_question(context + "\n\nAsk about a different aspect.", model=self.model)
            question = result['question']
        
        # Add to asked questions
        self.asked_questions.add(question.lower().strip())
        
        return {
            'question': question,
            'success': result['success'],
            'error': result.get('error'),
            'trigger_type': trigger_type,
            'timestamp': time.time()
        }
    
    def generate_followup_question(self, question_id=None):
        """
        Generate a follow-up question for a previous question.
        
        Args:
            question_id: ID of question to follow up on (None = latest)
        
        Returns:
            dict with question info or None
        """
        questions = self.state.get('questions', [])
        if not questions:
            return None
        
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
                return None
        
        # Get answer for this question
        answers = self.state.get('answers', [])
        question_answers = [a for a in answers if a.get('question_id') == question_id]
        
        if not question_answers:
            return None
        
        answer = question_answers[-1].get('answer', '')
        previous_question = question.get('question', '')
        
        # Get current context
        context = get_context_for_llm(self.state, seconds=60)
        
        # Generate follow-up
        result = generate_followup_question(
            previous_question=previous_question,
            answer=answer,
            context=context,
            model=self.model
        )
        
        followup_question = result['question']
        
        # Check for duplicates
        if self._is_duplicate(followup_question):
            result = generate_followup_question(
                previous_question=previous_question,
                answer=answer + " (ask about a different aspect)",
                context=context,
                model=self.model
            )
            followup_question = result['question']
        
        # Add to asked questions
        self.asked_questions.add(followup_question.lower().strip())
        
        return {
            'question': followup_question,
            'success': result['success'],
            'error': result.get('error'),
            'question_id': len(questions),
            'parent_question_id': question_id,
            'is_followup': True,
            'timestamp': time.time()
        }
    
    def add_to_queue(self, question_info):
        """
        Add question to queue.
        
        Args:
            question_info: Question dict from generate functions
        """
        if question_info and question_info.get('question'):
            self.question_queue.append(question_info)
    
    def get_next_question(self):
        """
        Get next question from queue.
        
        Returns:
            Question dict or None if queue empty
        """
        if self.question_queue:
            return self.question_queue.pop(0)
        return None
    
    def queue_size(self):
        """Get current queue size."""
        return len(self.question_queue)
    
    def _is_duplicate(self, question):
        """
        Check if question is duplicate.
        
        Args:
            question: Question string
        
        Returns:
            bool: True if duplicate
        """
        question_lower = question.lower().strip()
        
        # Check against asked questions
        if question_lower in self.asked_questions:
            return True
        
        # Check similarity with recent questions
        recent_questions = self.state.get('questions', [])[-5:]  # Last 5 questions
        for q in recent_questions:
            q_text = q.get('question', '').lower().strip()
            similarity = self._question_similarity(question_lower, q_text)
            if similarity > 0.7:  # 70% similarity threshold
                return True
        
        return False
    
    def _question_similarity(self, q1, q2):
        """
        Calculate similarity between two questions (0.0-1.0).
        
        Args:
            q1: First question
            q2: Second question
        
        Returns:
            float: Similarity score
        """
        if not q1 or not q2:
            return 0.0
        
        if q1 == q2:
            return 1.0
        
        # Check if one contains the other
        if q1 in q2 or q2 in q1:
            return 0.9
        
        # Word overlap
        words1 = set(q1.split())
        words2 = set(q2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _enhance_context_for_code(self, context):
        """Enhance context to focus on code snippets."""
        # Extract code sections
        if "=== CODE SNIPPETS ===" in context:
            code_section = context.split("=== CODE SNIPPETS ===")[1]
            if "===" in code_section:
                code_section = code_section.split("===")[0]
            return f"Focus on this code:\n{code_section}\n\nGenerate a technical question about this code."
        return context


# Convenience functions
def generate_question_for_state(state, context=None, trigger_type=None):
    """
    Generate a question for the given state.
    
    Args:
        state: State dictionary
        context: Optional context (fetches if None)
        trigger_type: Trigger type
    
    Returns:
        Question dict or None
    """
    generator = QuestionGenerator(state)
    return generator.generate_question_from_context(context, trigger_type)


def generate_followup_for_state(state, question_id=None):
    """
    Generate follow-up question for state.
    
    Args:
        state: State dictionary
        question_id: Question ID to follow up on
    
    Returns:
        Question dict or None
    """
    generator = QuestionGenerator(state)
    return generator.generate_followup_question(question_id)


# Test function
def test_question_generator():
    """Test question generator functionality."""
    print("Testing question generator...")
    
    from context.state import create_state, add_context_entry
    from context.fusion import fuse_ocr_and_transcript
    
    # Create state with some context
    state = create_state()
    
    # Add some content
    ocr_result = {'text': 'def calculate(x, y): return x + y', 'confidence': 85.0}
    transcript_result = {'text': 'This is my calculator project', 'language': 'en', 'segments': []}
    
    fused = fuse_ocr_and_transcript(ocr_result, transcript_result, state)
    
    print("\n1. Testing initial question generation:")
    generator = QuestionGenerator(state)
    initial = generator.generate_initial_question()
    print(f"  Question: {initial['question']}")
    print(f"  Success: {initial['success']}")
    
    print("\n2. Testing question from context:")
    question = generator.generate_question_from_context(trigger_type='topic_change')
    if question:
        print(f"  Question: {question['question']}")
        print(f"  Trigger: {question.get('trigger_type')}")
    
    print("\n3. Testing duplicate detection:")
    duplicate = generator._is_duplicate(initial['question'])
    print(f"  Is duplicate: {duplicate}")
    
    different_q = "What programming language did you use?"
    duplicate2 = generator._is_duplicate(different_q)
    print(f"  Different question is duplicate: {duplicate2}")
    
    print("\n4. Testing question queue:")
    generator.add_to_queue(question)
    print(f"  Queue size: {generator.queue_size()}")
    next_q = generator.get_next_question()
    print(f"  Next question: {next_q['question'] if next_q else None}")
    print(f"  Queue size after pop: {generator.queue_size()}")
    
    return True


if __name__ == "__main__":
    test_question_generator()
