"""
Prompt templates for LLM inference.
Contains all prompts for question generation, follow-ups, and analysis.
"""

# ============================================================================
# QUESTION GENERATION PROMPTS
# ============================================================================

QUESTION_GENERATION_PROMPT = """You are an AI interviewer conducting a technical interview for a student presenting their project.

Based on the following context from the student's presentation (screen content and speech), generate ONE insightful technical question that:
1. Tests their understanding of the implementation
2. Probes deeper into technical decisions
3. Is relevant to what they're currently showing/explaining
4. Is clear and concise (one sentence)

Context:
{context}

Generate only the question, nothing else:"""


QUESTION_GENERATION_PROMPT_DETAILED = """You are an experienced technical interviewer evaluating a student's project presentation.

The student is presenting their project. Based on the screen content and their speech, generate a technical question that:
- Tests deep understanding, not just surface knowledge
- Is specific to what they're currently discussing
- Challenges them to explain their design choices
- Is appropriate for their level (don't be too advanced or too basic)
- Is one clear, concise sentence

Presentation Context:
{context}

Generate only the question:"""


QUESTION_GENERATION_PROMPT_CODE_FOCUSED = """You are a technical interviewer. The student is showing code or technical implementation.

Based on the code and their explanation, ask ONE question that:
1. Tests if they understand how their code works
2. Probes their understanding of algorithms/data structures used
3. Asks about edge cases or error handling
4. Is specific to the code they're showing

Code Context:
{context}

Generate only the question:"""


# ============================================================================
# FOLLOW-UP QUESTION PROMPTS
# ============================================================================

FOLLOWUP_QUESTION_PROMPT = """You are an AI interviewer. The student just answered a question. Generate a follow-up question that:
1. Probes deeper into their answer
2. Tests if they truly understand the concept
3. Is relevant to their previous response
4. Is clear and concise (one sentence)

Previous Question: {previous_question}
Student's Answer: {answer}
Current Context: {context}

Generate only the follow-up question, nothing else:"""


FOLLOWUP_QUESTION_PROMPT_DETAILED = """You are conducting a technical interview. The student answered your question. Now ask a follow-up that:

- Digs deeper into their explanation
- Tests if they understand the "why" behind their answer
- Challenges them to think critically
- Is one clear sentence

Previous Question: {previous_question}
Student's Answer: {answer}
Current Presentation Context: {context}

Generate only the follow-up question:"""


FOLLOWUP_QUESTION_PROMPT_CLARIFICATION = """The student's answer was unclear or incomplete. Ask a clarifying follow-up question.

Previous Question: {previous_question}
Student's Answer: {answer}

Generate a clarifying question that helps them explain better:"""


# ============================================================================
# ANSWER ANALYSIS PROMPTS
# ============================================================================

ANSWER_ANALYSIS_PROMPT = """Analyze the student's answer to this technical interview question.

Question: {question}
Answer: {answer}
Context: {context}

Provide a brief analysis (2-3 sentences) evaluating:
1. Technical depth (how well they understand the technical aspects)
2. Clarity (how clearly they explained)
3. Understanding (how well they grasp the concept)

Format your response as:
Technical Depth: [brief assessment]
Clarity: [brief assessment]
Understanding: [brief assessment]
Overall Feedback: [brief summary]"""


ANSWER_ANALYSIS_PROMPT_DETAILED = """You are evaluating a student's answer in a technical interview.

Question: {question}
Student's Answer: {answer}
Presentation Context: {context}

Evaluate the answer on these criteria (1-10 scale, where 10 is excellent):
1. Technical Depth: How deep is their technical understanding?
2. Clarity: How clearly did they explain their answer?
3. Originality: How creative/unique is their approach?
4. Understanding: Do they truly grasp the concept?

Provide scores and brief justification for each:
Technical Depth (1-10): [score] - [justification]
Clarity (1-10): [score] - [justification]
Originality (1-10): [score] - [justification]
Understanding (1-10): [score] - [justification]
Overall Feedback: [2-3 sentence summary]"""


ANSWER_ANALYSIS_PROMPT_SIMPLE = """Evaluate this answer briefly:

Question: {question}
Answer: {answer}

Rate on scale 1-10:
- Technical Depth: [score]
- Clarity: [score]
- Understanding: [score]

Brief feedback: [1-2 sentences]"""


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

FINAL_EVALUATION_PROMPT = """You are providing final evaluation for a student's project presentation interview.

The student presented: {project_summary}
Questions asked: {questions}
Answers given: {answers}

Provide a comprehensive evaluation covering:
1. Technical Depth: How well do they understand the technical aspects?
2. Clarity: How clearly did they explain their project?
3. Originality: How creative/unique is their approach?
4. Understanding: Do they truly understand their implementation?

For each criterion, provide:
- Score (1-10)
- Brief justification
- Specific examples from their presentation

Format:
=== TECHNICAL DEPTH ===
Score: [1-10]
Justification: [2-3 sentences]
Examples: [specific examples]

=== CLARITY ===
Score: [1-10]
Justification: [2-3 sentences]
Examples: [specific examples]

=== ORIGINALITY ===
Score: [1-10]
Justification: [2-3 sentences]
Examples: [specific examples]

=== UNDERSTANDING ===
Score: [1-10]
Justification: [2-3 sentences]
Examples: [specific examples]

=== OVERALL FEEDBACK ===
[3-4 sentence summary with strengths and areas for improvement]"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_question_prompt(context, focus="general"):
    """
    Get appropriate question generation prompt.
    
    Args:
        context: Presentation context
        focus: "general", "detailed", or "code"
    
    Returns:
        Formatted prompt string
    """
    if focus == "code":
        return QUESTION_GENERATION_PROMPT_CODE_FOCUSED.format(context=context)
    elif focus == "detailed":
        return QUESTION_GENERATION_PROMPT_DETAILED.format(context=context)
    else:
        return QUESTION_GENERATION_PROMPT.format(context=context)


def get_followup_prompt(previous_question, answer, context, clarification=False):
    """
    Get appropriate follow-up question prompt.
    
    Args:
        previous_question: Previous question
        answer: Student's answer
        context: Current context
        clarification: Whether this is a clarification question
    
    Returns:
        Formatted prompt string
    """
    if clarification:
        return FOLLOWUP_QUESTION_PROMPT_CLARIFICATION.format(
            previous_question=previous_question,
            answer=answer
        )
    else:
        return FOLLOWUP_QUESTION_PROMPT.format(
            previous_question=previous_question,
            answer=answer,
            context=context
        )


def get_analysis_prompt(question, answer, context, detailed=False):
    """
    Get appropriate answer analysis prompt.
    
    Args:
        question: Question asked
        answer: Student's answer
        context: Presentation context
        detailed: Whether to use detailed prompt
    
    Returns:
        Formatted prompt string
    """
    if detailed:
        return ANSWER_ANALYSIS_PROMPT_DETAILED.format(
            question=question,
            answer=answer,
            context=context
        )
    else:
        return ANSWER_ANALYSIS_PROMPT.format(
            question=question,
            answer=answer,
            context=context
        )


def get_evaluation_prompt(project_summary, questions, answers):
    """
    Get final evaluation prompt.
    
    Args:
        project_summary: Summary of the project
        questions: List of questions asked
        answers: List of answers given
    
    Returns:
        Formatted prompt string
    """
    questions_text = "\n".join([f"- {q}" for q in questions])
    answers_text = "\n".join([f"- {a}" for a in answers])
    
    return FINAL_EVALUATION_PROMPT.format(
        project_summary=project_summary,
        questions=questions_text,
        answers=answers_text
    )


# ============================================================================
# PROMPT CONSTANTS FOR QUICK ACCESS
# ============================================================================

# Question types
QUESTION_TYPE_GENERAL = "general"
QUESTION_TYPE_DETAILED = "detailed"
QUESTION_TYPE_CODE = "code"

# Analysis types
ANALYSIS_TYPE_SIMPLE = "simple"
ANALYSIS_TYPE_STANDARD = "standard"
ANALYSIS_TYPE_DETAILED = "detailed"
