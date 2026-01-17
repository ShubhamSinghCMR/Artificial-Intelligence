def create_state():
    return {
        "is_active": True,

        # inputs
        "screen_text": "",
        "transcript": "",

        # interview
        "questions": [],
        "answers": [],

        # control
        "last_question_time": 0,
        "question_count": 0,

        # evaluation
        "scores": {},
    }
