from context.state import create_state
from interview.engine import run_interview

def main():
    print("AI Interviewer started")
    state = create_state()
    run_interview(state)

if __name__ == "__main__":
    main()
