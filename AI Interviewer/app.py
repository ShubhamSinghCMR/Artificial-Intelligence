"""
Streamlit GUI for AI Interviewer
Simple dashboard for starting interview and viewing results.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import time
import threading

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from context.state import create_state, get_session_stats
from interview.engine import InterviewEngine
from evaluation.scorer import get_final_score
from evaluation.feedback import generate_final_feedback, format_feedback_text
from config.settings import SESSION_LOG_PATH

# Suppress print statements (Streamlit will handle logging)
import logging
logging.getLogger().setLevel(logging.WARNING)


# Page config
st.set_page_config(
    page_title="AI Interviewer",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'interview_state' not in st.session_state:
    st.session_state.interview_state = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0


def load_session_data():
    """Load session data from log file."""
    try:
        if os.path.exists(SESSION_LOG_PATH):
            with open(SESSION_LOG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return None


def main():
    """Main Streamlit app."""
    st.title("🎤 AI Interviewer")
    st.markdown("Automated interviewer for project presentations")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        if not st.session_state.is_running:
            if st.button("🚀 Start Interview", type="primary", use_container_width=True):
                try:
                    state = create_state()
                    engine = InterviewEngine(state)
                    
                    if engine.start():
                        st.session_state.interview_state = state
                        st.session_state.engine = engine
                        st.session_state.is_running = True
                        st.session_state.last_update = time.time()
                        
                        # Run engine in background thread
                        def run_engine():
                            try:
                                engine.run()
                            except:
                                pass
                        
                        engine_thread = threading.Thread(target=run_engine, daemon=True)
                        engine_thread.start()
                        st.rerun()
                    else:
                        st.error("Failed to start interview. Check audio capture.")
                except Exception as e:
                    st.error(f"Error starting interview: {e}")
        else:
            if st.button("⏹️ Stop Interview", type="secondary", use_container_width=True):
                try:
                    if st.session_state.engine:
                        st.session_state.engine.stop()
                    st.session_state.is_running = False
                    st.session_state.last_update = time.time()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error stopping interview: {e}")
        
        st.divider()
        
        # Status
        st.header("Status")
        if st.session_state.is_running:
            st.success("🟢 Interview Active")
        else:
            st.info("⚪ Interview Inactive")
        
        # Quick stats
        if st.session_state.interview_state:
            stats = get_session_stats(st.session_state.interview_state)
            st.metric("Questions Asked", stats['question_count'])
            st.metric("Answers Received", stats['answer_count'])
            st.metric("Duration", f"{stats['duration_minutes']:.1f} min")
    
    # Main content area
    if st.session_state.is_running and st.session_state.interview_state:
        # Live interview view
        state = st.session_state.interview_state
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📝 Live Transcript")
            transcript_placeholder = st.empty()
            
            # Get latest transcript
            latest_transcript = state.get('latest_transcript')
            if latest_transcript:
                # latest_transcript is a dict with 'content' key
                content = latest_transcript.get('content', {})
                if isinstance(content, dict):
                    transcript_text = content.get('text', 'No speech detected yet...')
                else:
                    transcript_text = str(content) if content else 'No speech detected yet...'
            else:
                transcript_text = "Waiting for speech..."
            
            transcript_placeholder.text_area(
                "Student Speech",
                transcript_text,
                height=200,
                disabled=True,
                key="transcript_display"
            )
        
        with col2:
            st.header("❓ Current Question")
            question_placeholder = st.empty()
            
            questions = state.get('questions', [])
            if questions:
                latest_q = questions[-1]
                question_text = latest_q.get('question', 'No question yet...')
                question_placeholder.text_area(
                    "Interviewer Question",
                    question_text,
                    height=200,
                    disabled=True
                )
            else:
                question_placeholder.text_area(
                    "Interviewer Question",
                    "Waiting for first question...",
                    height=200,
                    disabled=True
                )
        
        # Answers and scores
        st.header("💬 Answers & Scores")
        answers = state.get('answers', [])
        questions = state.get('questions', [])
        
        if answers:
            for i, answer in enumerate(answers[-5:], 1):  # Show last 5
                question_id = answer.get('question_id')
                question_text = "Unknown question"
                for q in questions:
                    if q.get('question_id') == question_id:
                        question_text = q.get('question', '')
                        break
                
                scores = answer.get('metadata', {}).get('scores', {})
                
                with st.expander(f"Q{i}: {question_text[:50]}..."):
                    st.write(f"**Answer:** {answer.get('answer', '')}")
                    if scores:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Technical", f"{scores.get('technical_depth', 0):.1f}")
                        col2.metric("Clarity", f"{scores.get('clarity', 0):.1f}")
                        col3.metric("Originality", f"{scores.get('originality', 0):.1f}")
                        col4.metric("Understanding", f"{scores.get('understanding', 0):.1f}")
        else:
            st.info("No answers yet. Questions will appear here after you answer.")
        
        # Auto-refresh every 3 seconds
        time.sleep(3)
        st.rerun()
    
    elif st.session_state.interview_state and not st.session_state.is_running:
        # Show final report
        st.header("📊 Final Interview Report")
        
        state = st.session_state.interview_state
        final_scores = get_final_score(state)
        feedback = generate_final_feedback(state)
        
        # Scores
        st.subheader("Scores")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Technical Depth", f"{final_scores['technical_depth']:.1f}/10")
        col2.metric("Clarity", f"{final_scores['clarity']:.1f}/10")
        col3.metric("Originality", f"{final_scores['originality']:.1f}/10")
        col4.metric("Understanding", f"{final_scores['understanding']:.1f}/10")
        col5.metric("Overall", f"{final_scores['overall']:.1f}/10", 
                   delta="PASS" if final_scores['passing'] else "NEEDS IMPROVEMENT")
        
        # Feedback
        st.subheader("Feedback")
        feedback_text = format_feedback_text(feedback)
        st.text_area("Detailed Feedback", feedback_text, height=400, disabled=True)
        
        # Download report
        st.download_button(
            "📥 Download Report",
            feedback_text,
            file_name=f"interview_report_{int(time.time())}.txt",
            mime="text/plain"
        )
        
        # Session data
        with st.expander("📋 Session Details"):
            stats = get_session_stats(state)
            st.json({
                "Duration": f"{stats['duration_minutes']:.1f} minutes",
                "Questions": stats['question_count'],
                "Answers": stats['answer_count'],
                "Topics": stats['topics_discussed'],
                "Keywords": stats['keywords_mentioned']
            })
    
    else:
        # Welcome screen
        st.header("Welcome to AI Interviewer")
        st.markdown("""
        This system will:
        - Capture your screen and audio during presentation
        - Extract content using OCR and transcribe speech
        - Generate context-aware questions
        - Evaluate your responses
        - Provide comprehensive feedback
        
        **To start:**
        1. Click "Start Interview" in the sidebar
        2. Begin presenting your project
        3. Answer questions when asked
        4. Click "Stop Interview" when done
        """)
        
        # Show previous session if available
        session_data = load_session_data()
        if session_data:
            st.divider()
            st.subheader("📜 Previous Session")
            if 'final_scores' in session_data:
                scores = session_data['final_scores']
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Technical", f"{scores.get('technical_depth', 0):.1f}")
                col2.metric("Clarity", f"{scores.get('clarity', 0):.1f}")
                col3.metric("Originality", f"{scores.get('originality', 0):.1f}")
                col4.metric("Understanding", f"{scores.get('understanding', 0):.1f}")
                col5.metric("Overall", f"{scores.get('overall', 0):.1f}")


if __name__ == "__main__":
    main()
