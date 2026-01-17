# AI Interviewer

An intelligent AI system that listens to students presenting their projects (screen share + speech) and conducts adaptive interviews based on content and responses in real-time.

## About Project

This system automatically:
- **Captures** screen content and audio during project presentations
- **Extracts** text from screens using OCR and transcribes speech using STT
- **Analyzes** UI, code snippets, slides, and diagrams
- **Generates** context-aware questions from extracted content
- **Asks** follow-up questions based on responses and screen content
- **Evaluates** and provides feedback on:
  - Technical depth
  - Clarity of explanation
  - Originality
  - Understanding of implementation

### Key Features

- **Real-time Processing**: Captures and processes screen + audio simultaneously
- **Adaptive Interviewing**: Questions adapt based on presentation content
- **Context-Aware**: Combines visual (screen) and audio (speech) context
- **Performance Optimized**: Change detection, frame sampling, and caching
- **Comprehensive Evaluation**: Multi-criteria scoring with detailed feedback
- **100% Local**: No cloud services, no paid APIs - runs entirely on your machine

## Architecture / System Design

```
[ Screen Capture ]        [ Audio Capture ]
        |                        |
        v                        v
[ Frame Sampler ]         [ Audio Chunker ]
        |                        |
        v                        v
[ Change Detector ]       [ Silence Detector ]
        |                        |
        v                        v
[ OCR Engine ]            [ STT Engine ]
        |                        |
        +----------+-------------+
                   v
            [ Context Fusion Layer ]
                   |
                   v
            [ Interview Engine ]
                   |
        +----------+-------------+
        |                        |
        v                        v
[ Question Generator ]   [ Answer Scorer ]
        |                        |
        +----------+-------------+
                   v
           [ Score + Feedback ]
```

### Interview Flow (Turn-Based)

```
┌─────────────────────────────────────────┐
│ 1. LISTENING STATE                      │
│    • System listens and accumulates     │
│      candidate speech                   │
│    • Real-time transcription displayed  │
│    • Detects 3 seconds of silence       │
└──────────────┬──────────────────────────┘
               │
               v (Silence > 3s)
┌─────────────────────────────────────────┐
│ 2. CANDIDATE STOPPED                    │
│    • Shows full transcription            │
│    • Generates context-aware question   │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│ 3. QUESTION PHASE                       │
│    • Displays: "Question:"               │
│    • Shows interviewer question         │
│    • Prompts: "Now you answer."         │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│ 4. WAITING FOR ANSWER                   │
│    • Accumulates candidate answer       │
│    • Real-time transcription displayed  │
│    • Detects 3 seconds of silence       │
└──────────────┬──────────────────────────┘
               │
               v (Silence > 3s)
┌─────────────────────────────────────────┐
│ 5. ANSWER PROCESSED                     │
│    • Shows complete answer              │
│    • Scores answer (technical, clarity) │
│    • Says: "Please proceed..."           │
└──────────────┬──────────────────────────┘
               │
               v
         (Back to Step 1)
```

### System Flow

1. **Capture Layer**: Simultaneously captures screen frames and audio chunks
2. **Perception Layer**: Processes frames with OCR and audio with STT
3. **Content Analysis**: Extracts code snippets, keywords, and topics
4. **Context Fusion**: Combines screen and speech content into unified context
5. **Turn-Based Interview**: 
   - **LISTENING**: Accumulates candidate speech, detects silence (3s)
   - **QUESTION**: Generates and asks context-aware question
   - **ANSWER**: Captures and processes candidate response
   - **PROCEED**: Prompts candidate to continue
6. **Question Generation**: Creates context-aware questions using LLM based on accumulated speech
7. **Answer Processing**: Captures, transcribes, and scores student responses
8. **Evaluation**: Generates comprehensive feedback report

## Tech Stack

### Core Technologies

**LLM (Large Language Model)**
- **Ollama** - Local LLM runtime (fast setup, no large downloads needed)
- **Recommended Models**: llama3.2 (3B, fast), llama3.1 (8B, better quality), mistral (7B)
- **API**: HTTP-based local API (no complex dependencies)

**Speech-to-Text**
- **OpenAI Whisper** (local, offline)
- Model: Base (balanced speed/accuracy)

**Screen / Vision**
- **Tesseract OCR** - Text extraction from screens
- **OpenCV** - Image processing and screen capture

**Capture**
- **PyAudio** - Audio capture from microphone
- **mss** - Efficient screen capture
- **OpenCV** - Frame processing

**Backend / Orchestration**
- **Python** - Core language
- **Threading** - Multi-threaded capture and processing
- **FastAPI** - API framework (for future web interface)

**State / Logs**
- **In-memory** (Python dictionaries)
- **JSON files** - Session logging

**UI**
- **Streamlit** - Web-based GUI dashboard
- **CLI** - Command-line interface

### Design Principles

- ✅ **100% Local** - No cloud services, no paid APIs
- ✅ **Free & Open Source** - All tools are free
- ✅ **Efficient** - Optimized for performance and low latency
- ✅ **Modular** - Clean separation of concerns
- ✅ **Function-based** - Simple, maintainable code

## Instructions

### Clone Repository

```bash
git clone <repository-url>
cd "AI Interviewer"
```

### Setup

1. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables** (Optional)
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env if needed (only if Ollama is on different URL or Tesseract not auto-detected)
   ```
   
   **Note**: Most users don't need to edit `.env` - defaults work fine. Only edit if:
   - Ollama is running on a different URL/port
   - You want to use a different model name
   - Tesseract is not auto-detected

4. **Install Tesseract OCR**
   - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to default location: `C:\Program Files\Tesseract-OCR\`
   - The system will auto-detect it

5. **Install and Setup Ollama** (Much faster than downloading GGUF files!)
   ```bash
   # Step 1: Download and install Ollama
   # Visit: https://ollama.com/download
   # Download and install Ollama for your OS
   
   # Step 2: Start Ollama (runs in background automatically)
   # On Windows: Just install and it starts automatically
   # On Linux/Mac: Run 'ollama serve' in terminal
   
   # Step 3: Pull a model (choose one):
   ollama pull llama3.2      # Fast, 3B model (~2GB, recommended for quick setup)
   # OR
   ollama pull llama3.1      # Better quality, 8B model (~4.7GB)
   # OR
   ollama pull mistral       # Alternative 7B model (~4.1GB)
   ```
   
   **Note**: Model download happens automatically when you run `ollama pull`. 
   First pull may take 5-15 minutes depending on your internet speed.
   After that, models are cached locally.

6. **Verify Installation**
   ```bash
   # Test OCR
   python perception/ocr.py
   
   # Test STT
   python perception/stt.py
   
   # Test LLM
   python llm/loader.py
   ```

### Run Project

#### Option 1: GUI (Recommended for Demo)

**Start the Streamlit interface:**
```bash
streamlit run app.py
```

**What happens:**
- Opens web interface in your browser (usually `http://localhost:8501`)
- Shows welcome screen with instructions

**Using the GUI:**
1. Click **"🚀 Start Interview"** button in the sidebar
2. Begin presenting your project (screen share + speak)
3. Watch live updates:
   - **Live Transcript**: Your speech appears in real-time
   - **Current Question**: Questions appear automatically
   - **Answers & Scores**: Your answers are scored as you respond
4. Click **"⏹️ Stop Interview"** when done
5. View final report with scores and feedback
6. Download report if needed

**Features:**
- Real-time transcript display
- Current question display
- Live score updates
- Final feedback report with charts
- Downloadable report

---

#### Option 2: Command Line Interface

**Start the CLI:**
```bash
python main.py
```

**What happens:**
1. System validates environment and dependencies
2. Shows startup banner and status
3. Starts screen and audio capture
4. Begins processing presentation content
5. Asks questions automatically based on triggers
6. Captures and scores answers
7. Generates final feedback report on exit

**During Interview:**
- Start presenting your project (screen share + speak)
- System will automatically ask questions
- Answer questions when asked (speak clearly)
- Press `Ctrl+C` to end session and view report

**Output:**
- Real-time logs in console
- Detailed log file: `logs/interviewer.log`
- Session data: `logs/session.json`
- Final feedback report printed at end

---

### Execution Flow

**Typical Interview Session:**

1. **Startup** (5-10 seconds)
   - Environment validation
   - Component initialization
   - Capture devices ready

2. **Presentation Phase** (ongoing)
   - Screen capture running (2 FPS)
   - Audio capture running (16kHz)
   - OCR processing every 3 seconds
   - STT processing every 5 seconds
   - Context fusion and analysis

3. **Question Phase** (automatic)
   - First question after ~10 seconds
   - Questions every 30+ seconds
   - Follow-up questions based on answers
   - Maximum 10 questions per session

4. **Answer Phase** (automatic)
   - Speech captured after question
   - Answer detected after 3 seconds of silence
   - Automatic scoring (technical depth, clarity, etc.)
   - Scores displayed in real-time

5. **Completion**
   - Press `Ctrl+C` (CLI) or click "Stop" (GUI)
   - Final report generated
   - Scores and feedback displayed
   - Session data saved to `logs/session.json`

## Project Structure

```
AI Interviewer/
├── capture/          # Screen and audio capture
│   ├── screen_capture.py
│   ├── audio_capture.py
│   └── change_detector.py
├── perception/       # OCR and STT processing
│   ├── ocr.py
│   ├── stt.py
│   └── content_parser.py
├── context/          # Context fusion and state management
│   ├── fusion.py
│   └── state.py
├── interview/        # Interview orchestration
│   ├── engine.py
│   ├── question_generator.py
│   └── trigger_logic.py
├── evaluation/       # Scoring and feedback
│   ├── scorer.py
│   └── feedback.py
├── llm/              # LLM inference
│   ├── loader.py
│   └── inference.py
├── config/           # Configuration
│   ├── settings.py
│   ├── prompts.py
│   └── logging_config.py
├── logs/             # Session logs
├── models/           # (Not used with Ollama - models managed by Ollama)
├── main.py           # Entry point (CLI)
├── app.py            # Streamlit GUI entry point
└── requirements.txt  # Python dependencies
```

## Requirements

- **Python**: 3.11+
- **Tesseract OCR**: Installed and in PATH
- **Ollama**: Installed and running (download from https://ollama.com/download)
- **LLM Model**: Any Ollama model (llama3.2 recommended, ~2GB)
- **Hardware**: 
  - Microphone access
  - Screen capture permissions
  - ~8-10GB RAM (for LLM model)
  - CPU (GPU optional but not required)

## Troubleshooting

### Common Issues

**Tesseract not found**
- Windows: Install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Set `TESSERACT_PATH` in `.env` if not auto-detected

**Ollama not running**
- Install from: https://ollama.com/download
- Ensure Ollama service is running
- Check: `ollama list` should show your models

**Model not found**
- Run: `ollama pull llama3.2` (or your chosen model)
- Verify: `ollama list` shows the model

**Audio capture fails**
- Check microphone permissions in Windows settings
- Verify microphone is not muted
- Set `AUDIO_DEVICE_INDEX` in `.env` if using specific device

**Streamlit not opening**
- Ensure Streamlit is installed: `pip install streamlit`
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

**Performance issues**
- Reduce `OCR_PROCESS_INTERVAL` in `.env` (default: 3.0)
- Reduce `STT_CHUNK_DURATION` in `.env` (default: 5.0)
- Use smaller Whisper model: `WHISPER_MODEL=tiny` in `.env`

**No questions being asked**
- Wait at least 10 seconds (initial delay)
- Ensure you're speaking and showing content on screen
- Check logs: `logs/interviewer.log`

## License

This project is developed for hackathon purposes.
