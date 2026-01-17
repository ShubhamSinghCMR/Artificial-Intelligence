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
[ Question Generator ]   [ Follow-up Controller ]
        |                        |
        +----------+-------------+
                   v
             [ Evaluation Engine ]
                   |
                   v
           [ Score + Feedback ]
```

### System Flow

1. **Capture Layer**: Simultaneously captures screen frames and audio chunks
2. **Perception Layer**: Processes frames with OCR and audio with STT
3. **Content Analysis**: Extracts code snippets, keywords, and topics
4. **Context Fusion**: Combines screen and speech content into unified context
5. **Interview Engine**: Orchestrates question generation and answer capture
6. **Trigger Logic**: Determines optimal timing for questions (pauses, topic changes)
7. **Question Generation**: Creates context-aware questions using LLM
8. **Answer Processing**: Captures and scores student responses
9. **Evaluation**: Generates comprehensive feedback report

## Tech Stack

### Core Technologies

**LLM (Large Language Model)**
- **Primary**: Meta LLaMA-3 8B Instruct
- **Fallback**: Mistral AI Mistral 7B Instruct
- **Inference Runtime**: llama.cpp (local, CPU-based)
- **Library**: llama-cpp-python

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

3. **Install Tesseract OCR**
   - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to default location: `C:\Program Files\Tesseract-OCR\`
   - The system will auto-detect it

4. **Download LLM Model**
   ```bash
   # Option 1: Use download script
   python download_model.py
   
   # Option 2: Manual download
   # Visit: https://huggingface.co/models?search=llama-3-8b-instruct-gguf
   # Download Q4_K_M quantized model (~4.5GB)
   # Place in: models/llama-3-8b-instruct.gguf
   ```
   
   See `DOWNLOAD_MODEL.md` for detailed instructions.

5. **Verify Installation**
   ```bash
   # Test OCR
   python perception/ocr.py
   
   # Test STT
   python perception/stt.py
   
   # Test LLM
   python llm/loader.py
   ```

### Run Project

**Option 1: GUI (Recommended for Demo)**
```bash
streamlit run app.py
```
Opens web interface in browser with:
- Start/Stop interview controls
- Live transcript display
- Current question display
- Real-time scores
- Final feedback report

**Option 2: Command Line**
```bash
python main.py
```

**What happens:**
1. System validates environment and dependencies
2. Starts screen and audio capture
3. Begins processing presentation content
4. Asks questions automatically based on triggers
5. Captures and scores answers
6. Generates final feedback report on exit (Ctrl+C)

**During Interview:**
- Start presenting your project (screen share + speak)
- System will automatically ask questions
- Answer questions when asked
- Click "Stop Interview" (GUI) or Press `Ctrl+C` (CLI) to end session

**Output:**
- Real-time logs in console
- Detailed log file: `logs/interviewer.log`
- Session data: `logs/session.json`
- Final feedback report (displayed in GUI or printed in CLI)

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
├── models/           # LLM models (place GGUF files here)
├── main.py           # Entry point (CLI)
├── app.py            # Streamlit GUI entry point
└── requirements.txt  # Python dependencies
```

## Requirements

- **Python**: 3.11+
- **Tesseract OCR**: Installed and in PATH
- **LLM Model**: LLaMA-3 8B Instruct GGUF (~4.5GB)
- **Hardware**: 
  - Microphone access
  - Screen capture permissions
  - ~8-10GB RAM (for LLM model)
  - CPU (GPU optional but not required)

## Troubleshooting

- **Tesseract not found**: See installation instructions in error message
- **LLM model not found**: Download model to `models/` directory (see `DOWNLOAD_MODEL.md`)
- **Audio capture fails**: Check microphone permissions and device selection
- **Performance issues**: Reduce OCR/STT processing frequency in `config/settings.py`

## License

This project is developed for hackathon purposes.
