# AI Interviewer

An intelligent AI system that listens to students presenting their projects (screen share + speech) and conducts adaptive interviews based on content and responses in real-time.

## Overview

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

## Tech Stack

### Core Technologies

**LLM (Large Language Model)**
- **Primary**: Meta LLaMA-3 8B Instruct
- **Fallback**: Mistral AI Mistral 7B Instruct
- **Inference Runtime**: llama.cpp (local, CPU-based)

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
- **FastAPI** - API framework (for future web interface)
- **Python asyncio** - Asynchronous processing

**State / Logs**
- **In-memory** (Python dictionaries)
- **JSON files** - Session logging

**UI**
- **CLI** - Command-line interface
- **Streamlit** (optional, minimal)

### Design Principles

- ✅ **100% Local** - No cloud services, no paid APIs
- ✅ **Free & Open Source** - All tools are free
- ✅ **Efficient** - Optimized for performance and low latency
- ✅ **Modular** - Clean separation of concerns
- ✅ **Function-based** - Simple, maintainable code

### Architecture

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

## Project Structure

```
AI Interviewer/
├── capture/          # Screen and audio capture
├── perception/       # OCR and STT processing
├── context/          # Context fusion and state management
├── interview/        # Interview orchestration and question generation
├── evaluation/       # Scoring and feedback generation
├── llm/              # LLM inference and model loading
├── config/           # Configuration and prompts
├── logs/             # Session logs
└── main.py           # Entry point
```

## Features

- **Real-time Processing**: Captures and processes screen + audio simultaneously
- **Adaptive Interviewing**: Questions adapt based on presentation content
- **Context-Aware**: Combines visual (screen) and audio (speech) context
- **Performance Optimized**: Change detection, frame sampling, and caching
- **Comprehensive Evaluation**: Multi-criteria scoring with detailed feedback

## Requirements

- Python 3.11+
- Tesseract OCR
- LLaMA-3 8B Instruct GGUF model (or Mistral 7B as fallback)
- Microphone access
- Screen capture permissions

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install Tesseract OCR (see installation instructions)
4. Download LLaMA-3 8B Instruct GGUF model to `models/` directory
5. Run: `python main.py`
