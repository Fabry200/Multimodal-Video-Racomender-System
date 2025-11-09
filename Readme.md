# 🎥 Local Multimodal Video Recommender System

A fully local, privacy-preserving **video recommender system + natural language semantic search engine** that understands video content using:

- Computer Vision (BLIP image captioning + OCR)
- Audio transcription (Whisper ASR)
- LLM reasoning (LLaMA via Ollama)
- Semantic search (ChromaDB + cosine similarity)

The system recommends videos based on what the user *actually watches*, and allows natural language queries such as:

> "Find the videos where someone is cooking."

All processing runs locally — **no external API calls**.

---

## ✅ What it Does

For each video the user watches:

1. Extracts representative frames (OpenCV)
2. Generates captions + reads on-screen text (BLIP + OCR)
3. Transcribes audio (Whisper)
4. Summarizes the meaning using LLaMA (via Ollama)
5. Generates an embedding from the meaning (Gemma Embedding model)
6. Stores the embedding + user preference score in ChromaDB
7. Recommends new unseen videos based on similarity + watch-time score

Additionally, the system includes an **interactive terminal menu**, where users can:

- 📥 Add videos by “watching” them
- 🎯 Receive personalized recommendations
- 🔍 Query watched videos using natural language (vector search over ChromaDB)
- 📊 View user statistics (average watch time, std. deviation, etc.)

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Frame extraction | OpenCV |
| Image captioning | BLIP (HuggingFace) |
| OCR (detect text in video frames) | Tesseract |
| Audio → text | Whisper (transformers pipeline) |
| Video meaning summarization | LLaMA via Ollama |
| Embeddings | EmbeddingGemma (Ollama) |
| Vector database | ChromaDB |
| Recommendation scoring | Cosine similarity + Softmax + user scoring |

---

## 🔧 Installation

Clone the repo:

```bash
git clone https://github.com/<your-user>/<repo-name>.git
cd <repo-name>
