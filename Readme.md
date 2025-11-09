# 🎥 Local Multimodal Video Recommender System

A fully local, privacy-preserving **video recommender system** that understands video content using:

- Computer Vision (BLIP image captioning + OCR)
- Audio transcription (Whisper ASR)
- LLM reasoning (LLaMA via Ollama)
- Semantic search (ChromaDB + cosine similarity)

The system recommends videos based on what the user *actually watches*, not titles or tags.

---

## ✅ What it Does

For each video the user watches:

1. Extracts frames using OpenCV  
2. Generates captions + detects text in frame (BLIP + Tesseract OCR)  
3. Transcribes audio (Whisper)  
4. Uses an LLM to summarize the meaning of the video (LLaMA via Ollama)
5. Converts the summary into a semantic vector embedding (Gemma embedding model)
6. Stores embeddings + user preference score in **ChromaDB**
7. Recommends new videos using cosine similarity + user scoring

All models run **locally**, no external APIs.

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Frame extraction | OpenCV |
| Image captioning | BLIP (HuggingFace) |
| OCR (detect text in video frames) | Pytesseract |
| Audio → text | Whisper (transformers pipeline) |
| Video meaning summarization | LLaMA (Ollama) |
| Embeddings | EmbeddingGemma (Ollama) |
| Vector database | ChromaDB |
| Recommendation scoring | Cosine similarity + engagement scoring |

---

## 🔧 Installation

Clone the repo:

```bash
git clone https://github.com/<your-user>/<repo-name>.git
cd <repo-name>
