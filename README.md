# 📚 Cariin — Semantic Book Recommender

An LLM-powered book recommendation app: describe the kind of book you want in plain English and get semantically-matched recommendations, filterable by **category** and **emotional tone**.

<h3>Dashboard</h3>

![Cariin dashboard](https://github.com/user-attachments/assets/5f4611d0-d527-4eeb-a311-b5b9eddcb45a)

---

## ✨ Features

- **Semantic search** over book descriptions using sentence-embeddings + a vector store (no keyword matching).
- **Emotion-aware ranking** — books tagged by emotional tone (joy, sadness, suspense, …) so you can filter by the *feeling* you want.
- **Zero-shot category classification** to organize titles into genres.
- Interactive **Gradio** dashboard with cover art and one-line summaries.

## 🧠 How it works

```
Book descriptions ──► HuggingFace embeddings ──► Chroma / FAISS vector store
                                                          │
User query "a hopeful sci-fi about second chances" ──► similarity search
                                                          │
                          emotion + category filters ──► ranked recommendations
```

| Stage | Notebook |
| --- | --- |
| Data cleaning & prep | `model/Data-prepocessing.ipynb` |
| Emotion tagging | `model/sentiment-analysis.ipynb` |
| Genre classification | `model/text-classification.ipynb` |
| Vector search | `model/vector_search.ipynb` |

## 🚀 Run the dashboard

```bash
git clone https://github.com/4arkeinlvy/Cariin.git
cd Cariin
pip install -r requirements.txt
# add your key to .env:  OPENAI_API_KEY=...
python Dashbord/app.py
```

## 🛠️ Tech stack

LangChain · Hugging Face Embeddings · Chroma · FAISS · Sentence-Transformers · Gradio · Pandas · OpenAI
