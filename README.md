# 🧠 Simple RAG (Retrieval-Augmented Generation) System

This project is a lightweight implementation of a Retrieval-Augmented Generation (RAG) system using LangChain, HuggingFace Transformers, and FAISS. It lets you load and embed documents into a vector store and then query them using a language model.

---

## 📁 Project Structure
<pre><code>## 📁 Project Structure ``` project-root/ ├── docs/ │ └── my_text.txt # Input text file (your knowledge base) ├── faiss_store/ # (Auto-generated) Vector index from FAISS ├── rag.py # Main script for embedding and querying ├── requirements.txt # Python dependencies └── README.md # Project overview and usage instructions ``` </code></pre>

## 🚀 Setup Instructions

1. **Clone the repository** (or download the files)
2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
📝 Usage
1. Embed Documents
Make sure your text file is saved in docs/my_text.txt.

Then run:

python rag.py embed
This will:

Load and chunk your document

Create embeddings

Store them in a local FAISS index (faiss_store/)

2. Query the RAG System
Run:

python rag.py query
Then type your question when prompted. The system will:

Load the FAISS index

Retrieve the most relevant chunks

Generate an answer using the google/flan-t5-base model

🤖 LLM & Embeddings
LLM: google/flan-t5-base

Embeddings: sentence-transformers/all-MiniLM-L6-v2

📦 Notes
FAISS index is saved locally at faiss_store/

You can change the text source by modifying docs/my_text.txt

Feel free to swap out the language model or embedding model as needed.

📃 License
MIT License

💡 Credits
LangChain

HuggingFace Transformers


FAISS by Facebook AI



