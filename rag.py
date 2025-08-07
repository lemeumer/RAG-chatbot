import sys
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline


def embed_docs():
    print("[📄] Loading and embedding documents...")

    loader = TextLoader("docs/my_text.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local("faiss_store")

    print("[✅] Documents embedded and saved to FAISS index (faiss_store/).")


def query_docs():
    print("[🔍] Loading FAISS index and querying...")

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_store", embedder, allow_dangerous_deserialization=True)

    # Load lightweight LLM
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=pipe)

    question = input("\n🤖 Ask a question: ").strip()
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based on the context below:

{context}

Question: {question}
Answer:"""

    # Use the updated method
    response = llm.invoke(prompt)
    print("\n🧠 Answer:\n" + response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python rag.py [embed | query]")
        sys.exit(1)

    if sys.argv[1] == "embed":
        embed_docs()
    elif sys.argv[1] == "query":
        query_docs()
    else:
        print("❌ Invalid option. Use either 'embed' or 'query'")
