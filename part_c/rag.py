"""
Part C: Mini-RAG for Knowledge Base Answering
Uses LangChain with free/open-source tools:
- sentence-transformers for embeddings (free, local)
- FAISS for vector store (free, local)
- Ollama Mistral 7B for generation (free, local)
"""

import os
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama
import json

# Configuration
# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_ARTICLES_DIR = os.path.join(SCRIPT_DIR, "kb_articles")  # Folder with KB articles (relative to script)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, lightweight
LLM_MODEL = "mistral:latest"  # Using local Ollama
QUERIES = [
    "How do I configure automations in Hiver?",
    "Why is CSAT not appearing?"
]


def load_kb_articles(directory: str) -> List:
    """Load KB articles from directory"""
    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"KB articles directory '{directory}' not found!\n"
            f"Please create KB articles in '{directory}' folder first.\n"
            f"Expected format: .txt files with knowledge base content."
        )
    
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    
    if len(documents) == 0:
        raise ValueError(
            f"No .txt files found in '{directory}'!\n"
            f"Please add KB articles as .txt files in the directory."
        )
    
    print(f"Loaded {len(documents)} KB articles from {directory}")
    return documents


def create_embeddings():
    """Create embeddings using free sentence-transformers model"""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}  # Use CPU (free)
    )
    return embeddings


def create_vector_store(documents: List, embeddings) -> FAISS:
    """Create FAISS vector store from documents"""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")
    
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully")
    return vectorstore


def retrieve_relevant_docs(vectorstore: FAISS, query: str, k: int = 3) -> List:
    """Retrieve relevant documents from vector store"""
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def generate_answer(query: str, context_docs: List) -> str:
    """Generate answer using Ollama LLM with retrieved context"""
    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Use the following pieces of context from Hiver's knowledge base to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def calculate_confidence(answer: str, query: str, num_sources: int) -> float:
    """Calculate confidence score based on retrieved documents and answer quality"""
    if not answer or answer.lower().startswith("i don't know") or "error" in answer.lower():
        return 0.0
    
    if num_sources == 0:
        return 0.3
    
    # Higher confidence if we have multiple relevant sources
    base_confidence = 0.6
    
    # Check if answer contains query keywords
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    keyword_overlap = len(query_words.intersection(answer_words)) / len(query_words) if len(query_words) > 0 else 0
    
    confidence = min(0.95, base_confidence + (keyword_overlap * 0.3) + (num_sources * 0.05))
    
    return round(confidence, 2)


def query_rag(vectorstore: FAISS, query: str) -> Dict:
    """Query the RAG system"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Retrieve relevant documents
    source_docs = retrieve_relevant_docs(vectorstore, query, k=3)
    
    # Generate answer
    answer = generate_answer(query, source_docs)
    
    # Get retrieved articles info
    retrieved_articles = []
    for i, doc in enumerate(source_docs, 1):
        source_path = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        retrieved_articles.append({
            'rank': i,
            'source': os.path.basename(source_path),
            'content': content[:200] + "..." if len(content) > 200 else content
        })
    
    # Calculate confidence
    confidence = calculate_confidence(answer, query, len(source_docs))
    
    return {
        'query': query,
        'answer': answer,
        'retrieved_articles': retrieved_articles,
        'confidence': confidence,
        'num_sources': len(source_docs)
    }


def main():
    """Main execution"""
    print("=" * 60)
    print("Part C: Mini-RAG for Knowledge Base Answering")
    print("=" * 60)
    
    # Load KB articles
    documents = load_kb_articles(KB_ARTICLES_DIR)
    
    if len(documents) == 0:
        print("Error: No KB articles found!")
        return
    
    # Create embeddings
    embeddings = create_embeddings()
    
    # Create vector store
    vectorstore = create_vector_store(documents, embeddings)
    
    # Process queries
    results = []
    for query in QUERIES:
        result = query_rag(vectorstore, query)
        results.append(result)
        
        # Display results
        print(f"\n📄 Retrieved Articles ({result['num_sources']}):")
        for article in result['retrieved_articles']:
            print(f"\n  {article['rank']}. Source: {article['source']}")
            print(f"     Preview: {article['content']}")
        
        print(f"\n💡 Generated Answer:")
        print(f"   {result['answer']}")
        
        print(f"\n📊 Confidence Score: {result['confidence']:.2%}")
        print("\n" + "-" * 60)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'rag_results.json')
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()
