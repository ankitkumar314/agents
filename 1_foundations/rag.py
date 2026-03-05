import re
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class RAGPipeline:
    """
    General-purpose RAG pipeline using semantic similarity chunking.
    Works with any document - no hardcoded section markers needed.
    Uses ChromaDB's built-in MiniLM embedding model (runs locally, no API).

    Usage:
        rag = RAGPipeline()
        rag.initialize(linkedin_text, summary_text)
        context = rag.retrieve_context("do you hold a patent?")
    """

    def __init__(self, collection_name="resume_rag", similarity_threshold=0.5, min_chunk_size=2):
        """
        Args:
            collection_name: Name for the ChromaDB collection.
            similarity_threshold: Cosine similarity threshold for splitting (0.0-1.0).
                Lower = fewer, larger chunks. Higher = more, smaller chunks.
            min_chunk_size: Minimum number of sentences per chunk.
        """
        self._collection = None
        self._embed_fn = None
        self._collection_name = collection_name
        self._similarity_threshold = similarity_threshold
        self._min_chunk_size = min_chunk_size

    def _get_embed_fn(self):
        """Lazy-load the embedding function (same MiniLM model ChromaDB uses)."""
        try:
            if self._embed_fn is None:
                self._embed_fn = DefaultEmbeddingFunction()
            return self._embed_fn
        except Exception as e:
            print(f"Error loading embedding function: {e}")
            return None

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        """Compute cosine similarity between two vectors."""
        try:
            a = np.array(vec_a)
            b = np.array(vec_b)
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm == 0:
                return 0.0
            return float(dot / norm)
        except Exception:
            return 0.0

    @staticmethod
    def _split_into_sentences(text):
        """
        Split text into sentences/lines.
        Handles page markers, newlines, and merges very short lines.
        """
        try:
            if not text:
                return []
            # Clean page markers like 'Page 1 of 5'
            text = re.sub(r'\s*Page \d+ of \d+\s*', '\n', text)
            # Split on newlines (resume text is line-based)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Merge very short lines (< 5 chars) with the previous line
            merged = []
            for line in lines:
                if merged and len(line) < 5:
                    merged[-1] = merged[-1] + " " + line
                else:
                    merged.append(line)
            return merged
        except Exception as e:
            print(f"Error splitting sentences: {e}")
            return []

    def _semantic_similarity_chunk(self, text):
        """
        General-purpose semantic similarity chunking.

        Algorithm:
            1. Split text into sentences/lines
            2. Embed each sentence using MiniLM (local, no API)
            3. Compute cosine similarity between consecutive sentences
            4. When similarity drops below threshold -> new chunk
            5. Group consecutive similar sentences into one chunk

        Args:
            text: The input text to chunk.

        Returns:
            List of chunk strings.
        """
        try:
            if not text:
                return []

            embed_fn = self._get_embed_fn()
            if embed_fn is None:
                # Fallback: paragraph-based splitting
                return [p.strip() for p in text.split('\n\n') if p.strip()]

            sentences = self._split_into_sentences(text)
            if len(sentences) <= 1:
                return [text.strip()] if text.strip() else []

            # Step 1: Embed all sentences in one batch (efficient)
            embeddings = embed_fn(sentences)

            # Step 2: Compute cosine similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)

            # Step 3: Find split points where similarity drops below threshold
            split_indices = []
            for i, sim in enumerate(similarities):
                if sim < self._similarity_threshold:
                    split_indices.append(i + 1)

            # Step 4: Group sentences into chunks based on split points
            chunks = []
            prev_idx = 0
            for split_idx in split_indices:
                chunk_sentences = sentences[prev_idx:split_idx]
                if len(chunk_sentences) >= self._min_chunk_size:
                    chunks.append('\n'.join(chunk_sentences))
                elif chunks:
                    # Merge small chunks with the previous one
                    chunks[-1] = chunks[-1] + '\n' + '\n'.join(chunk_sentences)
                else:
                    chunks.append('\n'.join(chunk_sentences))
                prev_idx = split_idx

            # Add remaining sentences as the last chunk
            if prev_idx < len(sentences):
                remaining = '\n'.join(sentences[prev_idx:])
                if chunks and len(sentences[prev_idx:]) < self._min_chunk_size:
                    chunks[-1] = chunks[-1] + '\n' + remaining
                else:
                    chunks.append(remaining)

            return [c.strip() for c in chunks if c.strip()]

        except Exception as e:
            print(f"Error in semantic similarity chunking: {e}")
            return []

    def initialize(self, *documents):
        """
        Initialize the RAG pipeline with one or more documents.
        Each document is semantically chunked and stored in ChromaDB.

        Args:
            *documents: One or more text strings to chunk and index.
                e.g. rag.initialize(linkedin_text, summary_text)

        Returns:
            Number of chunks stored in the vector store.
        """
        try:
            all_chunks = []
            all_ids = []
            all_metadatas = []

            for doc_idx, doc_text in enumerate(documents):
                if not doc_text or not doc_text.strip():
                    continue

                chunks = self._semantic_similarity_chunk(doc_text)
                for chunk_idx, chunk_text in enumerate(chunks):
                    all_chunks.append(chunk_text)
                    all_ids.append(f"doc{doc_idx}_chunk_{chunk_idx}")
                    all_metadatas.append({"source": f"doc_{doc_idx}", "chunk_index": chunk_idx})

            if not all_chunks:
                print("No chunks generated from provided documents.")
                return 0

            chroma_client = chromadb.Client()

            try:
                chroma_client.delete_collection(self._collection_name)
            except Exception:
                pass

            self._collection = chroma_client.create_collection(name=self._collection_name)

            self._collection.add(
                documents=all_chunks,
                ids=all_ids,
                metadatas=all_metadatas
            )

            print(f"RAG initialized: {self._collection.count()} semantic chunks stored")
            print(f"Similarity threshold: {self._similarity_threshold}")
            for i, chunk in enumerate(all_chunks):
                preview = chunk[:100].replace('\n', ' ')
                print(f"  [{i}] {preview}...")
            return self._collection.count()

        except Exception as e:
            print(f"Error initializing RAG: {e}")
            return 0

    def retrieve_context(self, query, top_k=3):
        """
        Retrieve the most relevant chunks for a user query.

        Args:
            query: The user's question/message.
            top_k: Number of top chunks to retrieve.

        Returns:
            A string of the most relevant chunks joined by separators.
        """
        try:
            if self._collection is None:
                raise ValueError("RAG not initialized. Call initialize() first.")

            results = self._collection.query(
                query_texts=[query],
                n_results=top_k
            )

            if results and results["documents"]:
                retrieved_text = "\n\n---\n\n".join(results["documents"][0])
                return retrieved_text
            return ""

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
