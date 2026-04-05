# Advanced RAG Pipeline

> A high-performance Retrieval-Augmented Generation pipeline for technical Q&A, grounded in multi-stage retrieval theory. Combines hybrid retrieval (dense + BM25), query expansion, Reciprocal Rank Fusion (RRF), and cross-encoder re-ranking to improve retrieval precision and answer grounding. Evaluated with Ragas, showing measurable gains in context recall and faithfulness.

---

## 📊 Performance Comparison

| Metric | Traditional RAG | Advanced RAG | Improvement |
|--------|-----------------|--------------|-------------|
| **Context Recall** | 0.7949 | **0.8653** | +8.86% |
| **Faithfulness** | 0.9158 | **0.9776** | +6.75% |
| **Factual Correctness** | 0.7508 | **0.7508** | Depends |
| **Answer Relevancy** | — | **0.9542** | - |

---

## Traditional RAG vs Advanced RAG

### Traditional RAG Pipeline
```
Query → Embed → Retrieve Top-K → LLM → Answer
```

### Advanced RAG Pipeline
```
Query → Rewrite/Expand → Hybrid Retrieve(dense + sparse(BM25)) → RRF Fusion → Re-rank → LLM → Answer
```

---

## Key Differences

| Component | Traditional RAG | Advanced RAG | Why It Matters |
|-----------|-----------------|--------------|----------------|
| **Retrieval** | Dense embeddings only | Dense + Sparse (Hybrid) | Captures both semantic meaning AND exact keyword matches |
| **Query Strategy** | Single query | Query expansion (4 variations) | Different phrasings retrieve different relevant docs |
| **Fusion** | None | Reciprocal Rank Fusion | Combines strengths of multiple query results |
| **Re-ranking** | None | Cross-encoder re-ranker | Re-scores retrieved docs with precise query-doc similarity |
| **Chunk Size** | 1000 / 300 overlap | 1200 / 400 overlap | Larger chunks = complete ideas, less fragmentation |
| **Retrieved Docs (k)** | 5-10 | 20 → re-rank to 7 | More candidates = better final selection after re-ranking |

---

## Why Each Enhancement Works

### 1. Hybrid Retrieval (Dense + Sparse)

**Problem with Traditional RAG:**
- Dense-only retrieval misses exact keyword matches (acronyms, technical terms)
- "CNN" might not match "Convolutional Neural Network" in embedding space

**Solution:**
```
Dense (semantic)  → finds "neural network" ≈ "ANN"
Sparse (BM25)     → finds exact "backpropagation", "gradient descent"
Combined          → best of both worlds
```

**Impact:** +8.86% Context Recall

---

### 2. Query Expansion + Reciprocal Rank Fusion

**Problem with Traditional RAG:**
- Single query = single perspective
- Users phrase questions differently than documents

**Solution:**
Generate 4 query variations, retrieve from each, then fuse:

```
Original:  "What is backpropagation?"
Variation 1: "How does backpropagation work in neural networks?"
Variation 2: "Explain the backpropagation algorithm for training"
Variation 3: "What is the role of backpropagation in deep learning?"

RRF Score = Σ 1/(60 + rank) for each doc across all 4 result lists
```

**Why RRF works:**
- Docs appearing in multiple result lists get boosted
- Reduces risk of missing relevant docs due to poor phrasing

---

### 3. Why k=20 → Re-rank to k=7 Increases Accuracy

**Traditional RAG:**
```
Retrieve top-5 → Use all 5
Problem: Some of top-5 may be weak matches
```

**Advanced RAG:**
```
Retrieve top-20 → Re-rank → Use top-7
```

**Why this works:**

| Step | What Happens | Why Better |
|------|--------------|------------|
| Retrieve k=20 | Cast wider net | More candidates = less chance of missing relevant docs |
| Re-rank | Cross-encoder scores each doc against query | Bi-encoders (retrieval) are fast but approximate. Cross-encoders are slow but precise |
| Select top-7 | Keep only highest re-scored | More signal, less noise for the LLM |

**Analogy:**
- Traditional RAG = Interview 5 candidates, hire all 5
- Advanced RAG = Interview 20 candidates, re-evaluate carefully, hire top 7

---

### 4. Cross-Encoder Re-ranking

**Retrieval (Bi-encoder):**
```
Query embedding  ─┐
                  ├→ Cosine similarity → Fast but approximate
Doc embedding    ─┘
```

**Re-ranking (Cross-encoder):**
```
Query + Doc → [CLS] ... [SEP] ... [SEP] → Similarity score
              └─── Transformer processes both together ────┘
              → Slower but much more accurate
```

**Model used:** `BAAI/bge-reranker-v2-m3`
- Trained on technical/academic content
- Better at understanding ML/DL terminology

**Impact:** Directly improves Faithfulness (+6.75%) by surfacing more relevant context

---

### 5. Larger Chunk Size (1200/400 vs 1000/300)

**Traditional RAG:**
- Smaller chunks → ideas get split across boundaries
- "Forward pass... [cut] ...loss calculation" becomes two incomplete chunks

**Advanced RAG:**
- 1200 chars with 400 overlap → complete technical explanations stay together
- More context per chunk → LLM gets fuller picture

---

## Architecture Diagram

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│ Query Expansion (LLM)           │
│ Generates 4 variations          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Hybrid Retrieval (k=20 each)    │
│ Dense: text-embedding-3-large   │
│ Sparse: BM25                    │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Reciprocal Rank Fusion          │
│ Merges 4 result lists → 1 list  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Cross-Encoder Re-ranker         │
│ Re-scores all docs precisely    │
│ Selects top-7                   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ LLM (gpt-4.1-mini)              │
│ Generates answer from context   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

---

## Setup

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=your-key
```

---

## Usage

```python
result = advanced_rag_query(
    query="What is backpropagation?",
    k=20,              # Retrieve 20 docs per query variation
    rerank_top_k=7,    # Re-rank and keep top 7
    use_rewrite=True,  # Enable query expansion
    use_rrf=True       # Enable RRF fusion
)
```

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|------------------|
| **Context Recall** | How much of the ground truth was found in retrieved docs |
| **Faithfulness** | Is the answer grounded in context (no hallucination)? |
| **Factual Correctness** | Does the answer match ground truth facts? |
| **Answer Relevancy** | How directly does the answer address the query? |

---

## License

MIT
