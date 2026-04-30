# Advanced RAG Pipeline

> A high-performance Retrieval-Augmented Generation pipeline for technical Q&A, grounded in multi-stage retrieval theory. Combines hybrid retrieval (dense + BM25), query expansion, semantic chunking, Reciprocal Rank Fusion (RRF), and cross-encoder re-ranking to improve retrieval precision and answer grounding. Evaluated with Ragas, showing measurable gains in context recall and faithfulness.

---

## Performance Results

| Metric | Traditional RAG | Advanced RAG |
|--------|-----------------|--------------|
| **Context Recall** | 0.7949 | **0.9861** |
| **Faithfulness** | 0.9158 | **0.9891** |
| **Factual Correctness** | 0.7508 | **0.8200** |
| **Answer Relevancy** | — | **0.9034** |

---

## Pipeline Overview

### Traditional RAG
```
Query → Embed → Retrieve Top-K → LLM → Answer
```

### Advanced RAG
```
Query → Rewrite/Expand → Hybrid Retrieve (dense + sparse BM25) → RRF Fusion → Re-rank → LLM → Answer
```

---

## What Changed and Why

| Component | Traditional RAG | Advanced RAG | Why It Matters |
|-----------|-----------------|--------------|----------------|
| **Retrieval** | Dense embeddings only | Dense + Sparse (Hybrid) | Captures both semantic meaning AND exact keyword matches |
| **Query Strategy** | Single query | Query expansion (4 variations) | Different phrasings retrieve different relevant docs |
| **Fusion** | None | Reciprocal Rank Fusion | Combines strengths of multiple query results |
| **Re-ranking** | None | Cross-encoder re-ranker | Re-scores retrieved docs with precise query-doc similarity |
| **Chunking** | Fixed-size (1000/200) | Semantic chunking | Semantic chunks preserve complete ideas at natural boundaries |
| **Retrieved Docs (k)** | 5 | 20 → re-rank to 5 | More candidates = better final selection after re-ranking |

---

## Why Each Enhancement Works

### 1. Hybrid Retrieval (Dense + Sparse)

#### Dense Retrieval (Semantic Search)

- Converts text into vectors (embeddings)
- Matches based on **meaning**, not exact words

**Example:**
```
Query: "How do machines learn from data?"
Doc:   "Machine learning systems improve using data"
```
→ Match ✅ (same meaning)

**Vector idea:**
```
Query → [0.21, -0.44, 0.78]
Doc   → [0.19, -0.40, 0.80]
```

**Limitation:**
```
Query: "JWT"
Doc:   "JSON Web Token"
```
→ May miss ❌ (no exact keyword)

---

#### Sparse Retrieval (BM25 / Keyword Search)

- Uses **exact word matching**
- Represents text as word-frequency arrays

**Example:**
```
Vocabulary: ["apple", "banana", "cat", "dog"]

Query: "apple banana"
→ [1, 1, 0, 0]

Doc1: "apple apple dog"
→ [2, 0, 0, 1]

Doc2: "banana cat"
→ [0, 1, 1, 0]
```

**Values meaning:**
- `0` → word not present  
- `1` → appears once  
- `2+` → frequency count  

---

#### Why Hybrid?

```
Dense  → captures meaning
Sparse → captures exact keywords
```

**Result:** Better retrieval accuracy (higher recall)

---

### 2. Query Expansion + Reciprocal Rank Fusion

**Problem with Traditional RAG:**
- Single query = single perspective
- Users phrase questions differently than documents

**Solution:**
Generate 4 query variations, retrieve from each, then fuse:

```
Original:    "What is backpropagation?"
Variation 1: "How does backpropagation work in neural networks?"
Variation 2: "Explain the backpropagation algorithm for training"
Variation 3: "What is the role of backpropagation in deep learning?"

RRF Score = Σ 1/(60 + rank) for each doc across all 4 result lists
```

**Why RRF works:**
- Docs appearing in multiple result lists get boosted
- Reduces risk of missing relevant docs due to poor phrasing

---

### 3. Why k=20 → Re-rank to k=5 Increases Accuracy

**Traditional RAG:**
```
Retrieve top-5 → Use all 5
Problem: Some of top-5 may be weak matches
```

**Advanced RAG:**
```
Retrieve top-20 → Re-rank → Use top-5
```

| Step | What Happens | Why Better |
|------|--------------|------------|
| Retrieve k=20 | Cast wider net | More candidates = less chance of missing relevant docs |
| Re-rank | Cross-encoder scores each doc against query | Bi-encoders (retrieval) are fast but approximate. Cross-encoders are slow but precise |
| Select top-5 | Keep only highest re-scored | More signal, less noise for the LLM |

**Analogy:**
- Traditional RAG = Interview 5 candidates, hire all 5
- Advanced RAG = Interview 20 candidates, re-evaluate carefully, hire top 5

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
- Trained on multilingual, multi-domain data including technical/academic content
- Better at understanding nuanced terminology and long-context relevance

**Tradeoff:** Larger model = higher latency (~2-3x slower than lighter models), but delivers significantly more accurate query-doc relevance scores. The focus here was on maximizing retrieval precision over speed.

---

### 5. Semantic Chunking

**Traditional RAG (Fixed-size):**
- Splits text at arbitrary character counts
- Can cut right through a concept mid-sentence
- "Forward pass... [cut] ...loss calculation" becomes two incomplete chunks

**Advanced RAG (Semantic Chunking):**
- Uses `SemanticChunker` with standard deviation breakpoint detection
- Embeds each sentence and detects where meaning shifts significantly
- Splits only at natural semantic boundaries — complete ideas stay together

**Why this improves retrieval:**
- Each chunk represents a coherent, self-contained idea
- LLM receives full context rather than fragments
- Reduces the chance of a relevant chunk being split across two retrieval results

**Tradeoff:** Semantic chunking requires embedding calls during preprocessing (slower than fixed-size), but yields more coherent chunks that improve both retrieval quality and final answer accuracy.

---

### 6. Upgraded Re-ranker Model (`BAAI/bge-reranker-v2-m3`)

Traditional re-rankers like `ms-marco-electra-base` were trained primarily on English web search data, which limits their effectiveness on technical, academic, or multilingual content.

`BAAI/bge-reranker-v2-m3` is trained on multilingual, multi-domain corpora including academic and technical text. This makes it significantly better at:
- Understanding domain-specific terminology
- Scoring long documents against detailed queries
- Handling queries with multiple sub-questions

**Tradeoff:** Heavier model means higher inference latency per re-ranking call, but the improvement in relevance scoring justifies the cost when answer quality is the priority.

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
│ Selects top-5                   │
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
    rerank_top_k=5,    # Re-rank and keep top 5
    use_rrf=True,      # Enable RRF fusion
    n_query_variations=3
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
