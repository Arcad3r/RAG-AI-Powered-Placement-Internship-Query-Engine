# RAG-AI-Powered-Placement-Internship-Query-Engine

A Retrieval-Augmented Generation system using ColBERTv2 and RAG Fusion to enable natural language querying over BITS Pilani's Placement and Summer Internship Chronicles (2023-24), helping students efficiently access company requirements, CG cutoffs, preparation strategies, and interview experiences through intelligent document retrieval and local LLM-powered responses.

## 🎯 Features

- **Advanced Semantic Search**: ColBERTv2-based retrieval for high-quality, context-aware document matching
- **RAG Fusion**: Multi-query generation and reciprocal rank fusion for enhanced retrieval accuracy
- **Natural Language Queries**: Ask questions in plain English about placements, internships, and company requirements
- **Local LLM Deployment**: Privacy-preserving response generation using Phi-3.5 (3B parameters)
- **Comprehensive Coverage**: Indexes both Placement Chronicles 2023-24 and SI Chronicles 23-24 Sem I

## 📋 Prerequisites

- Python 3.12+
- Anaconda or Miniconda
- Local LLM server (LM Studio or similar) running Phi-3.5
- Sufficient RAM (8GB+ recommended)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd smile-rag-task
```

2. **Create and activate conda environment**
```bash
conda create -n rag-env python=3.12
conda activate rag-env
```

3. **Install required packages**
```bash
pip install ragatouille
pip install langchain langchain-openai
pip install pymupdf  # fitz for PDF processing
pip install torch
```

4. **Set up local LLM server**
- Install LM Studio or similar local LLM server
- Download and load the Phi-3.5:3b model
- Start the server on `http://127.0.0.1:1337/v1`

## 📁 Project Structure

```
.
├── Smile_RAG_TASK.ipynb          # Main notebook
├── Placement Chronicles 2023-24.pdf
├── SI Chronicles 23-24 Sem I.pdf
├── .ragatouille/
│   └── colbert/
│       └── indexes/
│           └── smile_task/       # Generated index files
└── README.md
```

## 🚀 Usage

### Setting Up the System

1. **Initialize the RAG model**
```python
from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
```

2. **Load and index documents**
```python
import fitz

pdf_sections = {
    'Placement Chronicles 2023-24.pdf': '=== Placement Chronicles ===\n',
    'SI Chronicles 23-24 Sem I.pdf': '=== SI Chronicles ===\n'
}

full_text = ""
for path, section_title in pdf_sections.items():
    full_text += section_title
    doc = fitz.open(path)
    for page in doc:
        full_text += page.get_text()
```

3. **Create the index**
```python
index_path = RAG.index(
    collection=[full_text],
    index_name="smile_task",
    max_document_length=180,
    split_documents=True
)
```

### Querying the System

**Example queries:**

```python
# Query 1: Company requirements
question = "What is the CG Cutoff for InfoEdge India Limited?"
response = final_rag_chain.invoke({"question": question})
print(response)
# Output: The CG Cutoff for InfoEdge India Limited is 6.5+

# Query 2: Course recommendations
question = "What courses are important for data analytics roles?"
response = final_rag_chain.invoke({"question": question})

# Query 3: General information
question = "What is data analytics?"
response = final_rag_chain.invoke({"question": question})
```

## 🔧 Technical Architecture

### Components

1. **Document Processing**
   - PDF extraction using PyMuPDF (fitz)
   - Text segmentation with 180-character max chunks
   - Section-based organization

2. **Retrieval System**
   - **Base Retriever**: ColBERTv2 with 2048 partitions
   - **RAG Fusion**: Generates multiple query variations
   - **Reciprocal Rank Fusion**: Combines results from multiple queries
   - Returns top 7 documents per query

3. **Generation Pipeline**
   - **LLM**: Phi-3.5 (3B parameters)
   - **Temperature**: 0 (deterministic outputs)
   - **Context Window**: Includes top retrieved documents
   - **Output Parsing**: Structured response extraction

### RAG Fusion Implementation

```python
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
    ]
    return reranked_results
```

## 📊 Performance Characteristics

- **Index Size**: ~48K embeddings from 390 passages
- **Average Document Length**: 123 tokens
- **Retrieval Speed**: ~1-2 seconds per query
- **Partitions**: 2048 clusters for efficient search
- **Model**: ColBERTv2.0 with late interaction

## 🎓 Use Cases

1. **Company Research**: Query specific company requirements, CG cutoffs, and branch eligibility
2. **Course Planning**: Discover relevant courses and certifications for target roles
3. **Interview Preparation**: Access interview experiences and tips from past students
4. **Profile Building**: Understand what skills and experiences companies value
5. **Analytics Roles**: Get specific guidance on data analytics preparation

## ⚙️ Configuration

### Environment Variables
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-local'  # Required for LangChain compatibility
```

### LLM Configuration
```python
llm = ChatOpenAI(
    model_name="phi-3.5:3b",
    openai_api_base="http://127.0.0.1:1337/v1",
    temperature=0
)
```

### Retrieval Parameters
- `k` (documents per query): 7
- `max_document_length`: 180 characters
- `split_documents`: True
- `use_faiss`: False (using experimental PLAID backend)

## 🔍 Example Outputs

**Query**: "What courses are recommended for data analytics?"

**Retrieved Documents** (top 3):
1. CS F212 - Database Systems (Score: 0.25)
2. MATH F112 - Probability and Statistics (Score: 0.2)
3. ECON F241 - Econometric Methods (Score: 0.15)

**Generated Response**:
"For data analytics roles, the following courses are highly recommended: CS F212 (Database Systems), MATH F112 (Probability and Statistics), and ECON F241 (Econometric Methods). Additionally, SQL proficiency is mandatory, and the Placement Training Module for analytics provides a good foundation."

## 🐛 Troubleshooting

### Common Issues

1. **CUDA warnings**: System works on CPU; warnings can be ignored
```python
warnings.filterwarnings("ignore", 
    message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available")
```

2. **FAISS compatibility**: Using PLAID backend instead
```
Pass use_faiss=True if you want to use FAISS (faster but requires proper installation)
```

3. **LLM connection errors**: Ensure local server is running
```bash
# Check if server is running
curl http://127.0.0.1:1337/v1/models
```

## 📝 Future Enhancements

- [ ] Add support for more recent chronicles
- [ ] Implement conversation history for follow-up questions
- [ ] Add filtering by branch, CG range, or company type
- [ ] Create a web interface for easier access
- [ ] Add analytics on most queried topics
- [ ] Implement caching for frequently asked questions
- [ ] Support for company-wise filtering
- [ ] Integration with live placement updates

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is intended for educational purposes at BITS Pilani.

## 👥 Authors

BITS Pilani Students

## 🙏 Acknowledgments

- BITS Pilani Placement Unit for the chronicles
- RAGatouille and ColBERT teams for the retrieval framework
- LangChain for the orchestration framework
- Microsoft for Phi-3.5 model

---

**Note**: This system is designed to assist students with placement preparation. Always verify critical information from official sources.
