# study_bug
A local RAG application that lets nursing students query their own textbooks in natural language. Answers are grounded in the actual source documents rather than general model knowledge.

## Setup

### 1. Install PyTorch

Install with the correct CUDA version for your system: https://pytorch.org/get-started/locally/

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add environment variables
Create a .env file in the project root:
ANTHROPIC_API_KEY=sk-ant-...

### 4. Add PDFs
Drop PDF files into textbooks/pdf/

## Usage

# ingest documents
python ingest.py --mode ingest

# query
python ingest.py --mode query --query "what is the reversal agent for opioid overdose"