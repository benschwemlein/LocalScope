
# **Local Codebase Search and Query System**

A fully local, privacy preserving system that lets you run natural language queries against large source code repositories.
Everything runs on your Macintosh. No data leaves the machine.

This system does the following:

• Indexes your entire repo using an embedding model
• Stores vectors in a local Chroma database
• Searches by semantic meaning, not keywords
• Feeds the top matching snippets into a local LLM
• Produces an answer based solely on those snippets

---

# **1. System Overview**

The system has two phases.

### **Indexing Phase**

• Scans the repository
• Splits files into small chunks
• Embeds each chunk using nomic-embed-text
• Stores vectors and metadata in ChromaDB
• Creates the folder chroma_repo which contains the vector index

### **Query Phase**

• Takes a natural language question or a bug.txt description
• Embeds the question
• Retrieves the most similar repository chunks
• Sends those chunks plus your question into the LLM
• Gives you a concise answer using only your codebase

This is basically your own local GitHub Copilot, powered by your own computer.

---

# **2. Requirements**

To run the system you need:

• macOS
• Python 3.10 or later
• Ollama installed
• The models nomic-embed-text and llama3.1 (downloaded once)
• chromadb and requests packages installed

Everything runs offline after installation.

---

# **3. Installing Ollama**

1. Visit [https://ollama.com/download](https://ollama.com/download)
2. Download the macOS DMG
3. Drag Ollama into the Applications folder
4. Run Ollama once to initialize the background service

### **Verify installation**

Which ollama
ollama --version

### **Pull required models**

ollama pull nomic-embed-text
ollama pull llama3.1

• nomic-embed-text is the embedding model
• llama3.1 is the answer generation model

---

# **4. Installing Python Dependencies**

Install ChromaDB and Requests:

pip3 install chromadb requests

If macOS complains about an externally managed environment, install Python directly from [https://python.org](https://python.org) and use that version.

---

# **5. Indexing a Repository**

To index a repo, run:

python3 index_repo.py /path/to/repository

This will:

• Recursively scan all supported file types
• Break files into chunks
• Compute embeddings
• Save vectors into chroma_repo
• Print progress (files processed and chunk counts)

Large repos may take time. The resulting index can be hundreds of MB.

---

# **6. Querying the Repository**

You can query with:

### **A natural language question**

python3 query_repo.py "How does authentication work in the Angular client"

### **A file containing a bug description**

python3 query_repo.py -f bug.txt

### **Increase the number of retrieved snippets**

python3 query_repo.py "Where is cart pricing calculated" -k 16

The system prints:

• Which file snippets were used
• The answer (LLM uses only those snippets)

---

# **7. How the System Works Internally**

### **Embedding Model (nomic-embed-text)**

This model converts text or code into a vector of floating point numbers.
Nearby vectors in space mean the texts are semantically similar.

### **Vector Database (ChromaDB)**

Each chunk is stored with:

• Document text
• Its embedding
• File path
• Chunk index (order within the file)

ChromaDB performs nearest neighbor search when you ask a question.

### **LLM Model (llama3.1)**

After retrieving the top matching code snippets, the system builds a structured prompt telling the LLM:

• Here are the relevant code snippets
• Answer only using them
• Keep the answer concise

The LLM then generates the final answer.

### **All models are pretrained**

You are not training anything.
You are using the models exactly as they were trained by their creators (Nomic AI for embeddings, Meta for Llama).

---

# **8. Directory Layout**

Your project directory should look like this:

code_llm
index_repo.py
query_repo.py
README.md
chroma_repo (created automatically after indexing)

You may delete chroma_repo at any time to rebuild the index.

---

# **9. Troubleshooting**

### **Embedding 500 errors**

If you see a message like:
“Ollama returned 500 for embeddings”

Fix by restarting the Ollama service:

launchctl stop io.ollama
launchctl start io.ollama

### **Weak search results**

Increase the number of snippets:

Use the -k flag such as -k 20 or -k 32.

### **Large queries**

Even long JIRA tickets work.
The embedding model can handle thousands of tokens at once.

---

# **10. Reindexing**

To rebuild the vector database:

1. Delete the chroma_repo folder
2. Run index_repo.py again

This allows you to reindex after major code changes.

---

# **11. Security Notes**

• No code leaves your computer
• Embeddings are local
• LLM inference is local
• No internet calls are made during indexing or querying
• The system is safe for proprietary source code


