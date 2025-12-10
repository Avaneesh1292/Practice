import os
import sys
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ==================================================
# CONFIGURATION
# ==================================================
# 🔴 ACTION REQUIRED: Paste your API key inside the quotes below
os.environ["GOOGLE_API_KEY"] = "PAIzaSyCrTkQ4YiSCDV0fPp8soULkWHXsruKCPe0"

# Check if knowledge base exists
if not os.path.exists("knowledge_base.txt"):
    print("Error: 'knowledge_base.txt' not found!")
    print("Please create this file and paste some text into it.")
    sys.exit()

# ==================================================
# STEP 1: Ingestion & Chunking (Theory: "Context Management")
# ==================================================
print("1. Loading and Chunking Data...")
loader = TextLoader("knowledge_base.txt")
docs = loader.load()

# THEORETICAL CONCEPT: CHUNKING
# We use 'RecursiveCharacterTextSplitter' to intelligently split text.
# - chunk_size=1000: Breaks text into ~1000 character pieces.
# - chunk_overlap=200: Keeps the last 200 chars of the previous chunk
#   to ensure sentences/context aren't cut in half.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"   --> Success! Split document into {len(splits)} chunks.")

# ==================================================
# STEP 2: Vector Store (Theory: "Semantic Search")
# ==================================================
print("2. Creating Embeddings & Vector Store...")

# THEORETICAL CONCEPT: EMBEDDINGS
# We use Google's 'text-embedding-004' model.
# This converts your text chunks into lists of numbers (Vectors).
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# THEORETICAL CONCEPT: VECTOR DATABASE
# We use FAISS (Facebook AI Similarity Search) to store these vectors.
# It allows us to mathematically find the "nearest neighbor" to a question.
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# ==================================================
# STEP 3: The LLM (Theory: "Generation")
# ==================================================
print("3. Initializing Gemini 1.5 Flash...")

# We use Gemini 1.5 Flash. It is fast, free, and has a large context window.
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

# ==================================================
# STEP 4: The RAG Chain
# ==================================================
# This prompt tells the AI exactly how to behave.
system_prompt = (
    "You are a helpful AI assistant. Use the context below to answer the question. "
    "If the answer is not in the context, say 'I don't know'—do not make it up."
    "\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Build the pipeline: Retrieval -> Prompt -> LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==================================================
# STEP 5: Interactive Loop
# ==================================================
print("\n" + "="*50)
print("Gemini RAG System Ready!")
print("(Type 'exit' or 'quit' to stop)")
print("="*50)

while True:
    user_query = input("\nAsk a question: ")
    if user_query.lower() in ['exit', 'quit']:
        break

    if not user_query.strip():
        continue

    print("   (Thinking...)")
    try:
        response = rag_chain.invoke({"input": user_query})
        print(f"\nGemini Answer: {response['answer']}")
    except Exception as e:
        print(f"\nError: {e}")