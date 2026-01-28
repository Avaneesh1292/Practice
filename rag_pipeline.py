from langfuse import observe
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

@observe(name="rag_run")
def run_rag(question, retriever):
    docs = retriever.retrieve(question)
    reranked = retriever.rerank(question, docs)

    context = "\n\n".join(d.page_content for d in reranked)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    result = chain.invoke(question)
    return result.content
