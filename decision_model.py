from langfuse import observe
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tools import ask_for_clarification

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

@observe(name="clarification_decision")
def decide_next_step(question: str):
    system_prompt = (
        "Decide whether the question can be answered accurately.\n"
        "If more context is required, call the tool `ask_for_clarification`\n"
        "with a short reason.\n"
        "If the question is clear, respond with NO_CLARIFICATION_NEEDED."
    )

    response = llm.invoke(
        [
            HumanMessage(content=system_prompt),
            HumanMessage(content=question),
        ],
        tools=[ask_for_clarification],
    )

    return response