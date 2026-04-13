from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- NEW: Re-ranker Imports ---
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

load_dotenv()


class GraphState(TypedDict):
    question: str
    classification: str
    response: str
    context: str


# --- NODE 1: Classifier ---
def classify_input_node(state: GraphState):
    print("🔍 [Node: Classifier] Determining intent...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Classify as 'FINANCE' or 'GENERAL'. One word only."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    classification = chain.invoke({"question": state["question"]}).strip().upper()
    return {"classification": classification}


# --- NODE 2: Re-ranker (THE NEW PART) ---
def rerank_retrieval_node(state: GraphState):
    print("🎯 [Node: Re-ranker] Finding the most relevant financial data...")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 1. Base Retriever (Fetch top 10 potential matches)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Setup FlashRank Re-ranker
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 3. Re-rank to find the top 3 best chunks
    ranked_docs = compression_retriever.invoke(state["question"])
    context_text = "\n\n".join([doc.page_content for doc in ranked_docs])

    return {"context": context_text}


# --- NODE 3: RAG Generator ---
def generate_answer_node(state: GraphState):
    print("✍️ [Node: Generator] Writing final response...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
    Answer the question based ONLY on the context below.
    Context: {context}
    Question: {question}
    """
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"context": state["context"], "question": state["question"]}
    )
    return {"response": response}


# --- NODE 4: General Chat ---
def handle_general_node(state: GraphState):
    print("💬 [Node: General] Chatting...")
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke(state["question"])
    return {"response": response.content}


# --- Routing Logic ---
def decide_route(state: GraphState) -> Literal["finance", "general"]:
    return "finance" if "FINANCE" in state["classification"] else "general"


# --- Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("classifier", classify_input_node)
workflow.add_node("reranker", rerank_retrieval_node)  # New Node
workflow.add_node("generator", generate_answer_node)  # Split from RAG
workflow.add_node("general", handle_general_node)

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier", decide_route, {"finance": "reranker", "general": "general"}
)

workflow.add_edge("reranker", "generator")  # Sequential flow
workflow.add_edge("generator", END)
workflow.add_edge("general", END)

app = workflow.compile()

if __name__ == "__main__":
    user_query = input("\nAsk about Bajaj: ")
    inputs = {"question": user_query}
    for output in app.stream(inputs):
        for key in output.keys():
            print(f"✅ Completed: {key}")

    final = app.invoke(inputs)
    print(f"\nFINAL ANSWER:\n{final['response']}")
