from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Initialize LLM (Using non-deprecated Groq model)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def classify_query(query):
    """Determines if the query is FINANCE-related or GENERAL."""
    system_prompt = (
        "You are a router. Classify the user query into 'FINANCE' or 'GENERAL'. "
        "If it asks about data, numbers, or company info, choose 'FINANCE'. "
        "Respond with ONLY one word."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{query}")]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip().upper()


def run_rag_pipeline(query):
    # Load the existing Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # System Prompt for Financial RAG
    rag_system_prompt = (
        "You are a Finance Assistant. Use the following retrieved context to answer. "
        "If the answer isn't in the context, say you don't have that data. "
        "\n\nContext: {context}"
    )

    rag_prompt = ChatPromptTemplate.from_messages(
        [("system", rag_system_prompt), ("human", "{question}")]
    )

    # RAG Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)


if __name__ == "__main__":
    user_query = input("Ask a question: ")
    category = classify_query(user_query)

    if "FINANCE" in category:
        print("🔍 Routing to Financial Database...")
        print(run_rag_pipeline(user_query))
    else:
        print("💬 Routing to General Chat...")
        response = llm.invoke(user_query)
        print(response.content)
