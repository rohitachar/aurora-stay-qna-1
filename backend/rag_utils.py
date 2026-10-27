
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant hotel-policy chunks.
def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    if vectorstore is None:
        raise ValueError("A vectorstore is required to retrieve policy information.")
    if not query or not query.strip():
        raise ValueError("The query must not be empty.")

    results = vectorstore.similarity_search(query, k=k)
    return [
        f"Content:\n{result.page_content}\n\nMetadata:\n{result.metadata}"
        for result in results
    ]

# Generate a grounded answer or explicitly refuse when retrieval has no evidence.
def generate(query, retrieved_chunks, model) -> str:
    if not retrieved_chunks:
        return "The available policy documents do not contain enough information to answer this question."

    prompt = f"""You are an AI assistant answering questions about AuroraStay hotel policies.

User question:
{query}

Retrieved policy information:
{"\n\n---\n\n".join(retrieved_chunks)}

Rules:
1. Use only information supported by the retrieved policy information.
2. Answer the specific question clearly and concisely.
3. Do not invent fees, dates, amenities, exceptions, or other policy details.
4. If evidence is missing or conflicting, say so and recommend staff review.
5. Mention a source document when it is included in the retrieved metadata.

Answer:"""
    return model.invoke(prompt).content

# Run retrieval and generation for one policy question.
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None,
):
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
    )
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query, retrieved_chunks, model)
    return answer, retrieved_chunks
