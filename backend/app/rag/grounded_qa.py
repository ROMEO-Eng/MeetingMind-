"""Retrieve transcript sources and answer questions only from those sources."""

from backend.app.chains.prompts import NO_ANSWER
from backend.app.embeddings.vector_store import FaissVectorStore
from backend.app.models import QuestionResponse, QuestionSource


def answer_question(
    question: str,
    vector_store: FaissVectorStore,
    embedding_model,
    qa_chain,
    top_k: int,
    min_score: float,
) -> QuestionResponse:
    question = question.strip()
    if not question:
        raise ValueError("Enter a question about the meeting.")

    matches = vector_store.search(question, embedding_model, top_k)
    if not matches or matches[0].score < min_score:
        return QuestionResponse(answer=NO_ANSWER, found=False, sources=[])

    context = "\n\n".join(
        f"[{match.chunk.chunk_id}]\n{match.chunk.text}" for match in matches
    )
    answer = qa_chain.invoke({"context": context, "question": question})["text"].strip()
    found = answer.casefold() != NO_ANSWER.casefold()
    sources = [
        QuestionSource(
            chunk_id=match.chunk.chunk_id,
            excerpt=match.chunk.text,
            score=match.score,
        )
        for match in matches
    ] if found else []
    return QuestionResponse(
        answer=answer if found else NO_ANSWER,
        found=found,
        sources=sources,
    )
