from backend.app.chains.prompts import NO_ANSWER
from backend.app.embeddings.vector_store import RetrievedChunk
from backend.app.preprocessing.chunking import TranscriptChunk
from backend.app.rag.grounded_qa import answer_question


class FakeStore:
    def __init__(self, score: float):
        self.score = score

    def search(self, question, embedding_model, top_k):
        return [
            RetrievedChunk(
                chunk=TranscriptChunk(
                    chunk_id="chunk-007",
                    text="Maya will send the launch checklist on Friday.",
                    start_word=0,
                    end_word=8,
                ),
                score=self.score,
            )
        ]


class FakeChain:
    def __init__(self, answer: str):
        self.answer = answer
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return {"text": self.answer}


def test_grounded_answer_includes_source_id_excerpt_and_question_context() -> None:
    chain = FakeChain("Maya will send it on Friday.")
    response = answer_question(
        "Who sends the checklist?",
        FakeStore(0.81),
        embedding_model=None,
        qa_chain=chain,
        top_k=4,
        min_score=0.2,
    )

    assert response.found
    assert response.sources[0].chunk_id == "chunk-007"
    assert "launch checklist" in response.sources[0].excerpt
    assert "Who sends" in chain.inputs["question"]
    assert "chunk-007" in chain.inputs["context"]


def test_low_similarity_returns_no_answer_and_no_sources() -> None:
    response = answer_question(
        "What is the speaker's favorite color?",
        FakeStore(0.05),
        embedding_model=None,
        qa_chain=FakeChain("unverified guess"),
        top_k=4,
        min_score=0.2,
    )

    assert response.answer == NO_ANSWER
    assert response.found is False
    assert response.sources == []


def test_model_no_answer_returns_empty_source_list() -> None:
    response = answer_question(
        "Was the catering arranged?",
        FakeStore(0.8),
        embedding_model=None,
        qa_chain=FakeChain(NO_ANSWER),
        top_k=4,
        min_score=0.2,
    )

    assert response.answer == NO_ANSWER
    assert response.sources == []
