# AI pipeline

1. **Input** — choose YouTube (Arabic/English captions), pasted transcript, or PDF/TXT/VTT/SRT.
2. **Clean** — validate the source, decode text, remove subtitle timing/markup, normalize whitespace, and reject empty or unreadable inputs.
3. **Chunk** — split into bounded overlapping word windows. Each window gets a stable `chunk-NNN` ID; long inputs are explicitly reported as truncated.
4. **Embed** — encode every source chunk and each user question with `sentence-transformers/all-MiniLM-L6-v2`; normalize vectors.
5. **Index** — create a per-meeting in-memory FAISS `IndexFlatIP` index over normalized embeddings.
6. **Extract** — run a LangChain `LLMChain`/`PromptTemplate` with the local 4-bit Mistral Nemo model once per transcript chunk. Prompts require explicit evidence and `null` for unstated task metadata.
7. **Parse** — parse with `StructuredOutputParser`/`ResponseSchema`; a JSON-only fallback handles absent code fences. Validate task priority and normalize empty owner/deadline values to null.
8. **Merge** — deduplicate decisions, tasks, key points, and open questions while retaining source IDs and excerpts; then generate a final title and executive summary from the extracted facts.
9. **Retrieve** — embed the question, search FAISS, and select at most the configured top-k source chunks. A configurable similarity threshold rejects weak matches.
10. **Answer** — pass retrieved excerpts and the question to a local, grounded QA chain. The prompt treats transcript instructions as untrusted data and requires the exact no-answer response when evidence is missing. Successful answers include the retrieved chunk IDs and excerpts.

## Guarantees and limitations

Prompting, schema validation, and retrieval thresholds reduce unsupported output; they are not a mathematical guarantee that a generative model never hallucinates. Review important decisions and actions against the cited transcript. The similarity threshold is a heuristic, not calibrated confidence. Meetings and their vector indexes are temporary in-memory state and are lost on backend restart.
