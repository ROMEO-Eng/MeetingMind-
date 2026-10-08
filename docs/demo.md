# Demo walkthrough

## Notebook (educational demo)

1. Open `notebooks/MeetingMind_Final_Project.ipynb` in Colab or Kaggle.
2. Select a CUDA T4 runtime and run all cells in order; the first cell installs the notebook packages and the model cell downloads weights.
3. Use **Load sample meeting** and **Analyse meeting** in Gradio.
4. Inspect the Overview, Decisions, and Action items tabs; download CSV and Markdown.
5. Ask a question about a stated launch decision, then expand Retrieved transcript sources.
6. Repeat with a YouTube video with Arabic/English captions, pasted notes, or a supported file.

## Web product demo

1. Start the backend and frontend using the local commands in the main README, or run `docker compose up --build`.
2. Visit `http://localhost:3000`, then choose **Try sample meeting**. The deterministic sample populates the dashboard without loading the model.
3. Verify that the legal-approval task has no owner or deadline rather than an invented person/date.
4. Choose **Analyse a meeting** to open `/workspace`; load the sample or submit a transcript, YouTube URL, or file.
5. Review decision/task source IDs and excerpts, export notes, then ask a question and expand its retrieved sources.

The analysis and Q&A endpoints need a CUDA-capable GPU. The deterministic UI sample works without model weights; sample Q&A still needs the local model.

## Five-line presentation script

1. MeetingMind transforms meeting transcripts into structured, source-addressable intelligence.
2. Mistral Nemo extracts summaries, decisions, tasks, key points, and open questions locally.
3. Missing task owners and deadlines stay empty instead of being guessed.
4. FAISS retrieves transcript evidence before the local model answers a question.
5. The Next.js dashboard is the product experience; the notebook remains the reproducible course implementation.
