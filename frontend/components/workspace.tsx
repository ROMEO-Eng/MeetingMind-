"use client";

import Link from "next/link";
import {
  ArrowLeft,
  ArrowRight,
  AudioLines,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  Download,
  FileText,
  LoaderCircle,
  ListChecks,
  MessageSquareText,
  Paperclip,
  Play,
  Send,
  Sparkles,
  Target,
  Video,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import type { ReactNode } from "react";
import Brand from "@/components/brand";
import RuntimeStatus from "@/components/runtime-status";
import { analyseMeeting, ApiRequestError, askMeeting, downloadNotes, fetchSample } from "@/lib/api";
import type { MeetingAnalysis, MeetingSource, Priority, QuestionResponse } from "@/lib/types";

type InputTab = "youtube" | "text" | "upload";
type ResultTab = "overview" | "decisions" | "tasks" | "ask";
type AnalysisPhase =
  | "idle"
  | "submitting"
  | "analysing"
  | "success"
  | "timeout"
  | "network_error"
  | "backend_error"
  | "model_unavailable"
  | "sample_loading"
  | "sample_error";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  found?: boolean;
  sources?: MeetingSource[];
}

const prompts = [
  "What did the team decide?",
  "Who owns the next action?",
  "What is still unresolved?",
];

function sourceLabel(ids: string[]) {
  return ids.length ? ids.join(", ") : "No source recorded";
}

function PriorityBadge({ priority }: { priority: Priority | null }) {
  if (!priority) return <span className="priority-badge priority-none">Not stated</span>;
  return <span className={`priority-badge priority-${priority.toLowerCase()}`}>{priority}</span>;
}

function SourceDisclosure({ ids, excerpt }: { ids: string[]; excerpt: string | null }) {
  if (!ids.length && !excerpt) return <span className="source-none">Not available</span>;
  return (
    <details className="source-disclosure">
      <summary><FileText size={13} /> {sourceLabel(ids)} <ChevronDown size={13} /></summary>
      <p>{excerpt || "Source excerpt not available."}</p>
    </details>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: ReactNode }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div><span>{label}</span><strong>{value}</strong></div>
    </div>
  );
}

export default function Workspace() {
  const [inputTab, setInputTab] = useState<InputTab>("youtube");
  const [resultTab, setResultTab] = useState<ResultTab>("overview");
  const [phase, setPhase] = useState<AnalysisPhase>("idle");
  const [status, setStatus] = useState("Add a source to begin.");
  const [error, setError] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [transcript, setTranscript] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<MeetingAnalysis | null>(null);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const [downloading, setDownloading] = useState<"csv" | "markdown" | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const isBusy = phase === "submitting" || phase === "analysing" || phase === "sample_loading";

  const applyAnalysis = useCallback((result: MeetingAnalysis, message: string) => {
    setAnalysis(result);
    setChat([]);
    setError("");
    setPhase("success");
    setStatus(message);
    setResultTab("overview");
  }, []);

  const loadSample = useCallback(async () => {
    setPhase("sample_loading");
    setError("");
    setStatus("Loading a ready-to-explore sample…");
    try {
      const result = await fetchSample();
      applyAnalysis(result, "Sample loaded · No model weights needed to preview the dashboard.");
    } catch (requestError) {
      setPhase("sample_error");
      const message = requestError instanceof Error ? requestError.message : "Could not load the sample.";
      setError(message);
      setStatus("The sample could not be loaded.");
    }
  }, [applyAnalysis]);

  useEffect(() => {
    const shouldLoadSample = new URLSearchParams(window.location.search).get("sample") === "1";
    const sampleTimer = shouldLoadSample
      ? window.setTimeout(() => void loadSample(), 0)
      : null;
    return () => {
      if (sampleTimer !== null) window.clearTimeout(sampleTimer);
    };
  }, [loadSample]);

  async function onAnalyse(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setPhase("submitting");
    setStatus("");
    try {
      const form = new FormData();
      form.set("source_type", inputTab);
      if (inputTab === "youtube") form.set("youtube_url", youtubeUrl);
      if (inputTab === "text") form.set("transcript", transcript);
      if (inputTab === "upload" && file) form.set("file", file);
      setPhase("analysing");
      const result = await analyseMeeting(form);
      applyAnalysis(
        result,
        result.chunks_truncated
          ? `Analysis complete · ${result.chunk_count} transcript segments were processed; later text was capped.`
          : `Analysis complete · ${result.word_count.toLocaleString()} words · ${result.chunk_count} source segments.`,
      );
    } catch (requestError) {
      if (requestError instanceof ApiRequestError && requestError.category === "timeout") {
        setPhase("timeout");
        setStatus("Analysis is taking longer than expected.");
        setError("The AI model may still be initializing or processing the meeting. Check the backend status before retrying.");
      } else if (requestError instanceof ApiRequestError && requestError.category === "network") {
        setPhase("network_error");
        setStatus("Unable to reach the AI backend.");
        setError("Check that the backend is running and reachable, then retry.");
      } else if (requestError instanceof ApiRequestError && requestError.statusCode === 503) {
        setPhase("model_unavailable");
        setStatus("AI model unavailable");
        setError(requestError.message || "The AI model is not available in the current environment.");
      } else {
        setPhase("backend_error");
        setStatus("Analysis could not be completed.");
        setError(
          requestError instanceof ApiRequestError
            ? requestError.message
            : "The backend returned an unexpected error. Check the backend status before retrying.",
        );
      }
    }
  }

  async function onAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!analysis || !question.trim() || asking) return;
    const currentQuestion = question.trim();
    setQuestion("");
    setError("");
    setAsking(true);
    setChat((items) => [...items, { id: crypto.randomUUID(), role: "user", text: currentQuestion }]);
    try {
      const result: QuestionResponse = await askMeeting(analysis.meeting_id, currentQuestion);
      setChat((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: result.answer,
          found: result.found,
          sources: result.sources,
        },
      ]);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "The meeting question failed.";
      setError(message);
    } finally {
      setAsking(false);
    }
  }

  async function onDownload(format: "csv" | "markdown") {
    if (!analysis || downloading) return;
    setDownloading(format);
    setError("");
    try {
      await downloadNotes(analysis.meeting_id, format);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The download failed.");
    } finally {
      setDownloading(null);
    }
  }

  const minutesLabel = analysis ? `${analysis.estimated_minutes} min` : "—";

  return (
    <main className="workspace-shell">
      <header className="workspace-topbar">
        <div className="brand-area"><Brand /><RuntimeStatus /></div>
        <div className="workspace-nav">
          <Link href="/" className="back-link"><ArrowLeft size={15} /> Home</Link>
        </div>
      </header>

      <section className="workspace-intro">
        <div><div className="eyebrow"><Sparkles size={13} /> AI MEETING INTELLIGENCE</div><h1>Your meeting, in focus.</h1><p>Bring a transcript. Leave with a clear record of what was said and what happens next.</p></div>
        {analysis && <div className="meeting-source-chip"><FileText size={14} /> {analysis.source_name}</div>}
      </section>

      <div className="workspace-grid">
        <aside className="input-panel">
          <div className="panel-heading"><span className="panel-index">01</span><div><h2>Meeting source</h2><p>Choose how to bring the conversation in.</p></div></div>
          <div className="source-tabs" role="tablist" aria-label="Meeting source">
            <button type="button" role="tab" aria-selected={inputTab === "youtube"} className={inputTab === "youtube" ? "source-tab selected" : "source-tab"} onClick={() => setInputTab("youtube")}><Video size={15} /> YouTube</button>
            <button type="button" role="tab" aria-selected={inputTab === "text"} className={inputTab === "text" ? "source-tab selected" : "source-tab"} onClick={() => setInputTab("text")}><FileText size={15} /> Paste text</button>
            <button type="button" role="tab" aria-selected={inputTab === "upload"} className={inputTab === "upload" ? "source-tab selected" : "source-tab"} onClick={() => setInputTab("upload")}><Paperclip size={15} /> Upload</button>
          </div>

          <form onSubmit={onAnalyse}>
            {inputTab === "youtube" && (
              <div className="input-content">
                <label htmlFor="youtube-url">Video link</label>
                <input id="youtube-url" type="url" value={youtubeUrl} onChange={(event) => setYoutubeUrl(event.target.value)} placeholder="https://youtube.com/watch?v=…" required />
                <p className="field-hint">English captions are supported.</p>
              </div>
            )}
            {inputTab === "text" && (
              <div className="input-content">
                <label htmlFor="transcript">Meeting transcript</label>
                <textarea id="transcript" value={transcript} onChange={(event) => setTranscript(event.target.value)} placeholder="Paste notes or a transcript with speaker names…" rows={10} required />
                <p className="field-hint">{transcript.trim() ? `${transcript.trim().split(/\s+/).length.toLocaleString()} words` : "English transcripts are supported."}</p>
              </div>
            )}
            {inputTab === "upload" && (
              <div className="input-content">
                <label htmlFor="transcript-file">Transcript file</label>
                <input ref={fileRef} className="file-input" id="transcript-file" type="file" accept=".pdf,.txt,.vtt,.srt,application/pdf,text/plain" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
                <label className="upload-drop" htmlFor="transcript-file"><Paperclip size={19} /><strong>{file?.name ?? "Choose a transcript"}</strong><span>PDF, TXT, VTT, SRT · up to 25 MB</span></label>
                <p className="field-hint">PDF files must contain selectable text.</p>
              </div>
            )}
            <button className="button button-primary button-full analyse-button" type="submit" disabled={isBusy}>
              {isBusy ? <><LoaderCircle className="spin" size={17} /> {phase === "submitting" ? "Submitting transcript" : phase === "sample_loading" ? "Loading sample" : "Analysing meeting..."}</> : <>Analyse meeting <ArrowRight size={16} /></>}
            </button>
          </form>
          <button className="sample-button" type="button" onClick={() => void loadSample()} disabled={isBusy}><Play size={14} /> Load sample meeting</button>
          <div className={`status-box ${phase === "success" ? "status-success" : ["timeout", "network_error", "backend_error", "model_unavailable", "sample_error"].includes(phase) ? "status-error" : isBusy ? "status-progress" : ""}`} role="status" aria-live="polite">
            {phase === "success" ? <Check size={15} /> : ["timeout", "network_error", "backend_error", "model_unavailable", "sample_error"].includes(phase) ? <CircleAlert size={15} /> : isBusy ? <LoaderCircle className="spin" size={15} /> : <span className="status-dot" />}
            <span>
              {phase === "submitting"
                ? "Submitting transcript…"
                : phase === "analysing"
                  ? <><strong>Analysing meeting...</strong><small>The AI model may take a moment to initialize on first use.</small></>
                  : status}
            </span>
          </div>
          {error && <div className="error-notice" role="alert"><CircleAlert size={15} /><span>{error}</span></div>}
          <div className="privacy-callout"><Target size={15} /><span><b>Grounded by design.</b> Missing owners and dates stay unstated.</span></div>
        </aside>

        <section className="results-panel" aria-label="Meeting results">
          <div className="results-heading">
            <div className="panel-heading"><span className="panel-index">02</span><div><h2>Meeting intelligence</h2><p>{analysis ? analysis.title : "Your structured meeting record will appear here."}</p></div></div>
            {analysis && <span className="ready-pill"><Check size={13} /> Ready</span>}
          </div>
          {!analysis ? (
            <div className="empty-results">
              <div className="empty-art"><AudioLines size={29} /><span /><span /><span /></div>
              <h3>{isBusy ? phase === "sample_loading" ? "Preparing the sample" : "Analysing meeting..." : "A clearer meeting record starts here"}</h3>
              <p>{isBusy ? phase === "sample_loading" ? status : "The AI model may take a moment to initialize on first use." : "Add a YouTube link, paste a transcript, or upload a file to see decisions, actions, and source-grounded answers."}</p>
              {!isBusy && <button className="text-link" type="button" onClick={() => void loadSample()}>Explore the sample <ArrowRight size={14} /></button>}
            </div>
          ) : (
            <>
              <div className="stats-grid">
                <StatCard label="Action items" value={analysis.tasks.length} icon={<Check size={16} />} />
                <StatCard label="Decisions" value={analysis.decisions.length} icon={<Target size={16} />} />
                <StatCard label="Words" value={analysis.word_count.toLocaleString()} icon={<FileText size={16} />} />
                <StatCard label="Est. duration" value={minutesLabel} icon={<Clock3 size={16} />} />
              </div>
              <div className="result-tabs" role="tablist" aria-label="Meeting results">
                {([
                  ["overview", "Overview"],
                  ["decisions", "Decisions"],
                  ["tasks", "Action items"],
                  ["ask", "Ask the meeting"],
                ] as const).map(([tab, label]) => (
                  <button key={tab} className={resultTab === tab ? "result-tab active" : "result-tab"} type="button" role="tab" aria-selected={resultTab === tab} onClick={() => setResultTab(tab)}>
                    {label}{tab === "tasks" && <span className="tab-count">{analysis.tasks.length}</span>}
                  </button>
                ))}
              </div>
              <div className="result-content">
                {resultTab === "overview" && <Overview analysis={analysis} onNavigate={setResultTab} />}
                {resultTab === "decisions" && <Decisions analysis={analysis} />}
                {resultTab === "tasks" && <Tasks analysis={analysis} downloading={downloading} onDownload={onDownload} />}
                {resultTab === "ask" && (
                  <section className="qa-panel">
                    <div className="section-title"><div><MessageSquareText size={17} /><h3>Ask about this meeting</h3></div><span>Answers stay tied to retrieved excerpts</span></div>
                    {chat.length === 0 ? (
                      <div className="chat-empty"><div className="chat-icon"><MessageSquareText size={19} /></div><h4>What would you like to know?</h4><p>Questions are answered from this transcript only.</p><div className="question-prompts">{prompts.map((prompt) => <button key={prompt} type="button" onClick={() => setQuestion(prompt)}>{prompt}<ArrowRight size={13} /></button>)}</div></div>
                    ) : (
                      <div className="chat-list">
                        {chat.map((message) => <ChatBubble key={message.id} message={message} />)}
                        {asking && <div className="assistant-thinking"><LoaderCircle size={15} className="spin" /> Retrieving sources and preparing an answer…</div>}
                      </div>
                    )}
                    <form className="question-form" onSubmit={onAsk}>
                      <label className="sr-only" htmlFor="meeting-question">Ask a question about the meeting</label>
                      <input id="meeting-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about a decision, owner, or open question…" disabled={asking} />
                      <button type="submit" aria-label="Send question" disabled={!question.trim() || asking}><Send size={16} /></button>
                    </form>
                    <p className="qa-footnote">No answer found? MeetingMind says so instead of filling in the gaps.</p>
                  </section>
                )}
              </div>
            </>
          )}
        </section>
      </div>
      <footer className="workspace-footer"><span><AudioLines size={14} /> MeetingMind</span><span>Review source excerpts before acting on generated notes.</span></footer>
    </main>
  );
}

function Overview({ analysis, onNavigate }: { analysis: MeetingAnalysis; onNavigate: (tab: ResultTab) => void }) {
  return (
    <div className="overview-content">
      <article className="summary-card">
        <div className="section-title"><div><Sparkles size={16} /><h3>Executive summary</h3></div><span>{analysis.chunk_count} source segments</span></div>
        <p>{analysis.executive_summary || "No summary was returned."}</p>
        {analysis.chunks_truncated && <div className="truncation-note"><CircleAlert size={14} /> This transcript exceeded the processing cap; later text was not analyzed.</div>}
      </article>
      <div className="overview-pair">
        <article className="compact-card">
          <div className="section-title"><div><Target size={15} /><h3>Decisions</h3></div><button className="text-link" onClick={() => onNavigate("decisions")} type="button">View all <ArrowRight size={13} /></button></div>
          {analysis.decisions.length ? <ul className="mini-list">{analysis.decisions.slice(0, 2).map((item) => <li key={`${item.decision}-${item.source_ids.join()}`}><span />{item.decision}</li>)}</ul> : <p className="muted-empty">No explicit decisions found.</p>}
        </article>
        <article className="compact-card">
          <div className="section-title"><div><CircleAlert size={15} /><h3>Open questions</h3></div><span className="subtle-count">{analysis.open_questions.length}</span></div>
          {analysis.open_questions.length ? <ul className="mini-list">{analysis.open_questions.slice(0, 2).map((item) => <li key={item}><span className="question-marker">?</span>{item}</li>)}</ul> : <p className="muted-empty">No open questions captured.</p>}
        </article>
      </div>
      <div className="overview-counts">
        <div><ListChecks size={15} /><span>{analysis.key_points.length} key points</span></div>
        <div><Clock3 size={15} /><span>About {analysis.estimated_minutes} minutes</span></div>
        <button className="text-link" type="button" onClick={() => onNavigate("tasks")}>Review action items <ArrowRight size={13} /></button>
      </div>
    </div>
  );
}

function Decisions({ analysis }: { analysis: MeetingAnalysis }) {
  if (!analysis.decisions.length) return <EmptyState title="No decisions captured" body="No explicit decisions were found in this transcript." />;
  return <div className="decision-list">{analysis.decisions.map((item, index) => <article className="decision-card" key={`${item.decision}-${index}`}><div className="decision-index">{String(index + 1).padStart(2, "0")}</div><div className="decision-main"><span className="eyebrow">DECISION</span><h3>{item.decision}</h3><p>{item.context || "No additional context stated."}</p><SourceDisclosure ids={item.source_ids} excerpt={item.source_excerpt} /></div></article>)}</div>;
}

function Tasks({ analysis, downloading, onDownload }: { analysis: MeetingAnalysis; downloading: "csv" | "markdown" | null; onDownload: (format: "csv" | "markdown") => void }) {
  return (
    <section className="tasks-panel">
      <div className="section-title task-heading"><div><div><h3>Action items</h3><span>{analysis.tasks.length} items extracted from the conversation</span></div></div>
        <div className="download-actions">
          <button type="button" className="download-button" onClick={() => onDownload("csv")} disabled={downloading !== null}><Download size={14} />{downloading === "csv" ? "Preparing…" : "CSV"}</button>
          <button type="button" className="download-button" onClick={() => onDownload("markdown")} disabled={downloading !== null}><Download size={14} />{downloading === "markdown" ? "Preparing…" : "Markdown"}</button>
        </div>
      </div>
      {!analysis.tasks.length ? <EmptyState title="No action items" body="No explicit action items were found in this transcript." /> : (
        <div className="table-scroll"><table className="task-table"><thead><tr><th>Task</th><th>Owner</th><th>Deadline</th><th>Priority</th><th>Source</th></tr></thead><tbody>
          {analysis.tasks.map((task, index) => <tr key={`${task.task}-${index}`}><td className="task-name">{task.task}</td><td>{task.owner || <span className="not-stated">Not stated</span>}</td><td>{task.deadline || <span className="not-stated">Not stated</span>}</td><td><PriorityBadge priority={task.priority} /></td><td><SourceDisclosure ids={task.source_ids} excerpt={task.source_excerpt} /></td></tr>)}
        </tbody></table></div>
      )}
      <p className="table-note"><CircleAlert size={13} /> “Not stated” means the transcript did not identify an owner or deadline.</p>
    </section>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  return (
    <div className={`chat-message ${message.role}`}>
      <div className="bubble-role">{message.role === "user" ? "YOU" : "MEETINGMIND"}</div>
      <div className="bubble-content">{message.text}</div>
      {message.role === "assistant" && message.found && message.sources?.length ? (
        <details className="answer-sources">
          <summary><FileText size={13} /> View sources <ChevronDown size={13} /></summary>
          <div className="answer-source-list">{message.sources.map((source) => <div className="answer-source" key={source.chunk_id}><strong>{source.chunk_id}</strong><p>{source.excerpt}</p></div>)}</div>
        </details>
      ) : message.role === "assistant" ? <div className="no-source-note"><CircleAlert size={13} /> No supporting source was found.</div> : null}
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return <div className="empty-state"><div><FileText size={18} /></div><h4>{title}</h4><p>{body}</p></div>;
}
