import Link from "next/link";
import {
  ArrowDown,
  ArrowRight,
  AudioLines,
  Check,
  CircleHelp,
  FileCheck2,
  ListChecks,
  Sparkles,
  Target,
} from "lucide-react";
import Brand from "@/components/brand";
import RuntimeStatus from "@/components/runtime-status";

const stages = [
  { label: "Transcript", detail: "One conversation, any format", icon: AudioLines },
  { label: "AI extraction", detail: "Local, evidence-first analysis", icon: Sparkles },
  { label: "Structured intelligence", detail: "Decisions and accountable work", icon: ListChecks },
  { label: "Grounded Q&A", detail: "Answers with a source trail", icon: CircleHelp },
];

export default function HomePage() {
  return (
    <main className="landing-shell">
      <nav className="topbar">
        <div className="brand-area"><Brand /><RuntimeStatus /></div>
        <div className="nav-right">
          <span className="privacy-note"><span className="live-dot" /> Local AI · Source-grounded</span>
          <Link className="nav-link" href="/workspace">Workspace <ArrowRight size={14} /></Link>
        </div>
      </nav>

      <section className="hero-grid">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={14} /> INTELLIGENCE THAT STAYS CLOSE TO THE SOURCE</div>
          <h1>Meetings end.<br /><span>Clarity stays.</span></h1>
          <p className="hero-description">
            Turn meetings into structured summaries, decisions, and actionable tasks.
            Every answer leads back to the words that support it.
          </p>
          <div className="hero-actions">
            <Link className="button button-primary" href="/workspace">Analyse a meeting <ArrowRight size={17} /></Link>
            <Link className="button button-secondary" href="/workspace?sample=1"><Sparkles size={16} /> Try sample meeting</Link>
          </div>
          <div className="hero-proof"><Check size={15} /> Open models · no hosted AI API · Arabic + English</div>
        </div>

        <div className="preview-card" aria-label="Illustrative meeting summary preview">
          <div className="preview-head">
            <div><span className="preview-kicker">MEETING BRIEF</span><h2>Product launch sync</h2></div>
            <span className="preview-status"><span className="live-dot" /> READY</span>
          </div>
          <div className="preview-summary">
            <span className="mini-label">EXECUTIVE SUMMARY</span>
            <p>The team aligned on a November launch, confirmed two workstreams, and flagged one open dependency.</p>
          </div>
          <div className="preview-stats">
            <div><strong>04</strong><span>Tasks</span></div>
            <div><strong>03</strong><span>Decisions</span></div>
            <div><strong>01</strong><span>Open question</span></div>
          </div>
          <div className="preview-task">
            <span className="task-check"><Check size={12} /></span>
            <span><b>Finalize launch copy</b><small>Omar · 7 November</small></span>
            <span className="priority-tag">HIGH</span>
          </div>
          <div className="preview-task">
            <span className="task-unassigned" />
            <span><b>Obtain legal approval</b><small>Owner not stated</small></span>
            <span className="source-mini">chunk-02</span>
          </div>
          <div className="preview-foot"><FileCheck2 size={14} /> Every action stays linked to its source</div>
        </div>
      </section>

      <section className="workflow-section">
        <div className="section-heading">
          <div><span className="eyebrow">FROM CONVERSATION TO FOLLOW-THROUGH</span><h2>A clear trail from words to work.</h2></div>
          <p>No mystery summaries. No invented owners.<br />Just useful meeting intelligence.</p>
        </div>
        <div className="workflow-grid">
          {stages.map((stage, index) => {
            const Icon = stage.icon;
            return (
              <div className="workflow-step" key={stage.label}>
                <div className="workflow-icon"><Icon size={19} /></div>
                <div className="workflow-number">0{index + 1}</div>
                <h3>{stage.label}</h3>
                <p>{stage.detail}</p>
                {index < stages.length - 1 && <ArrowDown className="workflow-arrow" size={16} aria-hidden="true" />}
              </div>
            );
          })}
        </div>
      </section>

      <section className="bottom-cta">
        <div><span className="eyebrow">YOUR NEXT MEETING, MADE USEFUL</span><h2>Start with the conversation.</h2></div>
        <Link className="button button-primary" href="/workspace">Open workspace <Target size={17} /></Link>
      </section>
      <footer className="site-footer">
        <span>MeetingMind <span className="footer-muted">AI Meeting Intelligence</span></span>
        <span>Local inference · Transparent sources · Human review encouraged</span>
      </footer>
    </main>
  );
}
