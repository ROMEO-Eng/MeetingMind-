"""Deterministic sample meeting used to explore the UI without loading a model."""

from backend.app.models import (
    DecisionItem,
    MeetingAnalysisResponse,
    SourceChunk,
    TaskItem,
)

SAMPLE_MEETING_ID = "meetingmind-sample"

SAMPLE_CHUNKS = [
    SourceChunk(
        chunk_id="chunk-001",
        text=(
            "Lina (Product), Omar (Engineering), Sara (Design), and Nabil (Operations) "
            "met to plan the November product launch. Decision: the team will launch the "
            "new onboarding flow on 14 November, subject to QA sign-off. Omar owns the "
            "final onboarding copy and will deliver it by 7 November. The signup "
            "completion rate improved by 8% in the latest usability test."
        ),
    ),
    SourceChunk(
        chunk_id="chunk-002",
        text=(
            "Decision: keep the existing pricing page for launch and review it after "
            "the first month. Sara owns the accessibility review, due 10 November. "
            "The team needs to obtain legal approval for the launch copy, but no owner "
            "was assigned. Nabil will confirm support coverage for launch weekend by "
            "12 November."
        ),
    ),
    SourceChunk(
        chunk_id="chunk-003",
        text=(
            "Decision: use the current analytics dashboard for launch; revisit "
            "additional metrics after the first week. Open question: will customer "
            "support coverage be available throughout launch weekend? Lina will share "
            "the final launch checklist with the team tomorrow."
        ),
    ),
]

SAMPLE_ANALYSIS = MeetingAnalysisResponse(
    meeting_id=SAMPLE_MEETING_ID,
    source_name="Built-in launch planning sample",
    title="November product launch planning",
    executive_summary=(
        "The team aligned on a conditional 14 November onboarding launch, keeping "
        "current pricing and analytics for launch while assigning preparation work. "
        "Legal approval still has no named owner, and weekend support coverage remains "
        "an open question."
    ),
    decisions=[
        DecisionItem(
            decision="Launch the new onboarding flow on 14 November, subject to QA sign-off.",
            context="The launch date depends on QA approval.",
            source_ids=["chunk-001"],
            source_excerpt=SAMPLE_CHUNKS[0].text,
        ),
        DecisionItem(
            decision="Keep the existing pricing page for launch and review it after the first month.",
            context="The pricing page is unchanged for the initial release.",
            source_ids=["chunk-002"],
            source_excerpt=SAMPLE_CHUNKS[1].text,
        ),
        DecisionItem(
            decision="Use the current analytics dashboard for launch.",
            context="Additional metrics will be revisited after the first week.",
            source_ids=["chunk-003"],
            source_excerpt=SAMPLE_CHUNKS[2].text,
        ),
    ],
    tasks=[
        TaskItem(
            task="Deliver the final onboarding copy",
            owner="Omar",
            deadline="7 November",
            priority="High",
            source_ids=["chunk-001"],
            source_excerpt=SAMPLE_CHUNKS[0].text,
        ),
        TaskItem(
            task="Run the accessibility review",
            owner="Sara",
            deadline="10 November",
            priority="Medium",
            source_ids=["chunk-002"],
            source_excerpt=SAMPLE_CHUNKS[1].text,
        ),
        TaskItem(
            task="Obtain legal approval for the launch copy",
            owner=None,
            deadline=None,
            priority=None,
            source_ids=["chunk-002"],
            source_excerpt=SAMPLE_CHUNKS[1].text,
        ),
        TaskItem(
            task="Confirm support coverage for launch weekend",
            owner="Nabil",
            deadline="12 November",
            priority="Medium",
            source_ids=["chunk-002"],
            source_excerpt=SAMPLE_CHUNKS[1].text,
        ),
        TaskItem(
            task="Share the final launch checklist",
            owner="Lina",
            deadline="Tomorrow",
            priority=None,
            source_ids=["chunk-003"],
            source_excerpt=SAMPLE_CHUNKS[2].text,
        ),
    ],
    key_points=[
        "Signup completion improved by 8% in the latest usability test.",
        "The launch depends on QA sign-off.",
    ],
    open_questions=[
        "Will customer support coverage be available throughout launch weekend?"
    ],
    word_count=sum(len(chunk.text.split()) for chunk in SAMPLE_CHUNKS),
    estimated_minutes=max(1, round(sum(len(chunk.text.split()) for chunk in SAMPLE_CHUNKS) / 130)),
    chunk_count=len(SAMPLE_CHUNKS),
    chunks_truncated=False,
)
