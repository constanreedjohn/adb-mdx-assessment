"""Acceptance (E1-05): every domain model round-trips to JSON unchanged.

One representative, fully-populated instance per model. `test_every_model_is_covered`
fails if a model is added to a domain package without a sample here.
"""

import inspect
from datetime import UTC, date, datetime

import pytest
from pydantic import BaseModel

import mdx_assessment.core.domain as core_domain
import mdx_assessment.usecases.mdx.domain as mdx_domain
from mdx_assessment.core.domain import (
    GENESIS_HASH,
    AuditEvent,
    ChainVerification,
    Decision,
    DecisionKind,
    DomainModel,
    Evidence,
    Fact,
    Finding,
    Measure,
    NewAuditEvent,
    Plan,
    PlanSource,
    PlanStep,
    PlanValidation,
    ReviewCheckResult,
    RunContext,
    RunObservation,
    SessionResult,
    SessionStatus,
    Severity,
    Task,
    TaskStatus,
    TraceRef,
    Versions,
)
from mdx_assessment.usecases.mdx.domain import (
    Case,
    CaseStatus,
    ColorCode,
    DataType,
    ExplanationSource,
    FileMetadata,
    MarketDataKind,
    MdxFlag,
    Observation,
    Point,
    Series,
    Tenor,
    Unit,
)

D = date(2026, 10, 15)
T0 = datetime(2026, 10, 15, 17, 20, tzinfo=UTC)
T1 = datetime(2026, 10, 15, 17, 21, 30, 123456, tzinfo=UTC)
EV = Evidence.create(
    "point_change",
    "mdx_data.get_flags",
    {"curve": "JPY", "tenor": "3M", "prev": 0.04, "curr": 0.07},
)
MEASURE = Measure(name="change_bps", value=3.0, unit="bps", evidence_ids=(EV.evidence_id,))
FINDING = Finding(
    finding_id="f-1",
    kind="threshold_artifact",
    subject="JPY 3M",
    summary="+75% is +3 bps on a near-zero rate",
    severity=Severity.LOW,
    measures=(MEASURE,),
    evidence_ids=(EV.evidence_id,),
)
STEP = PlanStep(step_id="s1", task="input_validation.check_files_and_intake", rationale="mandatory")
PLAN = Plan(
    source=PlanSource.MASTER,
    steps=(STEP, PlanStep(step_id="s2", task="market_data.near_zero_check")),
    follow_ups=(PlanStep(step_id="f1", task="input_validation.compare_prior_day"),),
    rationale="normal day",
    cited_fact_ids=("fact-flag-count",),
)
VALIDATION = PlanValidation(
    valid=False,
    checks={"P-1": False, "P-2": True, "P-3": True, "P-4": True},
    attempts=2,
    errors=("P-1: mandatory step 'a.b' is missing",),
    fallback_reason="plan_invalid",
)
TASK = Task(
    task_id="t1",
    session_id="sess-1",
    step_id="s1",
    task=STEP.task,
    status=TaskStatus.DONE,
    inputs={"business_date": "2026-10-15", "files": ["linear.csv"], "limit": 3},
    started_at=T0,
    finished_at=T1,
)
CHECK = ReviewCheckResult(
    check_id="C-1", name="flag reconciliation", passed=True, evidence_ids=(EV.evidence_id,)
)
DECISION = Decision(
    step_id="s1",
    kind=DecisionKind.FOLLOW_UP,
    reason="duplicate file suspected",
    follow_up=PlanStep(step_id="f2", task="input_validation.compare_prior_day"),
    evidence_ids=(EV.evidence_id,),
)

NEW_EVENT = NewAuditEvent(
    session_id="sess-1", actor="master_agent", event_type="plan", payload={"steps": 3, "ok": True}
)

FACT = Fact.create(
    "flag_count",
    28,
    threshold=300,
    source="mdx_data.get_flags",
    scope={"business_date": "2026-10-15"},
)
OBSERVATION = RunObservation(
    id="obs-late-file", text="File late, inside grace period", fact_ids=(FACT.id,)
)

SAMPLES: list[BaseModel] = [
    FACT,
    OBSERVATION,
    RunContext(facts=(FACT,), observations=(OBSERVATION,)),
    NEW_EVENT,
    AuditEvent(
        seq=1,
        ts=T0,
        session_id="sess-1",
        actor="master_agent",
        event_type="plan",
        payload={"steps": 3, "ok": True},
        prev_hash=GENESIS_HASH,
        hash="b" * 64,
    ),
    ChainVerification(valid=False, events_checked=4, first_broken_seq=3, reason="gap"),
    EV,
    MEASURE,
    FINDING,
    STEP,
    PLAN,
    VALIDATION,
    TraceRef(mlflow_trace_id="abc", url=None),
    TASK,
    CHECK,
    DECISION,
    Versions(config="sha256:abc", prompts={"master_plan": "v1"}, llm_deployment="gpt-x"),
    SessionResult(
        session_id="sess-1",
        business_date=D,
        data_type="linear",
        status=SessionStatus.COMPLETED_DEGRADED,
        started_at=T0,
        finished_at=T1,
        versions=Versions(config="sha256:abc"),
        plan=PLAN,
        execution=(TASK,),
        review_checks=(CHECK,),
        decisions=(DECISION,),
        findings=(FINDING,),
        fallbacks=("explanation_template",),
    ),
    Point(curve_id="JPY", tenor=Tenor.parse("3M"), business_date=D, value=0.07, unit=Unit.PERCENT),
    Observation(business_date=D, value=0.07),
    Series(
        curve_id="JPY",
        tenor=Tenor.parse("3M"),
        kind=MarketDataKind.RATE,
        unit=Unit.PERCENT,
        observations=(
            Observation(business_date=date(2026, 10, 14), value=0.04),
            Observation(business_date=D, value=0.07),
        ),
    ),
    MdxFlag(
        flag_id="flag-1",
        business_date=D,
        data_type=DataType.LINEAR,
        kind=MarketDataKind.RATE,
        curve_id="JPY",
        tenor=Tenor.parse("3M"),
        rule="relative_change",
        previous_value=0.04,
        current_value=0.07,
        unit=Unit.PERCENT,
        color_code=ColorCode.KEY,
        comment="vendor note",
    ),
    FileMetadata(
        file_name="mdx_linear_20261015.csv",
        business_date=D,
        data_type=DataType.LINEAR,
        arrived_at=T0,
        size_bytes=20480,
        record_count=80,
        content_sha256="a" * 64,
    ),
    Case(
        case_id="case-1",
        session_id="sess-1",
        curve_id="JPY",
        tenor=Tenor.parse("3M"),
        status=CaseStatus.OPEN,
        measures=(MEASURE,),
        findings=(FINDING,),
        explanation="Threshold artifact: +3 bps.",
        explanation_source=ExplanationSource.LLM,
        evidence_ids=(EV.evidence_id,),
        created_at=T1,
    ),
]


@pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
def test_json_round_trip(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model
    # And through plain dicts (what the API and the database JSONB columns hold).
    assert type(model).model_validate(model.model_dump(mode="json")) == model


def test_every_model_is_covered() -> None:
    domain_models = {
        obj
        for module in (core_domain, mdx_domain)
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, DomainModel) and obj is not DomainModel
    }
    covered = {type(m) for m in SAMPLES}
    assert domain_models - covered == set(), "add a sample for each new domain model"
