"""Thin Streamlit UI over run_agent and proposals.py. All data shaping lives in view.py;
this file only wires widgets to that data and to the two write paths that exist at all:
run_agent (an LLM call) and proposals.decide (a line in proposals.jsonl, never Azure).

  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

from app import view
from cloudguard import proposals
from cloudguard.agent import DEFAULT_MAX_STEPS, run_agent
from cloudguard.backends import MockBackend, ResourceGraphBackend
from cloudguard.llm import DEFAULT_MODEL, GeminiClient, LLMError
from cloudguard.tools import REPO_ROOT

DEFAULT_SNAPSHOT = REPO_ROOT / "mock" / "subscription_snapshot.json"
DEFAULT_MESSAGE = "audit this subscription and propose locks for unlocked resource groups"

st.set_page_config(page_title="CloudGuard", page_icon="🛡️", layout="wide")


def _build_backend(mock: bool):
    if mock:
        return MockBackend(DEFAULT_SNAPSHOT)
    sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if not sub:
        raise RuntimeError("Live mode needs AZURE_SUBSCRIPTION_ID set (or switch to mock mode).")
    return ResourceGraphBackend(sub)


with st.sidebar:
    st.markdown("## CloudGuard")
    mock_mode = st.toggle("Mock mode", value=True, help="Off reads AZURE_SUBSCRIPTION_ID via Resource Graph.")
    max_steps = st.number_input("Max steps", min_value=1, max_value=20, value=DEFAULT_MAX_STEPS)
    message = st.text_area("Request", value=DEFAULT_MESSAGE, height=100)
    run_clicked = st.button("Run audit", type="primary")
    st.caption(f"Model: {os.environ.get('CLOUDGUARD_MODEL', DEFAULT_MODEL)}")

if run_clicked:
    st.session_state.pop("llm_error", None)
    try:
        backend = _build_backend(mock_mode)
        llm = GeminiClient()
        st.session_state["run"] = run_agent(llm, backend, message, max_steps=int(max_steps))
    except (LLMError, RuntimeError) as e:
        st.session_state["llm_error"] = str(e)
        st.session_state.pop("run", None)

if "llm_error" in st.session_state:
    st.error(f"LLM call failed: {st.session_state['llm_error']}")

run = st.session_state.get("run")

if run is None:
    st.write("Set your options in the sidebar and click **Run audit**.")
else:
    st.markdown("### Findings")
    grouped = view.group_findings(run.findings)
    if not grouped:
        st.write("No findings this run.")
    for type_name, findings in grouped.items():
        st.markdown(f"**{type_name}** ({len(findings)})")
        for f in findings:
            cols = st.columns([1, 5])
            cols[0].text(f.severity.value)
            cols[1].text(f.resource_id)
            if f.untrusted_text_flags:
                st.warning(f"Untrusted text flagged on this resource: {', '.join(f.untrusted_text_flags)}")

    missing_detectors = [name for name, ran in run.coverage.items() if not ran]
    if missing_detectors:
        st.warning(f"Detectors not run in this audit: {', '.join(missing_detectors)}")

    st.markdown("### Answer")
    st.text(run.final_answer or "(no final answer — stopped at max_steps)")

    banner = view.grounding_banner(run)
    if banner.status == "ok":
        st.success("Grounding: ok")
    elif banner.status == "not_checked":
        st.info("Grounding: not checked (no final answer)")
    else:
        st.error("Grounding: FAILED — the answer cites IDs no tool result returned this run")
        for rid in banner.ungrounded_ids:
            st.text(rid)

    st.markdown("### Pending proposals")
    all_proposals = proposals.list_proposals()
    pending, decided = view.split_proposals(all_proposals)

    if not pending:
        st.write("No pending proposals.")
    for p in pending:
        with st.container(border=True):
            st.text(f"Resource group: {p['resource_group_id']}")
            st.text(f"Lock level: {p['lock_level']}")
            st.text(f"Run: {p['run_id']}   Backend: {p['backend']}")
            st.text(f"Reason: {p['reason']}")
            if p["reason_untrusted_text_flags"]:
                st.warning(f"Untrusted text flagged in this reason: {', '.join(p['reason_untrusted_text_flags'])}")
            st.code(p["suggested_command"], language="bash")
            st.caption("Approving records a decision only. Nothing is executed in Azure; "
                       "run the suggested command yourself.")

            decided_by = st.text_input("Decided by", key=f"decided_by_{p['proposal_id']}")
            note = st.text_input("Note (optional)", key=f"note_{p['proposal_id']}")
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"approve_{p['proposal_id']}"):
                try:
                    proposals.decide(p["proposal_id"], "approved", decided_by=decided_by, note=note)
                    st.rerun()
                except ValueError as e:
                    st.warning(str(e))
            if c2.button("Reject", key=f"reject_{p['proposal_id']}"):
                try:
                    proposals.decide(p["proposal_id"], "rejected", decided_by=decided_by, note=note)
                    st.rerun()
                except ValueError as e:
                    st.warning(str(e))

    st.markdown("### Decided proposals")
    if not decided:
        st.write("No decided proposals.")
    for p in decided:
        with st.container(border=True):
            st.text(f"Resource group: {p['resource_group_id']}")
            st.text(f"Run: {p['run_id']}   Backend: {p['backend']}")
            st.text(f"Decision: {p['status']}")
            st.text(f"Decided by: {p['decided_by']}   at: {p['decided_at']}")
            if p.get("note"):
                st.text(f"Note: {p['note']}")

    st.markdown("### Run stats")
    for key, value in view.run_stats(run, model=os.environ.get("CLOUDGUARD_MODEL", DEFAULT_MODEL)).items():
        st.text(f"{key}: {value}")
