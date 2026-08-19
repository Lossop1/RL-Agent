"""Operator Playbook — the LLM brain's SYSTEMATIZED operating knowledge, split into a ROBOT-AGNOSTIC
core (the disciplined workflow + grounding rules that hold for ANY robot tuning campaign) and a
PLUGGABLE per-robot PROFILE (the gate→lever map, benchmark shape, ops signatures, regime constants that
are specific to one robot). The system is ultimately for many robots, so the workflow must NOT hardcode
one robot's facts.

get_playbook(task, gate, robot) resolves: the agnostic workflow for the task + the active robot's profile
slice needed to execute it — the machine-actionable knowledge, retrieved by task, not a wall of prose.

Design intent (learned from the Taili campaign, 0706): the agent's failures were all "acted on a wrong or
unverified belief and burned a long training iteration." So the CORE discipline is: ground every claim in a
tool result, VERIFY a measurement before spending on it, never invent a lever, never repeat a rolled-back one.
That discipline is robot-agnostic; the levers/gates it operates on are per-robot data.
"""
from __future__ import annotations

from typing import Any

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ROBOT-AGNOSTIC CORE — holds for any robot / any tuning campaign
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# Grounding rules are PRINCIPLES; the numeric parameters they cite ({iter_cost}, {peak_steps}, gate names)
# come from the active robot profile and are interpolated at retrieval time.
GROUNDING_RULES = [
    "FACT-BEFORE-CLAIM: never assert a number, a gate result, or 'it climbed/stalled/regressed' without a "
    "tool result IN THIS TURN that shows it. No data → get it; do not infer, recall, or guess.",
    "VERIFY-BEFORE-BURN: a training iteration is expensive ({iter_cost}). Before spending it, the measurement "
    "driving the decision must PASS the verify_measurement workflow. Acting on an unverified belief is the #1 waste.",
    "NO-INVENTED-LEVERS: only apply a lever the active robot profile lists for that gate. No known lever → say so; "
    "do not guess a weight to bump (weight-immune levers were bumped repeatedly with zero effect).",
    "NO-REPEAT: before applying a lever, check the campaign journal — if tried and rolled back, do not repeat it.",
    "RIGHT-OBJECTIVE: act against the SCORED gate as the profile defines it (e.g. ascending vs descending are "
    "different gates). Optimising the wrong direction wastes the whole iteration.",
]

# The disciplined state machine. Steps reference 'the active robot profile' abstractly — no robot's facts here.
WORKFLOWS = {
    "run_tuning_loop": {
        "goal": "Push the benchmark up, one VERIFIED lever at a time. Ordered; guards are mandatory.",
        "steps": [
            {"n": 1, "do": "Establish ground truth: current per-gate verdict.",
             "tool": "get_acceptance (run_acceptance if stale/absent)", "produces": "gate verdict",
             "guard": "verdict is a FULL battery on the BEST checkpoint, training STOPPED, checkpoint at the "
                      "profile's peak_steps (not overtrained).",
             "on_fail": "re-run the full battery on a clean peak-steps checkpoint with training stopped."},
            {"n": 2, "do": "VERIFY the measurement before trusting it (run the verify_measurement workflow).",
             "tool": "get_playbook(task='verify_measurement')", "produces": "trusted / rejected",
             "guard": "analysis method correct, terrain matches the gate, no confounds.",
             "on_fail": "re-measure correctly; NEVER proceed on a suspect number."},
            {"n": 3, "do": "Select the target gate + lever from the profile.",
             "tool": "analyze_acceptance + get_playbook(gate=...)", "produces": "hypothesis (gate, lever, expected)",
             "guard": "lever is in the profile's gate_levers for that gate AND not tried-and-rolled-back (journal). "
                      "Prefer the highest-confidence tractable gate the profile marks.",
             "on_fail": "pick the next-ranked gate; never invent a lever."},
            {"n": 4, "do": "Apply the bounded config edit.",
             "tool": "apply_tuning", "produces": "config diff + journal entry (hypothesis logged)",
             "guard": "one lever this iteration; within allowlist bounds; autonomy tier permits.",
             "on_fail": "propose to the human if the tier is advisory/destructive."},
            {"n": 5, "do": "Train from BEST checkpoint to the profile's peak_steps.",
             "tool": "deploy_payload + start_training", "produces": "a run driven by the self-heal driver",
             "guard": "resume-phase restored per profile; driver KILLS at target (no overtrain); NO diagnostic "
                      "runs concurrently (memory-limit OOM).",
             "on_fail": "self-heal driver recovers a GPU/kernel stall (fresh-log check, not just process-alive)."},
            {"n": 6, "do": "Re-measure the full benchmark.", "tool": "run_acceptance (full battery)",
             "produces": "new verdict", "guard": "training STOPPED first; re-run step 2 (verify) on it.",
             "on_fail": "re-measure."},
            {"n": 7, "do": "Decide keep/rollback; log the outcome.",
             "tool": "compare + rollback_tuning if needed", "produces": "journal entry (confirmed/refuted)",
             "guard": "KEEP only if the target gate improved AND no passing gate regressed; else ROLLBACK.",
             "on_fail": "rollback_tuning and return to step 3 with the next lever."},
        ],
        "then": "loop to step 1.",
    },
    "verify_measurement": {
        "goal": "The anti-guess gate. Confirm a result is trustworthy BEFORE any decision rides on it.",
        "steps": [
            {"n": 1, "do": "Check the analysis method.",
             "guard": "each metric is computed over the RIGHT window/reduction for what it means (e.g. a "
                      "traversal/height metric over the FULL trajectory, not a leading slice; a settled metric "
                      "over settled frames). A wrong window silently fabricates a result."},
            {"n": 2, "do": "Check terrain/objective matches the gate.",
             "guard": "the run's condition is the one the gate actually scores (per the profile's gate defs)."},
            {"n": 3, "do": "Check for confounds.",
             "guard": "not overtrained, training stopped (no memory contention), the eval harness actually ran "
                      "all intended cases (know your harness's failure modes from the profile)."},
        ],
        "verdict": "Any guard fails → measurement REJECTED, re-measure correctly. Never tune on it.",
    },
    "diagnose_training": {
        "goal": "Answer 'how is training / is it stuck' from facts, not guesses.",
        "steps": [
            {"n": 1, "do": "Read the system's own state.", "tool": "get_operations_state",
             "guard": "read training_progress (is_resume, from-checkpoint, phase, loaded_payload, regime)."},
            {"n": 2, "do": "Match the pattern to a known signature.", "tool": "get_playbook(task='stall')",
             "guard": "the (step-age, process-alive, gpu) pattern matches a profile ops_signature."},
            {"n": 3, "do": "State the codified diagnosis+action.",
             "guard": "quote the signature; do NOT invent a new mechanism."},
        ],
    },
}


def build_playbook(
    *,
    task: str = "",
    gate: str = "",
    robot: str = "",
    profiles: dict[str, dict[str, Any]],
    default_robot: str,
) -> dict[str, Any]:
    """组合通用工作流与调用方提供的产品档案。"""
    robot = (robot or default_robot).lower()
    prof = profiles.get(robot, profiles[default_robot])
    q = (task or "").lower()
    regime = prof.get("regime", {})
    rules = [r.format(iter_cost=regime.get("iter_cost", "a long run"),
                      peak_steps=regime.get("peak_steps", "the peak")) for r in GROUNDING_RULES]
    out: dict[str, Any] = {"kind": "operator_playbook", "robot": robot, "robot_label": prof.get("label"),
                           "benchmark": prof.get("benchmark"), "regime": regime, "grounding_rules": rules}

    if gate:
        gl = prof.get("gate_levers", {})
        out["gate_lever"] = {gate: gl.get(gate)} if gate in gl else \
            {"note": f"{gate} not in {robot} profile; near: {[k for k in gl if gate.split('[')[0] in k]}"}

    picked = None
    for name, wf in WORKFLOWS.items():
        if name in q or any(w in q for w in name.split("_")):
            out["workflow"] = {name: wf}; picked = name; break
    if picked is None and q:
        gl = prof.get("gate_levers", {})
        hits = {g: v for g, v in gl.items() if q in g.lower() or q in v.get("symptom", "").lower()}
        if hits: out["gate_levers"] = hits
    if picked is None and not gate and not out.get("gate_levers"):
        out["workflows_available"] = {n: w["goal"] for n, w in WORKFLOWS.items()}

    if any(k in q for k in ("stall", "stuck", "stop", "frozen", "oom", "crash", "train", "diagnose")):
        out["ops_signatures"] = prof.get("ops_signatures")
        out["harness_gotchas"] = prof.get("harness_gotchas")
    out["instruction"] = ("Robot-AGNOSTIC workflow + this robot's PROFILE. Execute steps with the named tools, "
                          "obey grounding_rules and each step's guard (do not advance on a guess), apply only "
                          "profile-listed levers. Another robot = another profile, same workflow. "
                          "Retrieve more: get_playbook(task=..., gate=..., robot=...).")
    return out


def get_playbook(task: str = "", gate: str = "", robot: str = "", *, product_id: str | None = None) -> dict[str, Any]:
    """按产品合同加载 playbook 插件；旧调用仍可通过当前合同解析。"""
    from autotuner.product import load_product_plugin, resolve_product_contract

    contract = resolve_product_contract(product_id)
    plugin = load_product_plugin(contract, "playbook", "get_playbook")
    return plugin(task=task, gate=gate, robot=robot)
