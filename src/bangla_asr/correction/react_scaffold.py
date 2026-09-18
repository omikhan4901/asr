"""
ReAct-style correction scaffold (Phase 2 stub).

The loop is real enough to produce prompts and parse Action/Thought blocks,
but the "tools" do not inspect audio or a clinical database. They only
re-read the ASR string. Treat this as an API for GRPO ``prompt_style=react``,
not as a finished correction system.

STUB LIMITS
- No tool that looks at the waveform or N-best list beyond the 1-best string.
- No clinical knowledge base.
- ``finish`` just returns the last proposed correction, or the ASR hypothesis
  if the model never proposed one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from bangla_asr.correction.reasoning_model import ReasoningModel

ALLOWED_ACTIONS = ("inspect_hypothesis", "propose_correction", "finish")


def build_react_prompt(asr_hypothesis: str, scratchpad: str = "") -> str:
    """Build the ReAct instruction prefix plus any previous scratchpad turns."""
    header = (
        "তুমি একজন বাংলা ক্লিনিকাল প্রতিলিপি সম্পাদক। "
        "ReAct ফরম্যাটে কাজ করো। অনুমোদিত অ্যাকশন: "
        "inspect_hypothesis, propose_correction, finish.\n"
        "ফরম্যাট:\n"
        "Thought: ...\n"
        "Action: <name>\n"
        "Action Input: ...\n\n"
        f"ASR hypothesis: {asr_hypothesis}\n"
    )
    if scratchpad:
        return header + "\n" + scratchpad
    return header


_ACTION_RE = re.compile(r"Action:\s*(\S+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input:\s*(.*)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: str
    observation: str = ""


@dataclass
class ReActResult:
    """Outcome of a stub ReAct episode."""

    correction: str
    steps: List[ReActStep] = field(default_factory=list)
    finished: bool = False


def parse_react_output(text: str) -> ReActStep:
    """Parse a single Thought/Action/Action Input triple from model text."""
    thought_match = _THOUGHT_RE.search(text)
    action_match = _ACTION_RE.search(text)
    input_match = _INPUT_RE.search(text)
    action = (action_match.group(1).strip().strip(".,") if action_match else "")
    return ReActStep(
        thought=thought_match.group(1).splitlines()[0].strip() if thought_match else "",
        action=action,
        action_input=input_match.group(1).strip() if input_match else "",
    )


def apply_action(step: ReActStep, asr_hypothesis: str, last_proposal: Optional[str]) -> str:
    """Execute a stub tool. Unknown actions return an error observation."""
    if step.action == "inspect_hypothesis":
        return f"ASR hypothesis is: {asr_hypothesis}"
    if step.action == "propose_correction":
        return f"Recorded proposal: {step.action_input or asr_hypothesis}"
    if step.action == "finish":
        return last_proposal or step.action_input or asr_hypothesis
    return f"Unknown action {step.action!r}. Allowed: {', '.join(ALLOWED_ACTIONS)}"


def run_react_loop(
    asr_hypothesis: str,
    model: ReasoningModel,
    max_steps: int = 4,
) -> ReActResult:
    """Run a bounded ReAct episode. This is a stub, not a production editor."""
    scratchpad = ""
    last_proposal: Optional[str] = None
    steps: List[ReActStep] = []
    for _ in range(max_steps):
        prompt = build_react_prompt(asr_hypothesis, scratchpad)
        raw = model.generate(prompt)
        step = parse_react_output(raw)
        if step.action == "propose_correction" and step.action_input:
            last_proposal = step.action_input.splitlines()[0].strip()
        observation = apply_action(step, asr_hypothesis, last_proposal)
        step.observation = observation
        steps.append(step)
        scratchpad += (
            f"Thought: {step.thought}\n"
            f"Action: {step.action}\n"
            f"Action Input: {step.action_input}\n"
            f"Observation: {observation}\n"
        )
        if step.action == "finish":
            return ReActResult(correction=last_proposal or asr_hypothesis, steps=steps, finished=True)
    return ReActResult(correction=last_proposal or asr_hypothesis, steps=steps, finished=False)
