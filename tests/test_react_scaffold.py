"""ReAct stub parses actions and can be used as a GRPO prompt style."""

from __future__ import annotations

from bangla_asr.correction.react_scaffold import (
    build_react_prompt,
    parse_react_output,
    run_react_loop,
)
from bangla_asr.correction.reasoning_model import ReasoningModel
from bangla_asr.training.train_grpo import build_correction_prompt


class _ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, prompt: str, max_new_tokens=None) -> str:
        return self.outputs.pop(0)


def test_parse_react_output() -> None:
    step = parse_react_output(
        "Thought: negation may be flipped\n"
        "Action: propose_correction\n"
        "Action Input: উচ্চরক্তচাপ আছে\n"
    )
    assert step.action == "propose_correction"
    assert "উচ্চরক্তচাপ" in step.action_input


def test_react_loop_finishes_with_proposal() -> None:
    scripted = _ScriptedModel(
        [
            "Thought: inspect\nAction: inspect_hypothesis\nAction Input:\n",
            "Thought: fix\nAction: propose_correction\nAction Input: উচ্চরক্তচাপ আছে\n",
            "Thought: done\nAction: finish\nAction Input:\n",
        ]
    )
    # Bypass HF load by injecting a dummy tokenizer/model pair, then overwrite generate.
    reasoner = ReasoningModel(model=object(), tokenizer=object())
    reasoner.generate = scripted.generate  # type: ignore[method-assign]
    result = run_react_loop("নিম্নরক্তচাপ আছে", reasoner, max_steps=4)
    assert result.finished is True
    assert result.correction == "উচ্চরক্তচাপ আছে"


def test_grpo_react_prompt_style_imports_scaffold() -> None:
    prompt = build_correction_prompt("আমি গান গাই", prompt_style="react")
    assert "Action:" in prompt
    assert "আমি গান গাই" in prompt
    assert build_react_prompt("x")


if __name__ == "__main__":
    test_parse_react_output()
    test_react_loop_finishes_with_proposal()
    test_grpo_react_prompt_style_imports_scaffold()
    print("[PASS] ReAct scaffold tests")
