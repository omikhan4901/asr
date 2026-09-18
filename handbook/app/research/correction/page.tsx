import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function CorrectionPage() {
  return (
    <Article
      kicker="Our research"
      title="Correction and GRPO"
      lede="A small language model reads the ASR hypothesis and writes a repaired sentence. If we train it, we pay it for SWER improvement, not for looking literary."
      path="/research/correction"
    >
      <h2>The job, if we ever turn it on</h2>
      <p>
        Input: the ASR string (and later, maybe the N-best list). Output:
        a corrected string. Reward:
      </p>
      <pre>
        <code>reward = SWER(asr, gt) − SWER(correction, gt)</code>
      </pre>
      <p>
        Then scale and clip from config. Empty ground-truth examples are{" "}
        <strong>dropped from the batch</strong> (and counted in the log)
        instead of scored as 0, because a zero still dilutes the mean.
      </p>

      <h2>What exists in code</h2>
      <ul>
        <li><code>training/reward.py</code> — the function. Unit-tested for monotonicity on synthetic pairs.</li>
        <li><code>training/train_grpo.py</code> — the loop: sample a group, advantages, optional KL to a frozen copy.</li>
        <li><code>training/modal_train.py</code> — Modal A10G job, same volume pattern as N-best.</li>
        <li><code>correction/react_scaffold.py</code> — a stub that parses Thought / Action / Observation. Tools do not see audio or a knowledge base.</li>
      </ul>

      <h2>What does not exist yet</h2>
      <ul>
        <li>A confirmed Hugging Face id for “VibeThinker-3B”. The string in config may be wrong.</li>
        <li>Any training log from a real GPU run.</li>
        <li>Proof that the model can edit Bangla clinical text at all.</li>
      </ul>

      <Callout kind="danger">
        <p>
          The group rule: no GRPO run, even a smoke run, until the SWER
          sanity fixture has real labels and the rank-order test passes.
          A wrong reward is worse than no training.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "DeepSeekMath / GRPO (arXiv:2402.03300)",
            href: "https://arxiv.org/abs/2402.03300",
            why: "The algorithm we implemented a teaching-size version of.",
          },
          {
            title: "ReAct (Yao et al., arXiv:2210.03629)",
            href: "https://arxiv.org/abs/2210.03629",
            why: "The prompting pattern the stub is named after. We have not reproduced their agent.",
          },
        ]}
      />
    </Article>
  );
}
