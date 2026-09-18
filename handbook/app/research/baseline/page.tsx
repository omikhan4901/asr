import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function BaselinePage() {
  return (
    <Article
      kicker="Our research"
      title="The published baseline"
      lede="34.85% → 32.29% on 2,271 utterances. This page is the contract. If a change moves these numbers, it is not a refactor, it is a new experiment."
      path="/research/baseline"
    >
      <h2>The exact recipe</h2>
      <ul>
        <li>Model: regional Whisper-medium, language <code>bn</code>, task <code>transcribe</code>.</li>
        <li>N-best: 12 sequences, sampling temperature 0.7, length penalty 1.5, batch size 2.</li>
        <li>MBR temperature 0.5, pairwise loss = word error between candidates, empty strings penalized.</li>
        <li>1-best = rank 0 of that list. Oracle = lowest WER vs ground truth inside the list.</li>
      </ul>
      <p>
        Headroom recovery = (1-best − MBR) / (1-best − oracle). Plug in
        the WER fractions: (0.3485 − 0.3229) / (0.3485 − 0.2872) ≈ 0.418.
      </p>

      <h2>What you must not “improve”</h2>
      <p>
        Do not change those defaults to make the code prettier. Do not
        switch to grouped diverse beam search and keep the same table
        caption. Do not normalize text differently and compare to 34.85.
        There is a second experiment for every one of those ideas.
      </p>

      <Callout kind="ok">
        <p>
          The lock is <code>WhisperConfig</code> and <code>MBRConfig</code>{" "}
          in <code>src/bangla_asr/config.py</code>, plus{" "}
          <code>tests/test_smoke.py</code> and <code>tests/test_mbr.py</code>.
          If smoke fails after your edit, you broke the contract.
        </p>
      </Callout>

      <h2>How this sits next to LoRA</h2>
      <p>
        Bhuiyan et al. (arXiv:2605.08214) report a LoRA fine-tune around
        31.32% WER. We are not claiming we beat them. We are claiming
        that a no-training consensus rule closed a surprising fraction of
        the remaining gap on <em>this</em> split. A reviewer may still
        say “just fine-tune.” That is a fair comment, and it is why
        fine-tuning-plus-rescoring is an open research angle.
      </p>

      <Sources
        items={[
          {
            title: "Bhuiyan et al., arXiv:2605.08214",
            href: "https://arxiv.org/abs/2605.08214",
            why: "The trained Bangla number in our introduction.",
          },
          {
            title: "pipeline/05_rescore_mbr.py and pipeline/08_evaluate_benchmarks.py",
            why: "The commands that reproduce the table when the eval JSONL is present.",
          },
        ]}
      />
    </Article>
  );
}
