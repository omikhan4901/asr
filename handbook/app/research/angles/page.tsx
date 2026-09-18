import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function AnglesPage() {
  return (
    <Article
      kicker="Our research"
      title="Potential research angles"
      lede="Not a work board. These are directions a semester can still choose among. Each one is a different paper if it is done honestly."
      path="/research/angles"
    >
      <p>
        The supervisor already closed one door: “we ran MBR on Whisper
        N-best” as the <em>only</em> claim. The doors below are still
        open. Pick one and go deep rather than collecting all of them at
        20%.
      </p>

      <h2>1. SWER as a metric paper</h2>
      <p>
        <strong>Claim shape:</strong> WER under-penalizes clinical damage
        in Bangla transcripts; an explicit severity table on Levenshtein
        alignment ranks errors closer to expert judgment.
      </p>
      <p>
        <strong>What would have to become real:</strong> a lexicon built
        with a clinician or a standard drug/negation list; the sanity
        fixture filled with double-annotated examples; inter-annotator
        agreement; a clear statement vs Roy 2021 and “WER is Unaware.”
      </p>
      <p>
        <strong>Risk:</strong> if humans disagree with the weights, the
        metric is a story, not a contribution.
      </p>

      <h2>2. Error morphology: Whisper vs CTC</h2>
      <p>
        <strong>Claim shape:</strong> under the same rescoring pipeline,
        autoregressive and CTC models spend their remaining SWER on
        different edit types (fluent dangerous subs vs garbled spans).
      </p>
      <p>
        <strong>What would have to become real:</strong> CTC N-best on the
        full eval set; a per-category SWER breakdown; maybe KenLM on vs
        off as a sensitivity study (and an honest caption if LM is on).
      </p>
      <p>
        <strong>Risk:</strong> if both models just look “worse on rare
        words,” there is no morphology story.
      </p>

      <h2>3. Fine-tune, then rescore</h2>
      <p>
        <strong>Claim shape:</strong> LoRA (or full) fine-tuning and MBR
        are not substitutes; stacking them recovers more oracle headroom
        than either alone.
      </p>
      <p>
        <strong>What would have to become real:</strong> a train split
        that does not leak; a compute budget; a freeze of decode settings
        so the comparison is fair with 34.85.
      </p>
      <p>
        <strong>Risk:</strong> reviewers say “of course fine-tuning helps”
        unless the interaction with N-best diversity is surprising.
      </p>

      <h2>4. SWER-trained correction</h2>
      <p>
        <strong>Claim shape:</strong> a small corrector optimized for
        SWER improvement reduces dangerous errors more than a corrector
        optimized for WER, at similar WER.
      </p>
      <p>
        <strong>What would have to become real:</strong> a frozen SWER;
        a real policy checkpoint; ablations (reward = WER delta vs SWER
        delta); a tiny compute protocol so the result is not “we trained
        until the weekend ended.”
      </p>
      <p>
        <strong>Risk:</strong> the 3B model cannot edit Bangla well enough
        for the reward to matter. That is still a finding if you measure it.
      </p>

      <h2>5. The lexicon as an artifact</h2>
      <p>
        <strong>Claim shape:</strong> a reusable Bangla clinical severity
        table, with guidelines for insertions vs deletions, released with
        the code.
      </p>
      <p>
        <strong>What would have to become real:</strong> domain reference
        material, not ChatGPT pharmacology. Versioning. A process for
        adding a term.
      </p>

      <h2>6. English clinical as a parallel track (maybe never)</h2>
      <p>
        MedDialog-EN-ASR and PriMock-style audio would let us ask whether
        SWER’s idea transfers. The loaders are named and then they raise.
        Doing this <em>instead of</em> Bangla would abandon the baseline
        we already paid for. Doing it <em>in addition</em> is a second
        thesis if someone has time.
      </p>

      <h2>7. Reranker that knows SWER</h2>
      <p>
        The BanglaBERT reranker today targets empirical WER. Training it
        to predict SWER, or to pick the candidate with lowest expected
        SWER, is a small, well-scoped experiment that reuses a lot of
        existing code.
      </p>

      <Callout kind="note">
        <p>
          When the group is ready for a work board, each angle above
          becomes a milestone list. Until then, use this page to argue
          in meetings: “I think we should pick (2) because the CTC JSONL
          is the missing measurement, not more training.”
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "How to write a related-work section (any CHI / ACL mentoring guide)",
            href: "https://aclrollingreview.org/reviewerguidelines",
            why: "Reviewers reward a clear gap. Each angle above is a gap sentence waiting for a citation stack.",
          },
          {
            title: "The sources library in this handbook",
            why: "Start from papers we already named so the group does not fork into random blogs.",
          },
        ]}
      />
    </Article>
  );
}
