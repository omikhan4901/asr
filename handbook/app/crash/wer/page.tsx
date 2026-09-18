import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function WerPage() {
  return (
    <Article
      kicker="Crash course"
      title="WER and CER"
      lede="Word Error Rate is the industry default grade. It treats every word as equally important. That is its power and its scientific limit."
      path="/crash/wer"
    >
      <h2>The exam analogy</h2>
      <p>
        Reference (answer key): <code>আমি ইনসুলিন নেব না</code>
        <br />
        Hypothesis (student): <code>আমি ইনসুলিন নেব</code>
      </p>
      <p>
        WER asks: how many word-level edits (insert, delete, substitute)
        turn the student text into the key, divided by the number of words
        in the key? Here we deleted “না”. That is 1 edit / 4 words = 25%
        WER. A spelling-quiz teacher shrugs. A clinician does not.
      </p>

      <h2>The algorithm has a name you know</h2>
      <p>
        Those edits are <strong>Levenshtein distance</strong> (the same
        dynamic program as in algorithms class) run on tokens after we
        split on spaces. We first run <code>normalize_bangla_text</code> so
        that punctuation and nukta variants do not create fake errors.
      </p>
      <p>
        <strong>CER</strong> is the same idea on characters, not words. It
        is useful when word boundaries are messy. We report both. The
        headline number in the paper-shaped writeup is WER.
      </p>

      <h2>Corpus WER is not the average of utterance WERs</h2>
      <p>
        We add all the edits on the set, then divide by all the reference
        words on the set. A 3-word clip and a 40-word clip are not equals.
        Our <code>compute_wer</code> uses <code>jiwer</code> after dropping
        empty references (dividing by zero is not a metric).
      </p>

      <Callout kind="warn">
        <p>
          Papers have been yelling that WER is a bad proxy for usefulness.
          “WER is Unaware” (2025) is the recent one we cite by name. Roy
          2021 defined <em>Semantic</em> WER. We are proposing something
          else: a <em>clinical severity</em> table on the same alignment.
          Do not call SWER “semantic WER” in a slide.
        </p>
      </Callout>

      <h2>Oracle WER</h2>
      <p>
        If the model gives 12 guesses, and we secretly look at the answer
        key and pick the guess with lowest WER, that is the{" "}
        <strong>oracle</strong>. It is not a system you can ship. It is
        the ceiling of “how good could rescoring be if it were perfect?”
        Our oracle is 28.72%. The 1-best is 34.85%. The gap is the
        headroom MBR tries to climb.
      </p>

      <Sources
        items={[
          {
            title: "Levenshtein, Binary codes capable of correcting deletions… (1966)",
            why: "The original edit-distance paper. Historical; any algorithms book is enough in practice.",
          },
          {
            title: "jiwer documentation",
            href: "https://jitsi.github.io/jiwer/",
            why: "The library we call for corpus WER/CER.",
          },
          {
            title: "“WER is Unaware” (arXiv:2511.16544)",
            href: "https://arxiv.org/abs/2511.16544",
            why: "Recent argument that WER misses meaning. Read the abstract before you invent a new metric.",
          },
          {
            title: "Roy et al., Semantic WER for medical ASR (2021)",
            why: "The paper whose name we must not collide with. Semantic similarity ≠ our clinical multipliers.",
          },
        ]}
      />
    </Article>
  );
}
