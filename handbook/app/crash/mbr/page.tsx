import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function MbrPage() {
  return (
    <Article
      kicker="Crash course"
      title="Rescoring and MBR"
      lede="Rescoring means: the acoustic model already proposed several transcripts; we pick among them with a second rule. Minimum Bayes Risk picks the candidate that is closest, on average, to the others."
      path="/crash/mbr"
    >
      <h2>MAP vs MBR, in human words</h2>
      <p>
        The usual decode (sometimes called MAP) is: pick the sentence with
        the highest probability under the model. That is rank 0.
      </p>
      <p>
        <strong>Minimum Bayes Risk</strong> is different. Imagine each
        candidate is a person suggesting a caption. You do not elect the
        loudest person. You elect the caption that would annoy the rest of
        the group the least. “Annoy” here is word error between two
        candidates — we never look at the ground truth when choosing.
      </p>
      <p>
        In symbols, if you want them: pick hypothesis <em>i</em> that
        minimizes Σ<sub>j</sub> P(h<sub>j</sub> | audio) · WER(h<sub>i</sub>,
        h<sub>j</sub>). We turn the stored log-probs into P with a softmax
        whose temperature is <strong>0.5</strong> on the published path.
        Temperature below 1 sharpens: we trust the acoustic scores more.
        Temperature above 1 flattens: we trust the consensus more.
      </p>

      <h2>What we are allowed to claim</h2>
      <p>
        On our eval set, this rule moved WER from 34.85% to 32.29% and
        recovered about 42% of the oracle gap. Zero training. That is a
        real systems result. It is also, according to the supervisor, not
        enough novelty by itself — because MBR on N-best lists is a known
        idea. The contribution would have to be the{" "}
        <em>Bangla long-form characterization</em>, or a new metric, or a
        new training loop that uses that metric.
      </p>

      <Callout kind="warn">
        <p>
          CTC beam scores are not Whisper log-probs. Feeding them to the
          same temperature-0.5 softmax is an experiment, not the published
          method. <code>pipeline/09_compare_acoustic_models.py</code> lets
          you set a separate CTC temperature on purpose.
        </p>
      </Callout>

      <h2>The other rescorer in the repo</h2>
      <p>
        There is also a BanglaBERT reranker that <em>is</em> trained: it
        sees a candidate plus a couple of acoustic features and learns to
        predict error. That is a third column in{" "}
        <code>pipeline/08_evaluate_benchmarks.py</code>. It is not an
        acoustic model. Do not delete it because the new SWER story does
        not mention it; it is part of the existing thesis spine.
      </p>

      <Sources
        items={[
          {
            title: "Kumar & Byrne, Minimum Bayes-Risk Decoding for ASR (2002/2004)",
            why: "The classic speech MBR papers. Skim the objective; skip the old lattice machinery.",
          },
          {
            title: "Goel & Byrne, Minimum Bayes-risk automatic speech recognition (CSL 2000)",
            why: "Earlier CSL paper if you need a citation chain.",
          },
          {
            title: "Our mbr.py docstring and tests/test_mbr.py",
            why: "The exact temperature, NaN handling, and empty-candidate behavior we ship.",
          },
        ]}
      />
    </Article>
  );
}
