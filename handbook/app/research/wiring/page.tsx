import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function WiringPage() {
  return (
    <Article
      kicker="Our research"
      title="How the code is wired"
      lede="One story from a raw recording to a number on a slide. If you can retell this page, you understand the repository."
      path="/research/wiring"
    >
      <h2>The long path</h2>
      <div className="flow">
        <span>Raw audio + transcript</span>
        <i>→</i>
        <span>VAD chunks</span>
        <i>→</i>
        <span>Fuzzy align</span>
        <i>→</i>
        <span>Group split</span>
      </div>
      <div className="flow">
        <span>Eval clips</span>
        <i>→</i>
        <span>Whisper N-best JSONL</span>
        <i>→</i>
        <span>MBR / BERT / oracle</span>
        <i>→</i>
        <span>34.85 / 32.29 / 28.72</span>
      </div>
      <div className="flow">
        <span>Same eval clips</span>
        <i>→</i>
        <span>Wav2Vec2 N-best JSONL</span>
        <i>→</i>
        <span>compare_models</span>
        <i>→</i>
        <span>WER + SWER table</span>
      </div>
      <div className="flow">
        <span>ASR text + reference</span>
        <i>→</i>
        <span>correction model</span>
        <i>→</i>
        <span>SWER delta reward</span>
        <i>→</i>
        <span>GRPO (not run)</span>
      </div>

      <h2>Shared objects</h2>
      <ul>
        <li>
          <strong>Hypothesis / candidate dict.</strong> The coin of the
          realm. Everything that picks among transcripts uses{" "}
          <code>text</code>, <code>logprob</code>, <code>rank</code>.
        </li>
        <li>
          <strong>normalize_bangla_text.</strong> WER, MBR pairwise loss,
          and SWER all go through it. Two metrics that skip it are not
          comparable.
        </li>
        <li>
          <strong>align_tokens.</strong> WER uses the distance. SWER uses
          the same table plus a backtrace. Do not write a second aligner.
        </li>
        <li>
          <strong>compute_swer.</strong> Evaluation and the GRPO reward
          import the same function. If they ever drift, the project is
          lying to itself.
        </li>
      </ul>

      <h2>Where Modal sits</h2>
      <p>
        Long GPU jobs (Whisper N-best, Wav2Vec2 N-best, future GRPO) are
        the same idea: a Debian image, <code>PYTHONPATH=/root/src</code>,
        the <code>bangla-asr-data</code> volume, A10G, shards, then a
        merge. We did not invent a second cloud style. If you add a job,
        copy that pattern.
      </p>

      <Callout kind="note">
        <p>
          The web subtitle app under <code>web/</code> is a demo that
          also does VAD → Whisper → MBR. It is not the research eval
          loop. Do not tune the paper numbers from the demo UI.
        </p>
      </Callout>
    </Article>
  );
}
