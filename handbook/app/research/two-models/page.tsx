import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function TwoModelsPage() {
  return (
    <Article
      kicker="Our research"
      title="Two acoustic models"
      lede="Whisper and Wav2Vec2 must speak the same interface so MBR, SWER, and any later corrector do not care who produced the text."
      path="/research/two-models"
    >
      <h2>The interface</h2>
      <p>
        <code>ASRModel.transcribe_nbest(audio, n)</code> returns a list of{" "}
        <code>Hypothesis(text, logprob, rank)</code>.{" "}
        <code>to_candidate_dict()</code> is exactly what MBR already ate.
        Factory: <code>build_asr_model("whisper" | "wav2vec2_ctc")</code>.
      </p>

      <h2>Why a second model is a scientific instrument</h2>
      <p>
        If we only ever measure WER, “Whisper is 3 points better” is a
        leaderboard. If we measure SWER, we can ask: does Whisper’s extra
        fluency <em>create</em> expensive substitutions (wrong but real
        drug names), while CTC creates expensive garble (unreadable
        collapse of the same drug)? Those are different papers, and they
        need both models on the <em>same</em> held-out set.
      </p>

      <h2>Comparison rules we already encoded</h2>
      <ul>
        <li>Whisper 1-best = rank 0 (published definition).</li>
        <li>CTC 1-best = <code>greedy_text</code> when present, because a KenLM beam-1 is not greedy.</li>
        <li>Separate MBR temperatures. Do not silently reuse 0.5 on CTC scores.</li>
        <li>JSONL files tagged with <code>model_family</code> so they cannot be concatenated by accident.</li>
      </ul>

      <Callout kind="warn">
        <p>
          The dual-model table has not been run on the 2,271-utt set in
          this environment. <code>pipeline/09_compare_acoustic_models.py</code>{" "}
          is ready. The CTC JSONL is not sitting in this clone until
          someone generates it.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "XLS-R (arXiv:2111.09296) and the arijitx Bangla checkpoint card",
            href: "https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali",
            why: "What the CTC track actually is.",
          },
          {
            title: "OpenSLR-53 (Bengali ASR corpus)",
            href: "https://www.openslr.org/53/",
            why: "The fine-tune data behind many public Bangla CTC models.",
          },
        ]}
      />
    </Article>
  );
}
