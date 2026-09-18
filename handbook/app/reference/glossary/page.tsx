import { Article } from "@/components/Article";

export default function GlossaryPage() {
  return (
    <Article
      kicker="Reference"
      title="Glossary"
      lede="If a meeting word bounced off you, find it here, then go back to the crash-course page."
      path="/reference/glossary"
    >
      <p>
        <strong>Acoustic model.</strong> The network that looks at audio
        and proposes text. Whisper and Wav2Vec2 are two acoustic models.
      </p>
      <p>
        <strong>ASR.</strong> Automatic Speech Recognition. Speech in, text out.
      </p>
      <p>
        <strong>Autoregressive.</strong> Writes the next token given the
        previous tokens. Whisper’s decoder is autoregressive.
      </p>
      <p>
        <strong>Beam search.</strong> Keep the top B partial transcripts
        while decoding, instead of one greedy path.
      </p>
      <p>
        <strong>CER.</strong> Character Error Rate. Levenshtein on characters.
      </p>
      <p>
        <strong>CTC.</strong> Connectionist Temporal Classification. A
        training and decoding style that labels frames, then collapses
        blanks and repeats.
      </p>
      <p>
        <strong>Fine-tuning.</strong> Updating weights on your data.
      </p>
      <p>
        <strong>Garbled span.</strong> A stretch of hypothesis text that
        is not a real word / is phonetically collapsed. SWER’s expensive
        bin for “meaning destroyed.”
      </p>
      <p>
        <strong>GRPO.</strong> Group Relative Policy Optimization. Sample
        several answers, score them, push toward the better ones in the
        group. No critic network.
      </p>
      <p>
        <strong>Ground truth / reference.</strong> The text we treat as
        correct for scoring.
      </p>
      <p>
        <strong>Hallucination (ASR sense).</strong> Fluent extra or invented
        content, typical of decoder models, not a medical hallucination.
      </p>
      <p>
        <strong>Hypothesis.</strong> The model’s text. Also our dataclass
        for one N-best item.
      </p>
      <p>
        <strong>JSONL.</strong> A file with one JSON object per line. Our
        N-best handshake.
      </p>
      <p>
        <strong>KenLM.</strong> A classical n-gram language model often
        plugged into CTC beam search. Optional here.
      </p>
      <p>
        <strong>LoRA.</strong> Low-Rank Adaptation. Cheap fine-tuning of
        large models.
      </p>
      <p>
        <strong>MAP decode.</strong> Pick the highest-probability sentence.
        Roughly “1-best.”
      </p>
      <p>
        <strong>MBR.</strong> Minimum Bayes Risk. Pick the candidate with
        lowest expected pairwise loss under the other candidates.
      </p>
      <p>
        <strong>Modal.</strong> The GPU cloud we already use for long jobs.
      </p>
      <p>
        <strong>N-best.</strong> The top N transcripts, not just one.
      </p>
      <p>
        <strong>Oracle.</strong> If we peek at the reference and pick the
        best candidate in the list. A ceiling, not a system.
      </p>
      <p>
        <strong>Rescoring.</strong> Re-ranking an existing N-best list.
      </p>
      <p>
        <strong>SWER.</strong> Severity-Weighted Error Rate. Our draft
        clinical-weighted WER. Not Semantic WER.
      </p>
      <p>
        <strong>Token.</strong> A piece the model writes. For WER we mean
        a space-separated word after normalization.
      </p>
      <p>
        <strong>Utterance.</strong> One scored clip.
      </p>
      <p>
        <strong>VAD.</strong> Voice Activity Detection. Finding where
        speech is.
      </p>
      <p>
        <strong>WER.</strong> Word Error Rate. Edits / reference words.
      </p>
      <p>
        <strong>XLS-R.</strong> A multilingual wav2vec 2.0 family. Our
        Bangla CTC checkpoints start from it.
      </p>
    </Article>
  );
}
