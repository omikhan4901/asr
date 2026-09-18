import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function WhisperPage() {
  return (
    <Article
      kicker="Crash course"
      title="Whisper (autoregressive)"
      lede="Whisper looks at a spectrogram of the clip, then writes the transcript one token at a time. Each new token is chosen given everything it has already written. That is why its mistakes often look like fluent language."
      path="/crash/whisper"
    >
      <h2>Encoder, then decoder</h2>
      <p>
        Think of two students. The <strong>encoder</strong> listens to the
        whole clip and builds a memory of the sound. The{" "}
        <strong>decoder</strong> writes the sentence, left to right, peeking
        at that memory and at the words it already wrote. “Autoregressive”
        means: the next word depends on the previous words.
      </p>
      <p>
        Because the decoder is a language model in disguise, it prefers
        sentences that look like text it has seen. If the audio is muddy,
        it may still write a polished wrong sentence. People call that{" "}
        <strong>hallucination</strong> in this literature: extra or invented
        fluent content that was not in the audio.
      </p>

      <h2>The checkpoint we actually load</h2>
      <p>
        Default id:{" "}
        <code>bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium</code>.
        That is a Bangla-adapted Whisper-medium, not the vanilla OpenAI
        English-heavy checkpoint. We set language <code>bn</code> and task{" "}
        <code>transcribe</code>. Some older regional configs are missing
        modern language-id fields; <code>whisper_model.py</code> patches
        those from a fallback <code>openai/whisper-medium</code> config
        without changing the weights’ Bangla training.
      </p>

      <h2>Greedy 1-best vs the research path</h2>
      <p>
        For alignment we sometimes ask Whisper for a single greedy
        transcript. For the published numbers we ask it for an{" "}
        <strong>N-best list</strong> (next page). The official “1-best WER
        34.85%” is the rank-0 item of that list, not a separate secret
        decode. If you invent a new 1-best definition, the baseline moves
        and we can no longer claim 34.85.
      </p>

      <Callout kind="ok">
        <p>
          Wrapper classes: <code>WhisperASRModel</code> (loader) and{" "}
          <code>WhisperModel</code> (the shared <code>ASRModel</code>{" "}
          interface). Do not rewrite the generate kwargs on the published
          path. <code>num_beams=12</code>, sampling temperature{" "}
          <code>0.7</code>, length penalty <code>1.5</code>.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)",
            href: "https://arxiv.org/abs/2212.04356",
            why: "The paper that introduced Whisper. Sections 2–3 are enough.",
          },
          {
            title: "Hugging Face Whisper docs",
            href: "https://huggingface.co/docs/transformers/model_doc/whisper",
            why: "How generate(), language, and task tokens work in the library we call.",
          },
        ]}
      />
    </Article>
  );
}
