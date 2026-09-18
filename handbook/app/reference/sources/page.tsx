import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function SourcesLibraryPage() {
  return (
    <Article
      kicker="Reference"
      title="Sources library"
      lede="A single shelf. Pages also list the two or three links you need in the moment. Use this when you write related work."
      path="/reference/sources"
    >
      <Callout kind="note">
        <p>
          Prefer the PDF or the official HTML. Blog posts are for
          intuition, not for citations. If a link dies, search the title;
          titles here are the stable thing.
        </p>
      </Callout>

      <h2>Speech recognition, general</h2>
      <ul>
        <li>
          <a href="https://web.stanford.edu/~jurafsky/slp3/">Jurafsky &amp; Martin, SLP3</a> — course-level ASR and seq2seq.
        </li>
        <li>
          Huang, Acero, Hon, <em>Spoken Language Processing</em> (2001) — classical stack.
        </li>
      </ul>

      <h2>Whisper and seq2seq ASR</h2>
      <ul>
        <li>
          <a href="https://arxiv.org/abs/2212.04356">Radford et al., Whisper (arXiv:2212.04356)</a>
        </li>
        <li>
          <a href="https://huggingface.co/docs/transformers/model_doc/whisper">Hugging Face Whisper docs</a>
        </li>
        <li>
          <a href="https://arxiv.org/abs/2605.08214">Bhuiyan et al., Bangla Whisper LoRA (arXiv:2605.08214)</a>
        </li>
      </ul>

      <h2>CTC, wav2vec, XLS-R</h2>
      <ul>
        <li>
          <a href="https://www.cs.toronto.edu/~graves/icml_2006.pdf">Graves et al., CTC (ICML 2006)</a>
        </li>
        <li>
          <a href="https://distill.pub/2017/ctc/">Hannun, Sequence Modeling with CTC (Distill)</a>
        </li>
        <li>
          <a href="https://arxiv.org/abs/2006.11477">Baevski et al., wav2vec 2.0</a>
        </li>
        <li>
          <a href="https://arxiv.org/abs/2111.09296">Babu et al., XLS-R</a>
        </li>
        <li>
          <a href="https://github.com/kensho-technologies/pyctcdecode">pyctcdecode</a>
        </li>
        <li>
          <a href="https://www.openslr.org/53/">OpenSLR-53 Bengali</a>
        </li>
      </ul>

      <h2>Error rates and medical / meaning-aware evaluation</h2>
      <ul>
        <li>
          <a href="https://jitsi.github.io/jiwer/">jiwer</a> — the WER library we call.
        </li>
        <li>
          <a href="https://arxiv.org/abs/2511.16544">“WER is Unaware” (arXiv:2511.16544)</a>
        </li>
        <li>Roy et al., 2021, Semantic WER (medical ASR) — search the title; do not confuse with SWER.</li>
        <li>MedSpeak (2026) — clinical ASR evaluation; pull the PDF when writing.</li>
        <li>HALAS — listenability / ASR evaluation line; pull the PDF when writing.</li>
        <li>“Lost in Transcription” — medical ASR error impact; pull the PDF when writing.</li>
        <li>Answer Error Rate literature — downstream QA, not a transcript metric.</li>
      </ul>

      <h2>Rescoring and MBR</h2>
      <ul>
        <li>Goel &amp; Byrne, Minimum Bayes-risk automatic speech recognition, <em>Computer Speech &amp; Language</em>, 2000.</li>
        <li>Kumar &amp; Byrne, Minimum Bayes-Risk Decoding for ASR / MT (early 2000s).</li>
      </ul>

      <h2>Fine-tuning and GRPO</h2>
      <ul>
        <li>
          <a href="https://arxiv.org/abs/2106.09685">Hu et al., LoRA (arXiv:2106.09685)</a>
        </li>
        <li>
          <a href="http://incompleteideas.net/book/the-book-2nd.html">Sutton &amp; Barto, RL textbook (free)</a>
        </li>
        <li>
          <a href="https://arxiv.org/abs/2402.03300">Shao et al., DeepSeekMath / GRPO</a>
        </li>
        <li>
          <a href="https://arxiv.org/abs/2210.03629">Yao et al., ReAct</a>
        </li>
      </ul>

      <h2>Audio preprocessing</h2>
      <ul>
        <li>
          <a href="https://github.com/snakers4/silero-vad">Silero VAD</a>
        </li>
        <li>
          <a href="https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem">Nyquist–Shannon sampling</a>
        </li>
      </ul>

      <h2>In this repo</h2>
      <ul>
        <li><code>docs/STATUS.md</code> — engineering handoff, judgment calls, next steps.</li>
        <li><code>Readme.md</code> — command reference for the original pipeline.</li>
        <li><code>tests/</code> — the executable specification of “we did not break MBR.”</li>
      </ul>
    </Article>
  );
}
