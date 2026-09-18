import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function CtcPage() {
  return (
    <Article
      kicker="Crash course"
      title="Wav2Vec2 and CTC"
      lede="A CTC model writes a label for every tiny time step, including a blank, then collapses repeats. It does not plan a sentence the way Whisper does. Its failures look different, and that is why we added it."
      path="/crash/ctc"
    >
      <h2>A different contract with time</h2>
      <p>
        Whisper decides the <em>next word</em>. A CTC model decides, for
        each audio frame, “which character (or blank) is this frame?” The
        training loss (Connectionist Temporal Classification) sums over all
        alignments that would collapse to the same text. You never have to
        label each millisecond by hand.
      </p>
      <p>
        At decode time, <strong>greedy CTC</strong> takes the best label
        per frame and collapses. That is a true 1-best, and it is what we
        store as <code>greedy_text</code>. It is not a beam.
      </p>

      <h2>How we get N-best anyway</h2>
      <p>
        <code>pyctcdecode</code> runs a beam over those frame distributions
        and can optionally multiply in a KenLM language model. KenLM is{" "}
        <em>optional</em>. If you turn it on, the N-best list is no longer
        “purely acoustic,” and MBR’s claim changes. We do not ship a Bangla
        KenLM file.
      </p>
      <p>
        Default checkpoint:{" "}
        <code>arijitx/wav2vec2-xls-r-300m-bengali</code> (XLS-R 300M
        fine-tuned on OpenSLR-53). Config can swap to{" "}
        <code>tanmoyio/wav2vec2-large-xlsr-bengali</code> or a local
        fine-tune. That swap must not change Whisper defaults.
      </p>

      <h2>Why the errors sound different</h2>
      <table>
        <thead>
          <tr>
            <th></th>
            <th>Whisper</th>
            <th>Wav2Vec2 / CTC</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Typical wrong output</td>
            <td>A real word that was not said</td>
            <td>A collapsed or illegal-looking token</td>
          </tr>
          <tr>
            <td>Fluency</td>
            <td>High — it is also a language model</td>
            <td>Lower — it is mostly acoustic</td>
          </tr>
          <tr>
            <td>N-best source</td>
            <td>Decoder beam / sampling</td>
            <td>Must use CTC beam search</td>
          </tr>
          <tr>
            <td>Score scale</td>
            <td>Sequence log-prob</td>
            <td>pyctcdecode beam score (different units)</td>
          </tr>
        </tbody>
      </table>

      <Callout kind="danger">
        <p>
          If <code>pyctcdecode</code> is missing, our code{" "}
          <strong>raises</strong>. It will not quietly repeat greedy 12
          times. That is not hostility. That would fake an oracle gap.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Graves et al., Connectionist Temporal Classification (ICML 2006)",
            href: "https://www.cs.toronto.edu/~graves/icml_2006.pdf",
            why: "The CTC paper. The picture of “many alignments, one string” is the whole idea.",
          },
          {
            title: "Baevski et al., wav2vec 2.0 (arXiv:2006.11477)",
            href: "https://arxiv.org/abs/2006.11477",
            why: "Self-supervised speech encoder we inherit from.",
          },
          {
            title: "Babu et al., XLS-R (arXiv:2111.09296)",
            href: "https://arxiv.org/abs/2111.09296",
            why: "The multilingual initialization behind the Bangla CTC checkpoints.",
          },
          {
            title: "Hannun, Sequence Modeling with CTC (Distill 2017)",
            href: "https://distill.pub/2017/ctc/",
            why: "Best visual explanation. Read this if papers feel dry.",
          },
          {
            title: "pyctcdecode",
            href: "https://github.com/kensho-technologies/pyctcdecode",
            why: "The beam-search library we require for CTC N-best.",
          },
        ]}
      />
    </Article>
  );
}
