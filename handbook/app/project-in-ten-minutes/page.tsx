import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function TenMinutesPage() {
  return (
    <Article
      kicker="Start here"
      title="The project in ten minutes"
      lede="Read this before any meeting. It is the whole story at poster size."
      path="/project-in-ten-minutes"
    >
      <h2>The setting</h2>
      <p>
        Course: CSE499, North South University. Domain: Bangla speech, with
        a later interest in <strong>clinical</strong> transcripts (doctor
        talk, drug names, negation). The code lives in the{" "}
        <code>bangla_asr</code> Python package. English medical datasets
        are a possible side track, not the default language of the project.
      </p>

      <h2>What already works</h2>
      <p>
        On a held-out set of <strong>2,271</strong> utterances we have:
      </p>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Word error rate</th>
            <th>Plain meaning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Whisper 1-best</td>
            <td>34.85%</td>
            <td>The model’s official first guess</td>
          </tr>
          <tr>
            <td>MBR rescoring</td>
            <td>32.29%</td>
            <td>Pick the guess that agrees most with the other guesses</td>
          </tr>
          <tr>
            <td>Oracle</td>
            <td>28.72%</td>
            <td>If we cheated and picked the best guess using the true text</td>
          </tr>
        </tbody>
      </table>
      <p>
        MBR recovered about <strong>42%</strong> of the gap between “first
        guess” and “best guess in the list,” and it did that{" "}
        <strong>without updating a single model weight</strong>. A published
        LoRA fine-tune on a similar Bangla setting sits around 31.32% WER,
        so we got close to a trained baseline by only being clever at
        inference. That is the first paper-shaped result.
      </p>

      <Callout kind="warn">
        <p>
          Our supervisor’s feedback: inference-only rescoring is useful
          engineering, but probably <strong>not novel enough to publish
          alone</strong>. The next phase has to add a new scientific
          object — a metric, a training signal, or a new analysis — not
          just a nicer table of the same WER.
        </p>
      </Callout>

      <h2>What we are building next</h2>
      <ol>
        <li>
          <strong>SWER</strong> — Severity-Weighted Error Rate. Same
          alignment as WER, but a wrong drug name or a dropped “not” costs
          more than a filler word. This is{" "}
          <em>not</em> Roy 2021’s Semantic WER. We chose the name carefully.
        </li>
        <li>
          <strong>A second acoustic model</strong> — Wav2Vec2 (CTC), so we
          can ask whether a non-autoregressive model makes{" "}
          <em>different kinds</em> of mistakes under the same rescoring
          and SWER pipeline.
        </li>
        <li>
          <strong>Optional correction</strong> — a small reasoning model
          (candidate: VibeThinker-3B) that edits the ASR text. If we train
          it, the reward is “did SWER get better than the original ASR?”
          We have <strong>not</strong> run that training. Do not start it
          until SWER is sanity-checked on real labels.
        </li>
      </ol>

      <h2>What is still pretend data</h2>
      <ul>
        <li>The clinical lexicon is a handful of example words marked PLACEHOLDER.</li>
        <li>The Bangla wordlist used to detect “garbled” tokens is a starter list, not a dictionary.</li>
        <li>The hand-labeled SWER sanity file is empty on purpose.</li>
        <li>The English medical loaders raise <code>NotImplementedError</code>.</li>
        <li>The ReAct correction loop is a text stub. It cannot look at audio.</li>
      </ul>

      <div className="flow">
        <span>Audio</span>
        <i>→</i>
        <span>VAD chunks</span>
        <i>→</i>
        <span>Whisper or Wav2Vec2 N-best</span>
        <i>→</i>
        <span>MBR / oracle / SWER</span>
        <i>→</i>
        <span>optional correction</span>
      </div>

      <Sources
        items={[
          {
            title: "Bhuiyan et al., Bangla Whisper LoRA (arXiv:2605.08214)",
            href: "https://arxiv.org/abs/2605.08214",
            why: "The trained Bangla number we compare inference-only MBR against.",
          },
          {
            title: "docs/STATUS.md (in this repo)",
            why: "Engineering handoff: judgment calls, stubs, and the ordered next steps. Open the file on disk; it is not hosted as a web page.",
          },
        ]}
      />
    </Article>
  );
}
