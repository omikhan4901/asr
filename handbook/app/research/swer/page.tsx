import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function SwerPage() {
  return (
    <Article
      kicker="Our research"
      title="SWER, the new metric"
      lede="Severity-Weighted Error Rate uses the same word alignment as WER, then multiplies each edit by a clinical weight. It is a draft. Several numbers are still judgment calls."
      path="/research/swer"
    >
      <h2>The one-sentence definition</h2>
      <p>
        Align hypothesis and reference with the same Levenshtein backbone
        WER uses. Charge each substitution, deletion, and insertion a
        weight from a table. Divide by the sum of <em>reference-token</em>{" "}
        weights (critical words count for more in the denominator too).
      </p>
      <pre>
        <code>SWER = (sum of edit weights) / (sum of reference-token weights)</code>
      </pre>

      <h2>The weight table (draft)</h2>
      <table>
        <thead>
          <tr>
            <th>What happened</th>
            <th>Weight</th>
            <th>Intuition</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Critical substitution (negation flip, drug swap, opposite finding)</td>
            <td>5.0</td>
            <td>You said the wrong medical thing fluently</td>
          </tr>
          <tr>
            <td>Garbled span (unintelligible / collapsed)</td>
            <td>5.0</td>
            <td>The meaning was destroyed, CTC-style or otherwise</td>
          </tr>
          <tr>
            <td>Minor substitution</td>
            <td>1.0</td>
            <td>Same as ordinary WER</td>
          </tr>
          <tr>
            <td>Critical deletion</td>
            <td>4.0</td>
            <td>A needed clinical word vanished</td>
          </tr>
          <tr>
            <td>Non-critical deletion</td>
            <td>1.0</td>
            <td>Ordinary drop</td>
          </tr>
          <tr>
            <td>Hallucinated insertion</td>
            <td>3.0</td>
            <td>Whisper invented content</td>
          </tr>
          <tr>
            <td>Filler or repeated insertion</td>
            <td>1.0</td>
            <td>Cheap</td>
          </tr>
        </tbody>
      </table>

      <Callout kind="warn" title="Unresolved on purpose">
        <p>
          For one critical reference token, a substitution scores 5/5 = 1.0
          and a deletion scores 4/5 = 0.8. That means{" "}
          <strong>dropping a critical term is slightly cheaper than
          getting it wrong</strong>. Those two numbers were a first draft,
          not a supervisor decision. They are documented on{" "}
          <code>SWERConfig</code>. Do not silently equalize them to make
          the code look tidy.
        </p>
      </Callout>

      <h2>How we detect “garbled”</h2>
      <p>
        A list of example broken words will never catch real CTC output.
        The default detector is <code>composite</code>:
      </p>
      <ol>
        <li>token is in the explicit <code>garbled_tokens</code> list, or</li>
        <li>the spelling is illegal Bangla (triple letters, leading matra, broken conjuncts), or</li>
        <li>the token is not in the wordlist <em>and</em> it is more than 2 character edits from every listed word.</li>
      </ol>
      <p>
        Adjacent garbled tokens are one <strong>span</strong> and cost 5.0
        once, not 5.0 each. A 1-token collapse and a 3-token collapse of
        the same reference contribute the same garbled mass.
      </p>
      <p>
        The wordlist is a starter file. Unseen real names will over-fire
        until someone replaces it. A 1–2 edit typo of a listed word is{" "}
        <em>not</em> garbled (that is a near-miss, maybe a critical
        substitution if the pair is in the lexicon).
      </p>

      <h2>What is not SWER</h2>
      <ul>
        <li>Roy 2021 Semantic WER (embedding / meaning similarity).</li>
        <li>Anything that uses a neural scorer as the metric.</li>
        <li>Answer Error Rate (did the downstream QA get it right?).</li>
      </ul>

      <h2>The empty sanity test</h2>
      <p>
        <code>tests/fixtures/swer_sanity_examples.json</code> is a form
        with ten blank rows and severity labels (cosmetic / moderate /
        clinically dangerous). <code>tests/test_swer_sanity.py</code>{" "}
        skips until <code>ready</code> is true. Then it only checks{" "}
        <em>order</em>: dangerous examples should score higher SWER than
        cosmetic ones. That test is how the group will know whether the
        metric matches human judgment. It has not been filled.
      </p>

      <Callout kind="ok">
        <p>
          Code: <code>metrics/swer.py</code>, <code>metrics/garbled.py</code>,{" "}
          <code>metrics/severity_lexicon.py</code>. Alignment lives in{" "}
          <code>common/metrics.py</code> as <code>align_tokens</code>.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Roy 2021, Semantic WER",
            why: "The paper we distinguish ourselves from. Quote it, do not reuse the acronym casually.",
          },
          {
            title: "docs/STATUS.md §2",
            why: "Every interpretation (denominator mass, span cap, empty reference) in one engineering list.",
          },
        ]}
      />
    </Article>
  );
}
