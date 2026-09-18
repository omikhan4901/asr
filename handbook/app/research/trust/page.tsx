import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function TrustPage() {
  return (
    <Article
      kicker="Our research"
      title="What you can trust today"
      lede="A honesty page. If a slide disagrees with this table, the slide is wrong."
      path="/research/trust"
    >
      <table>
        <thead>
          <tr>
            <th>Thing</th>
            <th>Status</th>
            <th>Meaning</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Whisper N-best + MBR + oracle WER</td>
            <td>Trustworthy</td>
            <td>Tested; defaults locked; the 34.85 story.</td>
          </tr>
          <tr>
            <td>Bangla normalize + group split + VAD packing</td>
            <td>Trustworthy</td>
            <td>Covered by smoke tests.</td>
          </tr>
          <tr>
            <td>ASRModel interface + Whisper adapter</td>
            <td>Trustworthy</td>
            <td>Unit-tested without downloading weights.</td>
          </tr>
          <tr>
            <td>Wav2Vec2 N-best code</td>
            <td>Usable, unvalidated on the big set</td>
            <td>Fail-loud without pyctcdecode. No 2,271-utt table yet.</td>
          </tr>
          <tr>
            <td>SWER implementation</td>
            <td>Usable on synthetics</td>
            <td>Not signed off on real transcripts.</td>
          </tr>
          <tr>
            <td>Clinical lexicon</td>
            <td>Placeholder</td>
            <td>“না” is marked critical. It will over-fire.</td>
          </tr>
          <tr>
            <td>Bangla wordlist</td>
            <td>Starting point</td>
            <td>Too small for a paper claim.</td>
          </tr>
          <tr>
            <td>SWER sanity fixture</td>
            <td>Empty</td>
            <td>Test skips. Fill before trusting compare_models SWER.</td>
          </tr>
          <tr>
            <td>GRPO trainer</td>
            <td>Coded, not run</td>
            <td>Do not launch.</td>
          </tr>
          <tr>
            <td>ReAct scaffold</td>
            <td>Stub</td>
            <td>Parses text. No tools that matter.</td>
          </tr>
          <tr>
            <td>English clinical loaders</td>
            <td>Hooks only</td>
            <td>They raise. That is correct.</td>
          </tr>
          <tr>
            <td>VibeThinker-3B id</td>
            <td>Unverified</td>
            <td>Check Hugging Face yourself.</td>
          </tr>
        </tbody>
      </table>

      <Callout kind="ok">
        <p>
          The longer version of this table, including every judgment call,
          is <code>docs/STATUS.md</code>. This page is the version you can
          read aloud in five minutes.
        </p>
      </Callout>
    </Article>
  );
}
