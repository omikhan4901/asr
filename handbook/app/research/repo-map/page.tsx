import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function RepoMapPage() {
  return (
    <Article
      kicker="Our research"
      title="Repo map"
      lede="A guided tour of folders. Green means you may read freely. Red means ask before editing defaults."
      path="/research/repo-map"
    >
      <h2>The Python package</h2>
      <table>
        <thead>
          <tr>
            <th>Path</th>
            <th>Job</th>
            <th>Touch?</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>src/bangla_asr/config.py</code></td>
            <td>Every default. Whisper/MBR defaults are the baseline lock.</td>
            <td>Read always. Edit Whisper/MBR only for a named experiment.</td>
          </tr>
          <tr>
            <td><code>common/</code></td>
            <td>Audio load, Bangla normalize, WER/CER, alignment backtrace.</td>
            <td>Safe to read. Changing normalize changes every number.</td>
          </tr>
          <tr>
            <td><code>preprocessing/</code></td>
            <td>VAD + fuzzy align.</td>
            <td>Needed if you add data, not if you only read papers.</td>
          </tr>
          <tr>
            <td><code>models/autoregressive/</code></td>
            <td>Whisper load + N-best generate + Modal worker.</td>
            <td>Do not rewrite generate kwargs on the default path.</td>
          </tr>
          <tr>
            <td><code>models/ctc/</code></td>
            <td>Wav2Vec2 + pyctcdecode.</td>
            <td>New track. Fair game for experiments.</td>
          </tr>
          <tr>
            <td><code>models/bert_reranker/</code></td>
            <td>Trained candidate scorer.</td>
            <td>Existing thesis spine. Keep it.</td>
          </tr>
          <tr>
            <td><code>rescoring/</code></td>
            <td>MBR, oracle, old benchmark table.</td>
            <td>MBR math is locked by tests.</td>
          </tr>
          <tr>
            <td><code>metrics/</code></td>
            <td>SWER, garbled detectors, placeholder lexicon, starter wordlist.</td>
            <td>This is the research surface.</td>
          </tr>
          <tr>
            <td><code>training/</code>, <code>correction/</code></td>
            <td>Reward, GRPO, ReAct stub.</td>
            <td>Read. Do not run training yet.</td>
          </tr>
          <tr>
            <td><code>eval/</code></td>
            <td>Side-by-side Whisper vs CTC.</td>
            <td>Use when both JSONLs exist.</td>
          </tr>
          <tr>
            <td><code>dataset/loaders.py</code></td>
            <td>Bangla CSV loader. English names raise NotImplementedError.</td>
            <td>Do not silently implement MedDialog.</td>
          </tr>
        </tbody>
      </table>

      <h2>The numbered scripts</h2>
      <p>
        <code>pipeline/01</code> … <code>08</code> are the original story.
        <code>03b</code> / <code>04b</code> / <code>09</code> / <code>10</code>{" "}
        are the new dual-model / GRPO story. Prefer these scripts to
        one-off notebooks so everyone runs the same flags.
      </p>

      <h2>This handbook</h2>
      <p>
        Lives in <code>handbook/</code>. It is a Next.js app. It does not
        import the Python package. Changing a textbook page cannot break
        WER. Changing <code>config.py</code> can.
      </p>

      <Callout kind="danger">
        <p>
          Files that protect the 34.85 → 32.29 claim:{" "}
          <code>whisper_model.py</code>, <code>generator.py</code> defaults,{" "}
          <code>mbr.py</code>, <code>oracle.py</code> 1-best definition,{" "}
          <code>WhisperConfig</code>, <code>MBRConfig</code>,{" "}
          <code>tests/test_smoke.py</code>.
        </p>
      </Callout>
    </Article>
  );
}
