import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function QuestionPage() {
  return (
    <Article
      kicker="Our research"
      title="The scientific question"
      lede="A systems demo is not a paper. This page is the difference, in the language a supervisor uses."
      path="/research/question"
    >
      <h2>What we can already say</h2>
      <p>
        “On this Bangla long-form set, keeping a diverse 12-best from a
        regional Whisper and picking by probability-weighted MBR recovers
        about two-fifths of the oracle WER headroom with no training.”
        That is a characterization result. It is true, measured, and
        reproducible if we do not touch the defaults.
      </p>
      <p>
        The community already knows MBR. The community already knows
        Whisper N-best. So the novelty cannot be “we ran MBR.” It has to
        be one of:
      </p>
      <ul>
        <li>a <strong>new measurement</strong> that WER hides (SWER),</li>
        <li>a <strong>new empirical finding</strong> (CTC and Whisper fail differently under that measurement),</li>
        <li>a <strong>new training signal</strong> (correct with SWER-improvement reward),</li>
        <li>or a careful combination that a reviewer has not seen for Bangla clinical speech.</li>
      </ul>

      <h2>What we refuse to claim yet</h2>
      <ul>
        <li>That SWER is a validated clinical metric. The table is a placeholder.</li>
        <li>That Wav2Vec2 is better or worse on our set. We have not generated the eval JSONL here.</li>
        <li>That GRPO correction works. The loop is written. It has not been run.</li>
        <li>That we have a medical corpus. The primary data is still the Bangla long-form pipeline you already had.</li>
      </ul>

      <Callout kind="note">
        <p>
          Related names you should be able to pronounce in a meeting:
          Roy 2021 Semantic WER, “WER is Unaware,” MedSpeak, HALAS,
          “Lost in Transcription,” Answer Error Rate. We are in that
          conversation. We are not a copy of any of them. Details live
          on the SWER page and in the sources library.
        </p>
      </Callout>

      <h2>The working thesis (draft)</h2>
      <p>
        Word error rate cannot tell a clinically dangerous substitution
        from a harmless one. If we define an explicit severity-weighted
        edit rate on the same alignment as WER, we can (a) compare
        Whisper and CTC error <em>types</em> fairly, and (b) train a
        small corrector to reduce the expensive errors rather than the
        frequent ones. Whether that thesis survives contact with a real
        lexicon and real labels is the rest of the semester.
      </p>

      <Sources
        items={[
          {
            title: "“WER is Unaware” (arXiv:2511.16544)",
            href: "https://arxiv.org/abs/2511.16544",
            why: "Closest recent slogan to our motivation. Read it so we do not reinvent their wording.",
          },
          {
            title: "MedSpeak (2026) and HALAS",
            why: "Clinical / medical ASR evaluation efforts. Find the PDFs when you write related work; do not cite from memory.",
          },
          {
            title: "“Lost in Transcription” and Answer Error Rate papers",
            why: "Downstream-task error rates. Useful contrast: we are still a transcript metric, not a QA metric.",
          },
        ]}
      />
    </Article>
  );
}
