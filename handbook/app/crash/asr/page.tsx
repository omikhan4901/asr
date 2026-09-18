import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function AsrPage() {
  return (
    <Article
      kicker="Crash course"
      title="What is ASR?"
      lede="Automatic Speech Recognition is a program that listens to audio and types what was said. That is the whole job. Everything else in this repo is a way to make that typing less wrong, or to measure the remaining wrongness more honestly."
      path="/crash/asr"
    >
      <h2>A picture you already know</h2>
      <p>
        You have used this: YouTube captions, phone dictation, meeting notes.
        A microphone produces a waveform (air pressure over time). A model
        turns that waveform into tokens (pieces of words), then into a
        string.
      </p>
      <p>
        ASR is not “understand the meaning.” It is not translation. It is
        not speaker identification (“who talked?”). Those are neighboring
        problems. We only ask: <em>what words were spoken?</em>
      </p>

      <h2>Why Bangla is not “English ASR with a different alphabet”</h2>
      <ul>
        <li>Less public labeled audio than English.</li>
        <li>Many speakers, dialects, and code-mixing with English.</li>
        <li>Writing system details (nukta characters, different Unicode spellings of the same word) that make two matching strings look unequal if you are careless.</li>
        <li>Our recordings are often <strong>long-form</strong> — minutes, not 5-second clips. Models have a comfort window (Whisper likes chunks under ~30 seconds).</li>
      </ul>
      <p>
        Our code therefore spends real effort on <strong>chunking</strong>{" "}
        and on <strong>Bangla text normalization</strong> before anyone
        computes WER. If you skip those, you will “improve” a number by
        accident.
      </p>

      <h2>The clinical twist</h2>
      <p>
        If someone says “do <em>not</em> give insulin” and the model writes
        “do give insulin,” WER may count that as a one-word error. A doctor
        would not. Our research direction is: measure and maybe correct
        mistakes the way a clinical reader would, not the way a spelling
        quiz would.
      </p>

      <Callout kind="note">
        <p>
          You will see “utterance” everywhere. It just means one scored
          clip: a chunk of audio plus the text we believe was said
          (the <em>reference</em> or ground truth) plus the model’s text
          (the <em>hypothesis</em>).
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Jurafsky & Martin, Speech and Language Processing, ch. on ASR",
            href: "https://web.stanford.edu/~jurafsky/slp3/",
            why: "The standard CS textbook chapter. Free draft. Start here if you want the academic definition.",
          },
          {
            title: "Huang, Acero, Hon — Spoken Language Processing (2001)",
            why: "Older, but still the clearest book-length treatment of classical ASR. Library, not a blog post.",
          },
        ]}
      />
    </Article>
  );
}
