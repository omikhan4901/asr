import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function HowToReadPage() {
  return (
    <Article
      kicker="Start here"
      title="How to read this book"
      lede="If the sidebar looks like a lot, ignore it and follow this order. Each page ends with optional reading. Skip those links until a word bothers you."
      path="/how-to-read"
    >
      <h2>A weekend path</h2>
      <ol>
        <li>
          <strong>Saturday morning — the idea of speech-to-text.</strong>{" "}
          Read “What is ASR?”, “Audio, chunks, and VAD”, and “WER and CER”.
          When you finish, you should be able to explain why a 34.85% word
          error rate is both “usable” and “not good enough for medicine.”
        </li>
        <li>
          <strong>Saturday afternoon — two kinds of model.</strong> Read
          Whisper, N-best, and CTC. The only exam question is: “Why can
          Whisper invent a fluent wrong sentence, while Wav2Vec2 more often
          produces broken-looking words?”
        </li>
        <li>
          <strong>Sunday morning — how we pick among guesses.</strong> Read
          MBR, then “Fine-tuning and GRPO”. You do not need to implement
          either. You need the story: consensus among guesses vs changing
          the model with a score.
        </li>
        <li>
          <strong>Sunday afternoon — our paper-shaped project.</strong> Read
          the research section in order through “How the code is wired”.
          Stop before “How to run” unless you have the data on your machine.
        </li>
      </ol>

      <h2>How we use words</h2>
      <p>
        Everyday English first. Then the official name in parentheses, once.
        After that we use the official name, because that is what the code
        and the papers use. If you forget, open the{" "}
        <a href="/reference/glossary">glossary</a>.
      </p>
      <p>
        We will sometimes write a tiny formula. If a formula appears, the
        sentence before it already said the same thing in words. You can
        skip the symbols and still follow.
      </p>

      <h2>What “a source” means here</h2>
      <p>
        At the bottom of almost every page is a short list: paper or
        textbook chapter, a link if one exists, and one line on{" "}
        <em>why</em> we listed it. These are not homework. They are the
        doors you open if you want to write a related-work paragraph or
        argue with a design choice.
      </p>

      <Callout kind="note">
        <p>
          If you have taken algorithms, you already know edit distance.
          WER is edit distance on words. If you have taken probability,
          MBR is “pick the hypothesis with lowest expected loss.” If you
          have taken neither recently, the crash course still works.
        </p>
      </Callout>

      <h2>What this book will not do</h2>
      <ul>
        <li>It will not teach PyTorch from zero.</li>
        <li>It will not assign you a GitHub issue (not yet).</li>
        <li>It will not claim SWER is a finished scientific contribution. The metric is coded. The clinical table is still a placeholder.</li>
      </ul>
    </Article>
  );
}
