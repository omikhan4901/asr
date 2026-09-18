import Link from "next/link";
import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function WelcomePage() {
  return (
    <Article
      kicker="Start here"
      title="Welcome to the group textbook"
      lede="This is a local handbook for our CSE499 project on Bangla speech recognition. It is written for teammates who have taken a CS course and have never trained a neural net."
      path="/"
    >
      <p>
        You can read it like a short book. The left sidebar is the table of
        contents. Pages at the bottom have Previous / Next. Blue boxes are
        extra explanation. Amber boxes are easy ways to get confused. Red
        boxes are files or commands that can break the published result if
        you “clean them up.”
      </p>
      <p>
        This is <strong>not a work board yet</strong>. Nobody is assigned a
        ticket here. The goal is that after a weekend of reading, you can sit
        in a meeting and follow the research conversation — and later pick up
        real work without being lost.
      </p>

      <h2>What this project is, in one breath</h2>
      <p>
        We build software that turns <strong>Bangla speech</strong> into
        text. A strong existing model (Whisper) still gets about one word in
        three wrong on our long recordings. We already showed that a clever
        way of choosing among the model’s own guesses — called{" "}
        <strong>MBR rescoring</strong> — improves that a bit{" "}
        <em>without training anything</em>. Our supervisor said that alone
        is probably not novel enough to publish. The next bet is a metric
        that cares more about <strong>clinically dangerous mistakes</strong>{" "}
        than about filler words, and maybe a small model that tries to
        correct those mistakes.
      </p>

      <div className="cards">
        <Link className="card" href="/how-to-read">
          <div className="eyebrow">If you are new</div>
          <h3>How to read this book</h3>
          <p>A weekend path. Crash course first, research second.</p>
        </Link>
        <Link className="card" href="/project-in-ten-minutes">
          <div className="eyebrow">If you are in a hurry</div>
          <h3>The project in ten minutes</h3>
          <p>The numbers, the claim, and what is still fake data.</p>
        </Link>
        <Link className="card" href="/crash/asr">
          <div className="eyebrow">Crash course</div>
          <h3>Start at “What is ASR?”</h3>
          <p>No assumed AI background. Sources at the end of every page.</p>
        </Link>
        <Link className="card" href="/research/angles">
          <div className="eyebrow">Later</div>
          <h3>Potential research angles</h3>
          <p>Directions we could grow into. Not assignments.</p>
        </Link>
      </div>

      <Callout kind="ok">
        <p>
          Run this site from the <code>handbook/</code> folder with{" "}
          <code>npm install</code> then <code>npm run dev</code>. Open{" "}
          <code>http://localhost:3000</code>. You do not need a GPU or
          Python for the textbook.
        </p>
      </Callout>

      <Callout kind="warn">
        <p>
          The Python research repo and this website are neighbors, not the
          same program. Reading a page does not train a model. Changing a
          default in <code>WhisperConfig</code> <em>can</em> break the
          34.85% → 32.29% baseline. That warning will come back often, on
          purpose.
        </p>
      </Callout>
    </Article>
  );
}
