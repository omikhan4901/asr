import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function NbestPage() {
  return (
    <Article
      kicker="Crash course"
      title="Beam search and N-best lists"
      lede="A model can return its single favorite sentence, or its top N favorite sentences with scores. Rescoring only exists if we kept the list."
      path="/crash/nbest"
    >
      <h2>Why one answer is a bad habit</h2>
      <p>
        At each writing step Whisper has a distribution over the next
        token. Greedy decoding always takes the argmax. That is fast and
        often decent. It also throws away every runner-up. Sometimes the
        runner-up path is the one that matches the audio.
      </p>

      <h2>Beam search, without the mystique</h2>
      <p>
        Keep the <strong>B</strong> most promising partial sentences
        (the beam). At each step, expand them, score the expansions, and
        keep the top B again. At the end you can return the best one, or
        the whole beam. We return <strong>N = 12</strong> sequences.
      </p>
      <p>
        Our published setup also uses <em>sampling-based diversity</em>{" "}
        (temperature 0.7), not grouped diverse beam search. Grouped DBS is
        implemented but off, so the paper-shaped number stays reproducible.
      </p>
      <p>
        Each candidate is stored as{" "}
        <code>{`{"text", "logprob", "rank"}`}</code>. Higher log-probability
        is better (less negative). Rank 0 is the 1-best.
      </p>

      <h2>JSONL is the handshake</h2>
      <p>
        One line per utterance: id, audio path, ground truth, list of
        candidates. MBR, oracle, BERT reranker, and SWER all read that
        file. If you change the schema, you break every later stage. New
        fields we added later (<code>model_family</code>,{" "}
        <code>greedy_text</code>) are extra; old Whisper files still work.
      </p>

      <Callout kind="warn">
        <p>
          CTC models do <em>not</em> get N-best “for free” from Whisper-style
          generate(). If someone returns the same greedy string 12 times,
          MBR and oracle become theater. Our Wav2Vec2 path must use
          <code> pyctcdecode</code> or refuse. Next page.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Lowerre, The Harpy Speech Recognition System (1976) — beam search origin",
            why: "Historical. Any modern NLP lecture on beam search is enough (Jurafsky §, or Stanford CS224N notes).",
          },
          {
            title: "Hannun, “Speech Recognition and Beam Search” (blog / distill-style notes)",
            href: "https://distill.pub/2017/ctc/",
            why: "The Distill CTC article also explains beams in the CTC setting. Useful before the next page.",
          },
        ]}
      />
    </Article>
  );
}
