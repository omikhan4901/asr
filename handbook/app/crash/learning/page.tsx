import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function LearningPage() {
  return (
    <Article
      kicker="Crash course"
      title="Fine-tuning and GRPO"
      lede="Inference-only methods pick among existing guesses. Fine-tuning changes the model. Reinforcement learning changes it using a score we write, not using “the next token was token 17.”"
      path="/crash/learning"
    >
      <h2>Three knobs people confuse</h2>
      <table>
        <thead>
          <tr>
            <th>Knob</th>
            <th>What moves</th>
            <th>In this project</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Inference / rescoring</td>
            <td>Nothing inside the acoustic model</td>
            <td>MBR, BERT rerank, oracle</td>
          </tr>
          <tr>
            <td>Supervised fine-tune</td>
            <td>Weights, using audio + reference text</td>
            <td>Possible later (LoRA on Whisper); not the current headline</td>
          </tr>
          <tr>
            <td>RL fine-tune</td>
            <td>Weights, using a reward we define</td>
            <td>GRPO on a small correction model, coded, not run</td>
          </tr>
        </tbody>
      </table>

      <h2>Fine-tuning, gently</h2>
      <p>
        The model already knows a lot. Fine-tuning shows it more examples
        from <em>our</em> domain so it shifts a little. LoRA is a popular
        cheap version: freeze almost everything, train small adapter
        matrices. Bhuiyan et al. did that for Bangla Whisper and reported
        ~31.32% WER. That number is why 32.29% MBR is interesting: we
        approached a trained result without training.
      </p>

      <h2>Why RL at all?</h2>
      <p>
        Supervised training says “copy the reference.” A correction model
        that always copies the ASR output when the ASR is close would get
        a good token-level loss and still be useless. We want:{" "}
        <em>change the words that are clinically expensive</em>. That is a
        score over whole strings, not over the next token. Reinforcement
        learning is the family of methods that optimize a score.
      </p>

      <h2>GRPO in one paragraph</h2>
      <p>
        For one prompt (“here is the ASR text, please fix it”), we sample
        a <strong>group</strong> of G corrections from the current model.
        We score each with the same reward function. We turn those scores
        into <em>relative</em> advantages inside the group (better than
        the group mean, or worse). We push the model toward the better
        ones. There is no separate critic network. That is why GRPO fits
        a 3B model on one GPU better than older actor-critic recipes.
      </p>
      <p>
        Our reward is not “how pretty is the sentence?” It is:
      </p>
      <pre>
        <code>SWER(original ASR, ground truth) − SWER(correction, ground truth)</code>
      </pre>
      <p>
        If you make the transcript less severe, the reward is positive.
        If you make it worse, the reward is negative. Easy utterances that
        were already correct do not get free points for staying pretty.
      </p>

      <Callout kind="danger">
        <p>
          Do not launch Modal GRPO because the script exists. The reward
          is only as good as SWER. SWER is only as good as the lexicon and
          your hand labels. Those are not done. Training now would spend
          money to optimize a placeholder.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Howard & Ruder, ULMFiT (2018) / any LoRA explainer",
            href: "https://arxiv.org/abs/2106.09685",
            why: "Hu et al. LoRA (arXiv:2106.09685) is the adapter paper. ULMFiT is the older “fine-tune a language model” story.",
          },
          {
            title: "Sutton & Barto, Reinforcement Learning: An Introduction",
            href: "http://incompleteideas.net/book/the-book-2nd.html",
            why: "Free textbook. You need chapters 1–3, not the whole book.",
          },
          {
            title: "Shao et al., DeepSeekMath — GRPO (arXiv:2402.03300)",
            href: "https://arxiv.org/abs/2402.03300",
            why: "The GRPO paper. Read the group-relative advantage paragraph.",
          },
        ]}
      />
    </Article>
  );
}
