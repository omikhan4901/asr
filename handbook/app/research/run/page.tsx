import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";

export default function RunPage() {
  return (
    <Article
      kicker="Our research"
      title="How to run the pipeline"
      lede="Commands only. Read “What you can trust” before you spend a GPU-hour."
      path="/research/run"
    >
      <h2>Python install</h2>
      <pre>
        <code>{`cd "D:\\Projects\\ASR Paper Code"
pip install -e .
pip install -e ".[audio,vad,ctc,config,distributed,dev]"`}</code>
      </pre>
      <p>
        On Windows, if imports fail:{" "}
        <code>set PYTHONPATH=src</code> in the same terminal.
      </p>

      <h2>Reproduce the baseline (needs the eval JSONL)</h2>
      <pre>
        <code>{`python pipeline/05_rescore_mbr.py --nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl --temperature 0.5
python pipeline/08_evaluate_benchmarks.py --eval_nbest_jsonl data/processed/nbest/eval_nbest_candidates.jsonl`}</code>
      </pre>
      <p>
        Full 2,271-utt eval should sit near 34.85 / 32.29 / 28.72. A
        subset will not match the absolute numbers.
      </p>

      <h2>Regenerate Whisper N-best (expensive)</h2>
      <pre>
        <code>{`python pipeline/03_generate_nbest.py --split_csv data/processed/splits/eval_split.csv --num_beams 12 --temperature 0.7
modal run pipeline/04_modal_nbest.py --split-csv-path eval_split.csv --output-jsonl eval_nbest_candidates.jsonl --num-shards 10`}</code>
      </pre>

      <h2>Wav2Vec2 + comparison</h2>
      <pre>
        <code>{`pip install pyctcdecode
python pipeline/03b_generate_wav2vec2_nbest.py --split_csv data/processed/splits/eval_split.csv --n 12
python pipeline/09_compare_acoustic_models.py --whisper_jsonl data/processed/nbest/eval_nbest_candidates.jsonl --wav2vec2_jsonl data/processed/nbest/eval_wav2vec2_nbest.jsonl`}</code>
      </pre>

      <h2>Tests (no GPU)</h2>
      <pre>
        <code>{`set PYTHONPATH=src
python tests/test_smoke.py
python tests/test_swer.py
python tests/test_garbled.py
python tests/test_reward.py
python tests/test_swer_sanity.py`}</code>
      </pre>
      <p>
        The last one should skip until someone fills the fixture. A skip
        is a reminder, not a broken build.
      </p>

      <Callout kind="danger">
        <p>
          <code>pipeline/10_train_grpo.py</code> and{" "}
          <code>modal run src/bangla_asr/training/modal_train.py</code>{" "}
          exist. Do not run them as a first task.
        </p>
      </Callout>

      <h2>This textbook</h2>
      <pre>
        <code>{`cd handbook
npm install
npm run dev`}</code>
      </pre>
      <p>Then open http://localhost:3000</p>
    </Article>
  );
}
