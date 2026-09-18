import { Article } from "@/components/Article";
import { Callout } from "@/components/Callout";
import { Sources } from "@/components/Sources";

export default function AudioPage() {
  return (
    <Article
      kicker="Crash course"
      title="Audio, chunks, and VAD"
      lede="A two-hour recording is not one model input. We cut it into short speech spans, resample everything to 16 kHz mono, and only then call Whisper or Wav2Vec2."
      path="/crash/audio"
    >
      <h2>What the computer actually sees</h2>
      <p>
        Sound is a list of numbers. <strong>Sample rate</strong> is how
        many numbers per second. Almost every modern speech model wants{" "}
        <strong>16,000 samples per second</strong> (16 kHz) and one
        channel (mono). Our loader in <code>common/audio.py</code> tries
        torchaudio, then soundfile, then librosa, and resamples for you.
      </p>
      <p>
        We also pad a little silence on each side of a clip. That sounds
        pointless until you watch a model clip the first or last syllable.
        The pad is a default of 0.2 seconds in <code>AudioConfig</code>.
      </p>

      <h2>Why we cut the audio (VAD)</h2>
      <p>
        <strong>Voice Activity Detection</strong> asks: “is someone talking
        right now?” We use Silero VAD. It marks speech spans, then a
        packing pass merges short pauses and splits speech that is too
        long. Whisper was trained on segments up to 30 seconds. Long
        continuous speech is cut near a quiet frame, not in the middle of
        a word if we can help it.
      </p>
      <p>
        After chunking we still need the <em>text that belongs to that
        chunk</em>. That is a fuzzy alignment problem against a long
        transcript, implemented in <code>preprocessing/aligner.py</code>.
        Bad alignment produces a chunk whose “ground truth” is the wrong
        sentence. Then every metric looks insane.
      </p>

      <h2>Group-aware splits</h2>
      <p>
        If chunk 1 and chunk 400 come from the same original recording,
        putting one in train and one in test is cheating: the model has
        heard that room, that speaker, that story.{" "}
        <code>dataset/splitter.py</code> keeps a whole source file on one
        side of the split.
      </p>

      <Callout kind="ok">
        <p>
          Pipeline step 1 is <code>pipeline/01_preprocess.py</code>. Step 2
          is <code>pipeline/02_split_dataset.py</code>. You can discuss
          later stages without running these, but you cannot invent a new
          dataset without them.
        </p>
      </Callout>

      <Sources
        items={[
          {
            title: "Silero VAD",
            href: "https://github.com/snakers4/silero-vad",
            why: "The VAD we actually call. Read the README, not a random Medium post.",
          },
          {
            title: "Digital signal processing: sampling theorem (any DSP notes)",
            href: "https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem",
            why: "Why 16 kHz can represent speech up to 8 kHz. Optional physics.",
          },
        ]}
      />
    </Article>
  );
}
