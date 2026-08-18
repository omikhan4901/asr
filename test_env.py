from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

print("Loading model...")
model_id = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

print("Streaming a tiny sample dataset...")
# This streams 1 sample of Bengali Common Voice audio
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "bn", split="test", streaming=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Get the first item
sample = next(iter(dataset))

print("Transcribing...")
input_features = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"\nGround Truth: {sample['sentence']}")
print(f"Whisper Said: {transcription}")