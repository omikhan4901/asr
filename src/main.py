from tqdm import tqdm
from data_loader import load_and_resample_dataset
from inference import load_baseline_model
from evaluation import calculate_wer

def run_baseline(dataset_path):
    dataset = load_and_resample_dataset(dataset_path)
    processor, model, device = load_baseline_model()
    
    references = []
    predictions = []
    
    print("Starting 1-Best Baseline Inference...")
    for item in tqdm(dataset):
        audio_array = item["audio"]["array"]
        ground_truth = item["sentence"] 
        
        # Extract log-Mel spectrogram features
        input_features = processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate the transcript using greedy decoding
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        references.append(ground_truth)
        predictions.append(transcription)
        
    wer = calculate_wer(references, predictions)
    print(f"Final Baseline WER: {wer * 100:.2f}%")

if __name__ == "__main__":
    # Provide the path to your BengaliLoop JSON
    run_baseline("path/to/bengaliloop_test.json")