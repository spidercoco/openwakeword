import sys
import os
import numpy as np
import onnxruntime as ort
import openwakeword
import openwakeword.utils
import librosa

def check_wakeword(wav_path, model_path="beary.onnx"):
    # Load audio and resample to 16000 Hz
    print(f"Loading audio from {wav_path}...")
    audio, sr = librosa.load(wav_path, sr=16000)
    
    # Standardize length if too short (pad to at least 3 seconds)
    # The model expects the wake word to be at the END of the 3-second window.
    # So we pad at the BEGINNING.
    min_length = 3 * 16000
    if len(audio) < min_length:
        audio = np.pad(audio, (min_length - len(audio), 0))
        print("Audio too short, padded at the beginning to 3 seconds.")
    else:
        # To handle cases where the wake word is at the very beginning of a long file,
        # we can prepad with some silence so the sliding window can "see" it at the end.
        audio = np.pad(audio, (min_length, 0))
        print("Pre-padded with 3 seconds of silence for better detection.")

    # Convert to int16 (openwakeword expects this)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Add batch dimension for embedding extraction (1, N)
    audio_batch = audio_int16.reshape(1, -1)
    
    # Extract features using openwakeword's AudioFeatures
    print("Extracting features...")
    F = openwakeword.utils.AudioFeatures()
    features = F.embed_clips(audio_batch) # Returns shape (1, N_embeddings, 96)
    
    # Load ONNX model
    print(f"Loading model {model_path}...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Get model input window size (expected number of embeddings)
    # For beary.onnx, it's 28
    model_window_size = session.get_inputs()[0].shape[1]
    
    # Sliding window over the features to get the scores
    n_embeddings = features.shape[1]
    scores = []
    
    print(f"Running inference (window size: {model_window_size}, total embeddings: {n_embeddings})...")
    
    if n_embeddings < model_window_size:
        # Pad features if they are too few (though we padded audio, 
        # let's be safe for edge cases)
        pad_width = ((0, 0), (0, model_window_size - n_embeddings), (0, 0))
        features = np.pad(features, pad_width, mode='constant')
        n_embeddings = model_window_size

    # Slide window
    for i in range(0, n_embeddings - model_window_size + 1):
        window = features[:, i : i + model_window_size, :]
        outputs = session.run(None, {input_name: window})
        score = outputs[0][0][0]
        scores.append(score)
    
    max_score = max(scores)
    print(f"Finished checking. Found {len(scores)} windows.")
    return max_score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_wakeword.py <path_to_wav> [path_to_onnx]")
    else:
        wav_file = sys.argv[1]
        model_file = sys.argv[2] if len(sys.argv) > 2 else "beary.onnx"
        
        if not os.path.exists(wav_file):
            print(f"Error: File {wav_file} not found.")
        else:
            try:
                score = check_wakeword(wav_file, model_path=model_file)
                print(f"SCORE:{score:.4f}")
                if score > 0.5:
                    print("Status: WAKE WORD DETECTED!")
                else:
                    print("Status: No wake word detected.")
            except Exception as e:
                print(f"An error occurred: {e}")
