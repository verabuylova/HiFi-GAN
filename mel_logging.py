import wandb
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Define the path to the wav file
wav_file_path = "/Users/vera/Desktop/hifi/HiFi-GAN/data/datasets/ljspeech/train/LJ001-0001.wav"

# Load the audio file using librosa
y, sr = librosa.load(wav_file_path, sr=None)  # sr=None keeps the original sampling rate

# Compute the Mel spectrogram (using keyword arguments)
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert the Mel spectrogram to decibels (log scale)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')

# Save the plot as an image
mel_spectrogram_image_path = "/Users/vera/Downloads/pupin_mel_spec.png"
plt.savefig(mel_spectrogram_image_path)
plt.close()

# Initialize a W&B run
report = wandb.init(project="hifigan", entity="verabuylova-nes")

# Log the audio file and Mel spectrogram image to W&B
wandb.log({
    'tts_audio_true_1': wandb.Audio(wav_file_path),
    'tts_mel_spec_true_1': wandb.Image(mel_spectrogram_image_path)
})

# Log the audio with a caption
wandb.log({"tts_audio_true_1": wandb.Audio(wav_file_path, caption="True Audio")})