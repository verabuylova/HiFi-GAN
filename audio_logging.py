import wandb

wav_file_path = "/Users/vera/Desktop/hifi/HiFi-GAN/data/saved/texts_df/test/output_0.wav"
report = wandb.init(project="hifigan", entity="verabuylova-nes")
wandb.log({
    'tts_audio_pred_k': wandb.Audio(wav_file_path)
})
wandb.log({"tts_audio_pred_k": wandb.Audio(wav_file_path, caption="True Audio")})


