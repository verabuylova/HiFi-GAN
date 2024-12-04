from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd
from src.model.hifigan import HiFiGanGenerator
import soundfile as sf
import os

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

# from torch.profiler import profile, record_function, ProfilerActivity


def extract_generator_state_dict(full_checkpoint_path, output_path=None):
    """
    Extracts the generator's state_dict from the full checkpoint.

    Args:
        full_checkpoint_path (str): Path to the full checkpoint file.
        output_path (str, optional): Path to save the extracted generator's state_dict.
                                      If None, the state_dict is not saved to disk.

    Returns:
        dict: The generator's state_dict.
    """
    # Load the full checkpoint
    checkpoint = torch.load(full_checkpoint_path, map_location='cpu')

    # Ensure 'state_dict' is present
    if 'state_dict' not in checkpoint:
        raise KeyError("The checkpoint does not contain a 'state_dict' key.")

    # Extract the generator's state_dict directly
    generator_state_dict = checkpoint['state_dict']

    # Optionally, remove any prefix from the keys (e.g., 'model.')
    # Uncomment and modify the following line if your keys have prefixes
    # generator_state_dict = {k.replace('model.', ''): v for k, v in generator_state_dict.items()}

    # Optionally save the extracted state_dict to disk
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(generator_state_dict, output_path)
        print(f"Generator's state_dict saved to {output_path}")

    return generator_state_dict


def load_hifigan_model(state_dict_path, device='cpu'):
    """
    Loads the HiFiGAN generator model with the provided state_dict.

    Args:
        state_dict_path (str): Path to the generator's state_dict.
        device (str | torch.device): Device to load the model onto.

    Returns:
        HiFiGanGenerator: The loaded HiFiGAN generator model.
    """
    # Initialize the HiFiGAN generator model
    hifigan = HiFiGanGenerator()
    
    # Load the generator's state_dict
    generator_state_dict = torch.load(state_dict_path, map_location=device)
    
    # Load the state_dict into the model
    # Using strict=False to ignore missing/unexpected keys if necessary
    load_result = hifigan.load_state_dict(generator_state_dict, strict=False)
    
    # Print missing and unexpected keys for debugging
    if load_result.missing_keys:
        print("Missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys:", load_result.unexpected_keys)
    
    # Move the model to the specified device
    hifigan.to(device)
    hifigan.eval()
    
    return hifigan


def synthesize_text(text, fastspeech_model, fastspeech_generator, task, hifigan, device='cpu'):
    """
    Synthesizes audio from text using FastSpeech2 and HiFiGAN models.

    Args:
        text (str): The input text to synthesize.
        fastspeech_model (nn.Module): The FastSpeech2 model.
        fastspeech_generator (nn.Module): The generator for FastSpeech2.
        task (Task): The Fairseq task.
        hifigan (HiFiGanGenerator): The HiFiGAN generator model.
        device (str | torch.device): Device to perform inference on.

    Returns:
        np.ndarray: The synthesized audio waveform.
    """
    # Prepare the input
    sample = TTSHubInterface.get_model_input(task, text)
    sample = TTSHubInterface.move_to_sample(sample, device)
    
    # Generate mel-spectrogram using FastSpeech2
    with torch.no_grad():
        wav, mel = TTSHubInterface.get_prediction(task, fastspeech_model, fastspeech_generator, sample)
    
    # Ensure mel is on the correct device
    mel = mel.to(device)
    
    # Generate audio waveform using HiFiGAN
    with torch.no_grad():
        audio_dict = hifigan(mel.unsqueeze(0))  # Add batch dimension if necessary
        audio = audio_dict["output_audio"].squeeze().cpu().numpy()
    
    return audio


def main():
    # Paths
    full_checkpoint_path = 'saved/testing/model_best.pth'
    extracted_state_dict_path = 'saved/testing/hifigan_generator_state_dict.pth'

    # Extract and save the generator's state_dict
    try:
        generator_state_dict = extract_generator_state_dict(full_checkpoint_path, extracted_state_dict_path)
    except KeyError as e:
        print(f"Error during state_dict extraction: {e}")
        return

    # Load FastSpeech2 model and task from Hugging Face Hub
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )

    # Update the configuration with data settings
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

    # Build the generator for FastSpeech2
    fastspeech_generator = task.build_generator(models, cfg)
    fastspeech_model = models[0]
    fastspeech_model.eval()

    # Load HiFiGAN generator model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        hifigan = load_hifigan_model(extracted_state_dict_path, device)
    except KeyError as e:
        print(f"Error during HiFiGAN model loading: {e}")
        return

    # Example inference
    text = "Hello, this is a test run."
    try:
        synthesized_audio = synthesize_text(text, fastspeech_model, fastspeech_generator, task, hifigan, device)
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return

    # Play the synthesized audio in Jupyter Notebook
    ipd.display(ipd.Audio(synthesized_audio, rate=22050))

    # Save the synthesized audio to a WAV file
    sf.write('synthesized_output.wav', synthesized_audio, 22050)
    print("Audio saved to synthesized_output.wav")


if __name__ == "__main__":
    main()
