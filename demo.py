import io
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import mmap
import numpy
import soundfile
import torchaudio
import torch
import os

from collections import defaultdict
# from IPython.display import Audio, display
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play


from seamless_communication.inference import Translator
from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover

# Speech to Speech Translation
def s2st_inference(in_file="", play_input=False, play_output=False):
    # README:  https://github.com/facebookresearch/seamless_communication/tree/main/src/seamless_communication/cli/m4t/predict
    # Please use audios with duration under 20 seconds for optimal performance.

    # Resample the audio in 16khz if sample rate is not 16khz already.
    # torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)

    if in_file=="":
        in_file = "content/LJ037-0171_sr16k.wav"

    if (play_input):
        if (os.path.exists(in_file)):
            play(AudioSegment.from_wav(in_file))
        else:
            print(f"File not found: {in_file}")

    tgt_langs = ("spa", "fra", "deu", "ita", "hin", "cmn")

    for tgt_lang in tgt_langs:
        text_output, speech_output = translator.predict(
            input=in_file,
            task_str="s2st",
            tgt_lang=tgt_lang,
        )

        print(f"Translated text in {tgt_lang}: {text_output[0]}")
        print()

        out_file = f"/content/translated_LJ_{tgt_lang}.wav"

        torchaudio.save(out_file, speech_output.audio_wavs[0][0].to(torch.float32).cpu(), speech_output.sample_rate)

        print(f"Translated audio in {tgt_lang}:")
        if (play_output):
            audio=AudioSegment.from_wav(out_file)
            play(audio)    



if __name__ == "__main__":
    print("Pytorch Version: ", torch.__version__)
    print("CUDA Available: ", torch.cuda.is_available())  # This should be False on M1 Macs
    print("MPS Available: ", torch.backends.mps.is_available())  # This should be True if using macOS 12+ and PyTorch 1.12+

    # Initialize a Translator object with a multitask model, vocoder on the GPU.
    model_name = "seamlessM4T_v2_large"
    vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    translator = Translator(
        model_name,
        vocoder_name,
        device=device,
        dtype=dtype,
    )

    s2st_inference(play_output=True)