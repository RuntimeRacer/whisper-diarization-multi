import argparse
import os
from helpers import *
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
parser.add_argument(
    "--processing-dir",
    dest="processing_dir",
    default="temp_outputs",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
args = parser.parse_args()

# convert audio to mono for NeMo combatibility
sound = AudioSegment.from_file(args.audio).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, args.processing_dir)
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()
