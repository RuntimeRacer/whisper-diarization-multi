import argparse
import os
import fnmatch
import threading
from typing import Callable, Any, Iterable, Mapping

from helpers import *
from faster_whisper import WhisperModel
from tqdm import tqdm
import whisperx
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from transformers import pipeline
import re
import subprocess
import logging


# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--audio-dir", dest="audio_dir", help="name of the folder containing the target audio files", required=True
)

parser.add_argument(
    "-o", "--output-dir", dest="output_dir", help="name of the folder containing the processed audio and transcript files"
)

parser.add_argument(
    "-p", "--pattern",
    dest="pattern",
    default="*.mp3",
    help="pattern of the audio files to search for",
)

# parser.add_argument(
#     "-sd", "--include-subdirs",
#     action="store_true",
#     dest="include_subdirs",
#     default=False,
#     help="Recursively include sub-directories"
#     "Include directories below the audio-dir for searching files for transcribe.",
# )

parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--no-nemo",
    action="store_false",
    dest="nemo",
    default=True,
    help="Disable NeMo."
    "Disables NeMo for Speaker Diarization and relies completely on Whisper for Transcription.",
)

parser.add_argument(
    "--no-punctuation",
    action="store_false",
    dest="punctuation",
    default=True,
    help="Disable Punctuation Restore."
    "Disables punctuation restauration and relies completely on Whisper for Transcription.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--devices",
    dest="devices",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have one or multiple GPUs use 'cuda' for all cuda devices, or 'cuda:0' for single device, otherwise 'cpu'",
)

parser.add_argument(
    "-ct", "--compute-type",
    dest="compute_type",
    default="float16" if torch.cuda.is_available() else "int8",
    help="data type to use for loading the models",
)

parser.add_argument(
    "-t", "--threads",
    dest="threads",
    type=int,
    default=1,
    help="number of threads to use per device",
)

parser.add_argument(
    "-s", "--split-audio",
    action="store_true",
    dest="split_audio",
    default=False,
    help="Split Audio files on voice activity and speaker. Does not generate a .srt file.",
)

parser.add_argument(
    "-sr", "--sample-rate",
    dest="sample_rate",
    default=24000,
    help="Target sample rate for splitted output files (if split enabled). set to -1 to disable conversion.",
)

args = parser.parse_args()


class DiarizationDeviceThread(threading.Thread):

    def __init__(
            self,
            device_id,
            thread_id,
            device,
            files,
            global_args
    ):
        # execute the base constructor
        threading.Thread.__init__(self)
        # store params
        self.files = files
        self.device = device
        self.proc_id = "{0}_{1}".format(device_id, thread_id)
        # Init the worker
        self.active = False
        self.whisper_model = None
        self.msdd_model = None
        self.msdd_temp_path = None
        self.punctuation_model = None
        self.alignment_models = {}
        self.global_args = global_args

    def run(self) -> None:
        self.active = True
        # Initialize Whisper
        self.initialize_whisper(
            model_name=self.global_args.model_name,
            compute_type=self.global_args.compute_type
        )
        # Initialize NeMo Diarization model
        if self.global_args.nemo:
            self.initialize_nemo()
        # Initialize Punctuation model - Properly
        if self.global_args.punctuation:
            self.initialize_punctuation()
        # Create a progress bar for this thread
        progress_bar = tqdm(total=len(self.files), desc=f"Thread {self.proc_id}")
        # Process audio one by one in this thread
        for audio in self.files:
            self.diarize_audio(audio)
            progress_bar.update(1)
            # Clear cache after each run
            # torch.cuda.empty_cache()

        # Finalize Progress bar
        progress_bar.close()

        # Clean temp dir
        if self.msdd_temp_path:
            cleanup(self.msdd_temp_path)

    def initialize_whisper(self, model_name="medium.en", compute_type="float16"):
        # Initialize Whipser on GPU
        if "cuda:" in self.device:
            device_target = self.device.split(":")
            self.whisper_model = WhisperModel(
                model_name, device=device_target[0], device_index=int(device_target[1]), compute_type=compute_type
            )
        else:
            self.whisper_model = WhisperModel(
                model_name, device=device, compute_type=compute_type
            )

    def initialize_nemo(self):
        # Initialize NeMo on GPU
        self.msdd_temp_path = os.path.join(os.getcwd(), "temp_outputs_{0}".format(self.proc_id))
        os.makedirs(self.msdd_temp_path, exist_ok=True)
        # Ensure no logspam
        logger = logging.getLogger('nemo_logger')
        logger.setLevel(logging.ERROR)  # Set log level to 'error'
        self.msdd_model = NeuralDiarizer(cfg=create_config(self.msdd_temp_path)).to(self.device)

    def initialize_punctuation(self):
        if "cuda:" in self.device:
            device_target = self.device.split(":")
            self.punctuation_model = PunctuationModel(model="kredor/punctuate-all")
            del self.punctuation_model.pipe
            self.punctuation_model.pipe = pipeline(
                "ner",
                "kredor/punctuate-all",
                aggregation_strategy="none",
                device=int(device_target[1])
            )
        else:
            self.punctuation_model = PunctuationModel(model="kredor/punctuate-all")

    def diarize_audio(
            self,
            audio,
    ):
        if self.global_args.stemming:
            # Isolate vocals from the rest of the audio
            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o "temp_outputs"'
            )

            if return_code != 0:
                logging.warning(
                    "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
                )
                vocal_target = audio
            else:
                vocal_target = os.path.join(
                    "temp_outputs",
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio

        if self.global_args.suppress_numerals:
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.whisper_model.hf_tokenizer)
        else:
            numeral_symbol_tokens = None

        segments, info = self.whisper_model.transcribe(
            vocal_target,
            beam_size=5,
            word_timestamps=True,
            suppress_tokens=numeral_symbol_tokens,
            vad_filter=True,
        )
        whisper_results = []
        word_timestamps = []
        for segment in segments:
            whisper_results.append(segment._asdict())

        if self.global_args.nemo and info.language in wav2vec2_langs:
            try:
                # Load alignment model only once to speed up processing
                if info.language not in self.alignment_models:
                    alignment_model, metadata = whisperx.load_align_model(
                        language_code=info.language, device=device
                    )
                    self.alignment_models[info.language] = {
                        "model": alignment_model,
                        "meta": metadata
                    }
                else:
                    alignment_model = self.alignment_models[info.language]["model"]
                    metadata = self.alignment_models[info.language]["meta"]

                result_aligned = whisperx.align(
                    whisper_results, alignment_model, metadata, vocal_target, device
                )
                word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

            except IndexError as e:
                logging.warning("Forced alignment not possible for file {0}. Error: {1} ".format(vocal_target, e))
                logging.warning("Applying whisper timestamps...")

        if len(word_timestamps) == 0:
            for segment in whisper_results:
                for word in segment["words"]:
                    word_timestamps.append({"word": word[2].strip(), "start": word[0], "end": word[1]})

        if self.global_args.nemo:
            # Speaker Labeling using NeMo
            # convert audio to mono for NeMo combatibility
            sound = AudioSegment.from_file(vocal_target).set_channels(1)
            sound.export(os.path.join(self.msdd_temp_path, "mono_file.wav"), format="wav")
            try:
                self.msdd_model.diarize()
            except (ValueError, IndexError, KeyError) as e:
                logging.warning("File {0} could not be processed, likely no speech data. Error: {1} ".format(vocal_target, e))
                logging.warning("skipping...")
                return

            speaker_ts = []
            with open(os.path.join(self.msdd_temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line_list = line.split(" ")
                    s = int(float(line_list[5]) * 1000)
                    e = s + int(float(line_list[8]) * 1000)
                    speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

            wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        else:
            wsm = get_words_mapping(word_timestamps)

        if self.global_args.punctuation:
            if info.language in punct_model_langs and len(wsm) > 0:
                # restoring punctuation in the transcript to help realign the sentences
                words_list = list(map(lambda x: x["word"], wsm))
                labled_words = self.punctuation_model.predict(words_list)

                ending_puncts = ".?!"
                model_puncts = ".,;:!?"

                # We don't want to punctuate U.S.A. with a period. Right?
                is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

                for word_dict, labeled_tuple in zip(wsm, labled_words):
                    word = word_dict["word"]
                    if (
                            word
                            and labeled_tuple[1] in ending_puncts
                            and (word[-1] not in model_puncts or is_acronym(word))
                    ):
                        word += labeled_tuple[1]
                        if word.endswith(".."):
                            word = word.rstrip(".")
                        word_dict["word"] = word
                if self.global_args.nemo:
                    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
            else:
                logging.warning(
                    f"Punctuation restoration is not available for {info.language} language and empty utterances."
                )

        if self.global_args.nemo:
            ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
        else:
            ssm = get_sentences(wsm)

        if self.global_args.nemo:
            if not self.global_args.split_audio:
                with open(f"{os.path.splitext(audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
                    get_speaker_aware_transcript(ssm, f)
                with open(f"{os.path.splitext(audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
                    write_srt(ssm, srt)
            else:
                split_by_vad_and_speaker(
                    audio,
                    self.global_args.audio_dir,
                    self.global_args.output_dir,
                    ssm,
                    self.global_args.sample_rate
                )
        else:
            save_transcript(
                audio,
                self.global_args.audio_dir,
                self.global_args.output_dir,
                ssm,
                self.global_args.sample_rate
            )


# Ensure output dir
if not args.output_dir or args.output_dir in (None, ""):
    args.output_dir = args.audio_dir
os.makedirs(args.output_dir, exist_ok=True)

# Determine available devices
if args.devices == "cuda":
    selected_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
else:
    # Split devices by comma
    selected_devices = args.devices.split(",")

# Get all files in target dir and subdirectories
file_list = []
for path, _, files in os.walk(args.audio_dir):
    for filename in files:
        if fnmatch.fnmatch(filename, f'{args.pattern}'):
            file_list.append(os.path.join(path, filename))


# Build Processing pipeline
thread_list = []
total_threads = len(selected_devices) * args.threads
files_total = len(file_list)
files_per_thread = files_total // total_threads
remainder = files_total % total_threads

for d_id, device in enumerate(selected_devices):
    for t_id in range(0, args.threads):
        # Get the subset of files for the thread
        start_idx = (d_id*args.threads + t_id) * files_per_thread
        end_idx = start_idx + files_per_thread
        if t_id < remainder:
            end_idx += 1
        files_subset = file_list[start_idx:end_idx]

        # Init the thread
        processing_thread = DiarizationDeviceThread(
            device_id=d_id,
            thread_id=t_id,
            files=files_subset,
            device=device,
            global_args=args
        )
        thread_list.append(processing_thread)

# Star processing for all threads
for thread in thread_list:
    thread.start()

# Wait for all threads to finish
for thread in thread_list:
    thread.join()

print("All threads have finished. Thanks for playing")
