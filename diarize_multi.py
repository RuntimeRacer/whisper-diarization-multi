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
from deepmultilingualpunctuation import PunctuationModel
import re
import subprocess
import logging


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
        self.alignment_models = {}
        self.global_args = global_args

    def run(self) -> None:
        self.active = True
        # Initialize Whisper
        self.initialize_whisper(
            model_name=self.global_args.model_name,
            device=self.global_args.device,
            compute_type=self.global_args.compute_type
        )
        # Create a progress bar for this thread
        progress_bar = tqdm(total=len(self.files), desc=f"Thread {self.proc_id}")
        # Process audio one by one in this thread
        for audio in self.files:
            self.diarize_audio(audio)
            progress_bar.update(1)
        progress_bar.close()

    def initialize_whisper(self, model_name="medium.en", device="cpu", compute_type="float16"):
        # Initialize Whipser on GPU
        self.whisper_model = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )

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

        logging.info("Starting Nemo process with vocal_target: ", vocal_target)
        nemo_process = subprocess.Popen(
            ["python3", "nemo_process.py", "-a", vocal_target, "--device", device],
        )

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
        for segment in segments:
            whisper_results.append(segment._asdict())

        if info.language in wav2vec2_langs:
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
        else:
            word_timestamps = []
            for segment in whisper_results:
                for word in segment["words"]:
                    word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

        # Reading timestamps <> Speaker Labels mapping
        nemo_process.communicate()
        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, "temp_outputs_{0}".format(self.proc_id))

        speaker_ts = []
        with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if info.language in punct_model_langs:
            # restoring punctuation in the transcript to help realign the sentences
            punct_model = PunctuationModel(model="kredor/punctuate-all")

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list)

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

            wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        else:
            logging.warning(
                f"Punctuation restoration is not available for {info.language} language."
            )

        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        if not self.global_args.split_audio:
            with open(f"{os.path.splitext(audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
                get_speaker_aware_transcript(ssm, f)
            with open(f"{os.path.splitext(audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
                write_srt(ssm, srt)
        else:
            split_by_vad_and_speaker(audio, self.global_args.output_dir, ssm, self.global_args.sample_rate)

        # Clean processing data for this iteration
        cleanup(temp_path)


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

parser.add_argument(
    "-sd", "--include-subdirs",
    action="store_true",
    dest="include_subdirs",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

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
    help="number of threads to use per device",
)

parser.add_argument(
    "-t", "--threads",
    dest="threads",
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
    help="Target sample rate for splitted output files (if split enabled)",
)

args = parser.parse_args()

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
        start_idx = (d_id*2 + t_id) * files_per_thread
        end_idx = start_idx + files_per_thread
        if (end_idx + remainder) == files_total:
            end_idx += remainder
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
