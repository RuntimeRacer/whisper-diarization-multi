import argparse
import os
import fnmatch
import threading
from typing import Callable, Any, Iterable, Mapping

from concurrent.futures import ThreadPoolExecutor
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

# parser.add_argument(
#     "--no-stem",
#     action="store_false",
#     dest="stemming",
#     default=True,
#     help="Disables source separation."
#     "This helps with long files that don't contain a lot of music.",
# )
#
# parser.add_argument(
#     "--suppress_numerals",
#     action="store_true",
#     dest="suppress_numerals",
#     default=False,
#     help="Suppresses Numerical Digits."
#     "This helps the diarization accuracy but converts all digits into written text.",
# )
#
# parser.add_argument(
#     "--no-nemo",
#     action="store_false",
#     dest="nemo",
#     default=True,
#     help="Disable NeMo."
#     "Disables NeMo for Speaker Diarization and relies completely on Whisper for Transcription.",
# )
#
# parser.add_argument(
#     "--no-punctuation",
#     action="store_false",
#     dest="punctuation",
#     default=True,
#     help="Disable Punctuation Restore."
#     "Disables punctuation restauration and relies completely on Whisper for Transcription.",
# )

parser.add_argument(
    "-ft", "--file-threads",
    dest="file_threads",
    type=int,
    default=20,
    help="number of file processing threads",
)

parser.add_argument(
    "-rt", "--result-threads",
    dest="result_threads",
    type=int,
    default=8,
    help="number of result processing threads",
)

# parser.add_argument(
#     "-s", "--split-audio",
#     action="store_true",
#     dest="split_audio",
#     default=False,
#     help="Split Audio files on voice activity and speaker. Does not generate a .srt file.",
# )

parser.add_argument(
    "-sr", "--sample-rate",
    dest="sample_rate",
    default=24000,
    help="Target sample rate for splitted output files (if split enabled). set to -1 to disable conversion.",
)

args = parser.parse_args()


class DiarizationProcessingThread(threading.Thread):

    def __init__(
            self,
            thread_id,
            result_executor,
            files,
            global_args
    ):
        # execute the base constructor
        threading.Thread.__init__(self)
        # store params
        self.files = files
        self.result_executor = result_executor
        # Init the worker
        self.thread_id = thread_id
        self.active = False
        self.global_args = global_args

    def run(self) -> None:
        self.active = True
        # Process audio one by one in this thread
        for audio in self.files:
            self.remote_diarize_audio(audio)
        logging.info('Thread ID {0} successfully scheduled all {1} files for processing.',format(self.thread_id))

    def remote_diarize_audio(self, audio):


    def split_transcribed_file(
            self,
            src_audio_path,
            sentence_data
    ):
        save_transcript_sync(
            src_audio_path,
            self.global_args.audio_dir,
            self.global_args.output_dir,
            sentence_data,
            self.global_args.sample_rate
        )


# Ensure output dir
if not args.output_dir or args.output_dir in (None, ""):
    args.output_dir = args.audio_dir
os.makedirs(args.output_dir, exist_ok=True)

# Get all files in target dir and subdirectories
file_list = []
for path, _, files in os.walk(args.audio_dir):
    for filename in files:
        if fnmatch.fnmatch(filename, f'{args.pattern}'):
            file_list.append(os.path.join(path, filename))


# Build Processing pipeline
thread_list = []
total_threads = args.threads
files_total = len(file_list)
files_per_thread = files_total // total_threads
remainder = files_total % total_threads
result_processing_pool = ThreadPoolExecutor(max_workers=args.result_threads)

for t_id in range(0, args.threads):
    # Get the subset of files for the thread
    start_idx = (args.threads + t_id) * files_per_thread
    end_idx = start_idx + files_per_thread
    if t_id < remainder:
        end_idx += 1
    files_subset = file_list[start_idx:end_idx]

    # Init the thread
    processing_thread = DiarizationProcessingThread(
        thread_id=t_id,
        files=files_subset,
        result_executor=result_processing_pool,
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
