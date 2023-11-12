import argparse
import base64
import os
import fnmatch
import threading
import time
import uuid
from typing import Callable, Any, Iterable, Mapping

from concurrent.futures import ThreadPoolExecutor

import pika

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

parser.add_argument("-u", "--rabbitmq_user", type=str, help="username to establish broker connection")
parser.add_argument("-pw", "--rabbitmq_pass", type=str, help="password to establish broker connection")
parser.add_argument("-rh", "--rabbitmq_host", type=str, help="host of the rabbitmq server")
parser.add_argument("-rp", "--rabbitmq_port", type=int, default=5672, help="port of the rabbitmq server")
parser.add_argument("-tc", "--task_channel", type=str, default="diarize_tasks", help="channel to push tasks into")
parser.add_argument("-rc", "--result_channel", type=str, default="diarize_results", help="channel to listen for results from diarization workers")

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


class FileUploaderManagerThread(threading.Thread):

    def __init__(
        self,
        pending_files,
        global_args,
        max_queue_size=100,
        processing_timeout=21600,
    ):
        # execute the base constructor
        threading.Thread.__init__(self)
        # store params
        self.global_args = global_args
        self.pending_files = pending_files
        self.total_files_count = len(pending_files)
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        # Init the worker
        self.active = False
        # Processing vars
        self.files_in_queue = {}
        self.pushing_connection = None
        self.pushing_channel_ref = None
    def run(self):
        self.active = True
        start_time = time.time()
        # This Thread is responsible for uploading pending files
        # It will run until all pending and processing files have been processed successfully
        while len(self.pending_files) > 0 or len(self.files_in_queue) > 0:
            if len(self.files_in_queue) < self.max_queue_size and len(self.pending_files) > 0:
                # Get next file from list
                next_file = self.pending_files.pop(0)
                # Base64 encode file and upload it to RabbitMQ
                with open(next_file, "rb") as file:
                    binary_content = file.read()
                base64_encoded = base64.b64encode(binary_content)
                base64_data = base64_encoded.decode('utf-8')

                # Build MQ Message
                message = {
                    "MessageID": uuid.uuid4(),
                    "MessageMetadata": {
                        "filename": os.path.basename(next_file)
                    },
                    "MessageBody": base64_data,
                }
                message_json = json.dumps(message)

                # Publish to task queue
                logging.info("Sending message ID '{}' for file {} ({} bytes)".format(message['MessageID'], next_file, len(message_json)))
                message_sent = False
                while not message_sent:
                    try:
                        self.pushing_connection = pika.BlockingConnection(pika.ConnectionParameters(
                            host=self.global_args.rabbitmq_host,
                            port=self.global_args.rabbitmq_port,
                            credentials=pika.credentials.PlainCredentials(username=self.global_args.rabbitmq_user, password=self.global_args.rabbitmq_pass),
                            heartbeat=30
                        ))
                        self.pushing_channel_ref = self.pushing_connection.channel()
                        self.pushing_channel_ref.queue_declare(queue=self.global_args.task_channel, durable=True)
                        self.pushing_channel_ref.basic_publish(exchange='', routing_key=self.global_args.task_channel, body=message_json)
                        message_sent = True
                        self.pushing_channel_ref.close()
                        self.pushing_connection.close()
                    except Exception as e:
                        if not message_sent:
                            logging.error("Failed to send tsk: {0}".format(str(e)))
                            logging.error("Retrying in 10 seconds...")
                            time.sleep(10)
                            continue
                        else:
                            logging.warning("Exception after sending task: {0}".format(str(e)))

                # Mark file as pending
                self.files_in_queue[next_file] = {
                    "path": next_file,
                    "submit_time": time.time()
                }

                # As long as we're filling up the queue and have pending files left, do not check for timeouts yet
                continue

            # Check for timeouts among files in queue
            files_to_reset = []
            for file_status in self.files_in_queue.values():
                current_time = time.time()
                submit_time = file_status['submit_time']
                seconds_elapsed = current_time - submit_time
                if seconds_elapsed > self.processing_timeout:
                    files_to_reset.append(file_status['path'])

            # Remove files in timeout from queue and append them back to the pending list
            for _, filepath in files_to_reset:
                del self.files_in_queue[filepath]
                self.pending_files.append(filepath)
                logging.info("Removed file {0} from in-progress-queue due to {1} seconds of inactivity.".format(filepath, self.processing_timeout))

            # Print status info
            pending = len(self.pending_files)
            hours, rem = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logging.info("Currently {0} files in queue; {1} pending files of {2} files total. {3}% done - Time elapsed: {4}.".format(
                len(self.files_in_queue),
                pending,
                self.total_files_count,
                round((1-(pending/self.total_files_count))*100, 2),
                "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            ))

            # Sleep 60 second between each loop
            time.sleep(60)

        # Everything has been successfully processed. End thread
        logging.info("All files have been processed successfully")
        self.active = False


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
