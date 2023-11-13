import argparse
import base64
import fnmatch
import threading
import time
import uuid
import sys
import functools
import pika

from helpers import *

import logging
from logging import Formatter


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
    "-t", "--threads",
    dest="threads",
    type=int,
    default=8,
    help="number of transcription result processing threads",
)

parser.add_argument(
    "-qs", "--queue_size",
    dest="queue_size",
    type=int,
    default=100,
    help="max files to be uploaded to the RabbitMQ Queue",
)

parser.add_argument(
    "-qt", "--queue_timeout",
    dest="queue_timeout",
    type=int,
    default=21600,
    help="max time in seconds for a file to stay pending in queue before being removed",
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


def ack_message(ch, delivery_tag):
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        pass


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
                    "MessageID": str(uuid.uuid4()),
                    "MessageMetadata": {
                        "filename": os.path.basename(next_file)
                    },
                    "MessageBody": base64_data,
                }
                message_json = json.dumps(message)

                # Publish to task queue
                logging.info("Sending task message ID '{0}' for file {1} ({2} bytes)".format(message['MessageID'], next_file, len(message_json)))
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
                            logging.error("Failed to send task message: {0}".format(str(e)))
                            logging.error("Retrying in 10 seconds...")
                            time.sleep(10)
                            continue
                        else:
                            logging.warning("Exception after sending task message: {0}".format(str(e)))

                # Mark file as pending
                self.files_in_queue[message['MessageID']] = {
                    "path": next_file,
                    "submit_time": time.time()
                }

                # As long as we're filling up the queue and have pending files left, do not check for timeouts yet
                continue

            # Check for timeouts among files in queue
            # files_to_reset = []
            # for message_id, file_status in self.files_in_queue.items():
            #     current_time = time.time()
            #     submit_time = file_status['submit_time']
            #     seconds_elapsed = current_time - submit_time
            #     if seconds_elapsed > self.processing_timeout:
            #         files_to_reset.append(message_id)
            #
            # # Remove files in timeout from queue and append them back to the pending list
            # for message_id in files_to_reset:
            #     filepath = self.files_in_queue[message_id]['path']
            #     del self.files_in_queue[message_id]
            #     self.pending_files.append(filepath)
            #     logging.info("Removed file {0} from in-progress-queue due to {1} seconds of inactivity.".format(filepath, self.processing_timeout))

            # Print status info
            queued = len(self.files_in_queue)
            pending = len(self.pending_files)
            hours, rem = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logging.info("--------------------------------------------------------------------------------------------------")
            logging.info("Currently {0} files in queue; {1} pending files of {2} files total. {3}% done - Time elapsed: {4}.".format(
                queued,
                pending,
                self.total_files_count,
                round((1-((pending+queued)/self.total_files_count))*100, 2),
                "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            ))
            logging.info("--------------------------------------------------------------------------------------------------")

            # Sleep 10 second between each loop
            time.sleep(10)

        # Everything has been successfully processed. End thread
        logging.info("All files have been processed successfully")
        self.active = False

    def get_task_metadata(self, message_id):
        return self.files_in_queue[message_id]

    def mark_task_complete(self, message_id):
        del self.files_in_queue[message_id]


class DiarizationResultProcessor(threading.Thread):

    def __init__(
            self,
            thread_id,
            upload_worker,
            global_args,
            cache_size = 1
    ):
        # execute the base constructor
        threading.Thread.__init__(self)
        # Init the worker
        self.thread_id = thread_id
        self.upload_worker = upload_worker
        self.active = False
        self.global_args = global_args

        # Message Handling params
        self.connection_active = False
        self.cache_thread = None
        self.cache_size = cache_size
        self.cached_messages = []
        self.polling_connection = None
        self.polling_channel_ref = None

    def run(self) -> None:
        self.active = True
        while self.active:
            # Connect to RabbitMQ
            if not self.connection_active:
                try:
                    logging.info("DiarizationResultProcessor-{0}: : Establishing connection...".format(self.thread_id))
                    self.polling_connection = pika.BlockingConnection(pika.ConnectionParameters(
                        host=self.global_args.rabbitmq_host,
                        port=self.global_args.rabbitmq_port,
                        credentials=pika.credentials.PlainCredentials(username=self.global_args.rabbitmq_user, password=self.global_args.rabbitmq_pass),
                        heartbeat=30
                    ))
                    self.polling_channel_ref = self.polling_connection.channel()
                    self.polling_channel_ref.basic_qos(prefetch_count=self.cache_size)
                    self.polling_channel_ref.queue_declare(queue=self.global_args.result_channel, durable=True)
                    self.connection_active = True
                    logging.info("DiarizationResultProcessor-{0}: : Successfully connected to RabbitMQ host".format(self.thread_id))
                except RuntimeError as e:
                    self.connection_active = False
                    logging.error("DiarizationResultProcessor-{0}: : Unable to connect to RabbitMQ host: {1}".format(self.thread_id, str(e)))
                    logging.error("DiarizationResultProcessor-{0}: : Retrying in 10 seconds...".format(self.thread_id))
                    time.sleep(10)
                    continue

            # Add cache processing thread and start it
            self.cache_thread = CacheProcessingThread(worker=self)
            self.cache_thread.start()

            # Start listening
            try:
                self.polling_channel_ref.basic_consume(queue=self.global_args.result_channel, on_message_callback=self.handle_result_message)
                logging.info("DiarizationResultProcessor-{0}: Listening for messages on queue {1}...".format(self.thread_id, self.global_args.result_channel))
                self.polling_channel_ref.start_consuming()
            except Exception as e:
                logging.error("DiarizationResultProcessor-{0}: Lost connection to RabbitMQ host: {1}".format(self.thread_id, str(e)))
                self.connection_active = False
                # Shutdown cache processing thread
                logging.error("DiarizationResultProcessor-{0}: Waiting for processing thread to finish...".format(self.thread_id))
                self.cache_thread.active = False
                self.cache_thread.join()

        # Shutdown cache processing thread
        logging.error("DiarizationResultProcessor-{0}: Waiting for processing thread to finish...".format(self.thread_id))
        self.cache_thread.active = False
        self.cache_thread.join()
        logging.info("DiarizationResultProcessor-{0} finished execution".format(self.thread_id))

    def shutdown(self):
        self.active = False
        try:
            self.polling_channel_ref.close()
            self.polling_connection.close()
        except Exception as e:
            logging.warning("DiarizationResultProcessor-{0}: Error on closing connection: {1}".format(self.thread_id, str(e)))
            pass
        self.connection_active = False

    def handle_result_message(self, channel, method, properties, body):
        # Parse message
        logging.info("DiarizationResultProcessor-{0}: Received message from channel '{1}'".format(self.thread_id, self.global_args.result_channel))
        data = json.loads(body)
        message_id = data['MessageID'] if 'MessageID' in data else ''
        message_body = data['MessageBody'] if 'MessageBody' in data else ''
        message_metadata = data['MessageMetadata'] if 'MessageMetadata' in data else ''

        # Only process valid data
        if len(message_id) == 0:
            logging.warning("DiarizationResultProcessor-{0}: Message received was invalid. Skipping...".format(self.thread_id))
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return
        elif len(message_body) == 0:
            logging.warning("DiarizationResultProcessor-{0}: Message body received was empty. Assuming no valid speech in audio...".format(self.thread_id))
            self.upload_worker.mark_task_complete(message_id)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        logging.debug("DiarizationResultProcessor-{0}: Received result for task message with ID '{1}'".format(self.thread_id, message_id))

        # Add to cache - Before checking cache size
        # This ensures we always process at least one message, even if cache is set to 0, because
        # due to async nature oft the worker, it will always pull at least 1 message no matter what.
        # Cache check will check cache size after adding the message, so this pulling connection is blocked until
        # Cache is freed up again.
        self.cached_messages.append({
            'MessageID': message_id,
            'MessageBody': message_body,
            'MessageMetadata': message_metadata,
            'ChannelRef': channel,
            'DeliveryTag': method.delivery_tag
        })
        logging.debug("DiarizationResultProcessor-{0}: Added message with ID '{1}' to cache".format(self.thread_id, message_id))

        # Check for Cache capacity and block if reached
        if len(self.cached_messages) > self.cache_size:
            logging.debug("DiarizationResultProcessor-{0}: Cache is full, waiting for clearance...".format(self.thread_id))
        # while len(self.cached_messages) > self.cache_size:
        #     time.sleep(5)
        #     self.polling_connection.process_data_events()

    def process_cached_messages(self):
        while len(self.cached_messages) > 0:
            # Get first message from cache
            message = self.cached_messages[0]
            message_id = message['MessageID']
            message_body = message['MessageBody']
            # Get RabbitMQ related data
            channel = message['ChannelRef']
            delivery_tag = message['DeliveryTag']

            # Get Metadata
            metadata = self.upload_worker.get_task_metadata(message_id)
            # Get Path from Metadata
            filepath = metadata['path']

            # Start Splitting the Audio based on result data
            self.split_transcribed_file(filepath, message_body)
            # Mark task as done in upload worker
            self.upload_worker.mark_task_complete(message_id)

            # Remove message from cache, AFTER being processed
            self.cached_messages.pop(0)
            # Send ACK to results channel
            cb = functools.partial(ack_message, channel, delivery_tag)
            self.polling_connection.add_callback_threadsafe(cb)

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


class CacheProcessingThread(threading.Thread):

    def __init__(self, worker: DiarizationResultProcessor):
        # execute the base constructor
        threading.Thread.__init__(self)
        # store the reference
        self.worker = worker
        # Internal vars
        self.active = False

    def run(self):
        if self.worker is not None:
            logging.info("DiarizationResultProcessor-{0}: Starting Cache Processing Thread".format(self.worker.thread_id))
            self.active = True
            while self.active:
                self.worker.process_cached_messages()
                # Sleep if not processing anything
                time.sleep(0.01)
        else:
            raise RuntimeError("RabbitMQ Worker not initialized")


# Init logger
Formatter.converter = time.gmtime
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %z'
)
# Pika log level to warning to avoid logspam
logging.getLogger("pika").setLevel(logging.WARNING)

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
logging.info("Detected {0} files to be diarized".format(files_total))

# Build Upload Worker
upload_worker = FileUploaderManagerThread(
    pending_files=file_list,
    global_args=args,
    max_queue_size=args.queue_size,
    processing_timeout=args.queue_timeout
)

# Build Result Processing Threads
for t_id in range(0, args.threads):
    # Init the thread
    processing_thread = DiarizationResultProcessor(
        thread_id=t_id,
        global_args=args,
        upload_worker=upload_worker
    )
    thread_list.append(processing_thread)

# Start Upload Worker
upload_worker.start()

# Start result threads
for thread in thread_list:
    thread.start()

# Wait for upload worker to finish (happens once last file in queue has been processed)
upload_worker.join()

# Shutdown all processing thread connections
for thread in thread_list:
    thread.shutdown()

# Wait for all threads to finish
for thread in thread_list:
    thread.join()

print("All threads have finished. Thanks for playing")
