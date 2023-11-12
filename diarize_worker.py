# RabbitMQ request processor for Cloud-Diarize - (c) 2023 RuntimeRacer
import base64
import json
import os
import argparse
import sys
import threading
import time

import pika
import torch
import requests
import logging

from helpers import *
from faster_whisper import WhisperModel
from pathlib import Path
from logging import Formatter


class DiarizeWorker:
    def __init__(
            self,
            rabbitmq_user: str,
            rabbitmq_pass: str,
            rabbitmq_host: str,
            rabbitmq_port: int,
            poll_channel: str,
            push_channel: str,
            processing_dir: str = '/tmp/',
            model_name: str = 'medium.en',
            device: str = 'cuda:0',
            cache_size: int = 1
    ):
        # Very simple validity checks
        if len(rabbitmq_user) == 0:
            raise RuntimeError("rabbitmq_user not set")
        if len(rabbitmq_pass) == 0:
            raise RuntimeError("rabbitmq_pass not set")
        if len(rabbitmq_host) == 0:
            raise RuntimeError("rabbitmq_host not set")
        if len(poll_channel) == 0:
            raise RuntimeError("poll_channel not set")
        if len(push_channel) == 0:
            raise RuntimeError("push_channel not set")
        if len(processing_dir) == 0:
            raise RuntimeError("temp_dir not set")
        if cache_size < 0:
            raise RuntimeError("invalid cache size")

        # Setup Params
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_pass = rabbitmq_pass
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.poll_channel = poll_channel
        self.push_channel = push_channel
        self.processing_dir = processing_dir
        self.model_name = model_name
        self.device = device

        # Diarize Params
        self.whisper_model = None

        # Message Handling params
        self.connection_active = False
        self.cache_thread = None
        self.cache_size = cache_size
        self.cached_messages = []
        self.polling_connection = None
        self.polling_channel_ref = None
        self.pushing_connection = None
        self.pushing_channel_ref = None

    def run(self):
        # Initialize Whisper
        self.initialize_whisper(model_name=self.model_name)
        # Connect to RabbitMQ
        while not self.connection_active:
            try:
                logging.info("Polling Worker: Establishing connection...")
                self.polling_connection = pika.BlockingConnection(pika.ConnectionParameters(
                    host=self.rabbitmq_host,
                    port=self.rabbitmq_port,
                    credentials=pika.credentials.PlainCredentials(username=self.rabbitmq_user, password=self.rabbitmq_pass),
                    heartbeat=30
                ))
                self.polling_channel_ref = self.polling_connection.channel()
                self.polling_channel_ref.queue_declare(queue=self.poll_channel, durable=True)
                self.connection_active = True
                logging.info("Polling Worker: Successfully connected to RabbitMQ host")
            except RuntimeError as e:
                self.connection_active = False
                logging.error("Polling Worker: Unable to connect to RabbitMQ host: {}".format(str(e)))
                logging.error("Polling Worker: Retrying in 10 seconds...")
                time.sleep(10)
                continue

            # Add cache processing thread and start it
            self.cache_thread = CacheProcessingThread(worker=self)
            self.cache_thread.start()

            # Start listening
            try:
                self.polling_channel_ref.basic_consume(queue=self.poll_channel, on_message_callback=self.handle_prompt_message, auto_ack=True)
                logging.info("Listening for messages on queue {}...".format(self.poll_channel))
                self.polling_channel_ref.start_consuming()
            except Exception as e:
                logging.error("Lost connection to RabbitMQ host: {}".format(str(e)))
                self.connection_active = False
                logging.error("Waiting for processing thread to finish...")
                # Shutdown cache processing thread
                self.cache_thread.active = False
                self.cache_thread.join()

    def initialize_whisper(self, model_name="medium.en", compute_type="float16"):
        # Initialize Whipser on GPU
        if "cuda:" in self.device:
            device_target = self.device.split(":")
            self.whisper_model = WhisperModel(
                model_name, device=device_target[0], device_index=int(device_target[1]), compute_type=compute_type
            )
        else:
            self.whisper_model = WhisperModel(
                model_name, device=self.device, compute_type=compute_type
            )

    """
    Expecting the following structure in param 'body':    
    {
        "MessageID": str, // ID of the message defined by the sender, can be general UUID
        "MessageBody": str // Request Body 
        "MessageMetadata": str // Metadata for context to be shared between request sender and result receiver
    }

    Returns the following structure in result:
    {
        "MessageID": str, // ID of the message defined by the sender, can be general UUID
        "MessageMetadata": str // Metadata for context to be shared between request sender and result receiver
        "Result": str // Body of the response
    }
    """

    def handle_prompt_message(self, channel, method, properties, body):
        # Parse message
        logging.info("Received message from channel '{}': {}".format(self.poll_channel, body))
        data = json.loads(body)
        message_id = data['MessageID'] if 'MessageID' in data else ''
        message_body = data['MessageBody'] if 'MessageBody' in data else ''
        message_metadata = data['MessageMetadata'] if 'MessageMetadata' in data else ''

        # Only process valid data
        if len(message_id) == 0 or len(message_body) == 0:
            logging.warning("Message received was invalid. Skipping...")
            return

        # Add to cache - Before checking cache size
        # This ensures we always process at least one message, even if cache is set to 0, because
        # due to async nature oft the worker, it will always pull at least 1 message no matter what.
        # Cache check will check cache size after adding the message, so this pulling connection is blocked until
        # Cache is freed up again.
        self.cached_messages.append({
            'MessageID': message_id,
            'MessageBody': message_body,
            'MessageMetadata': message_metadata
        })
        logging.debug("Added message with ID {} to cache".format(message_id))

        # Check for Cache capacity and block if reached
        if len(self.cached_messages) > self.cache_size:
            logging.debug("Cache is full, waiting for clearance...".format(message_id))
        while len(self.cached_messages) > self.cache_size:
            time.sleep(0.1)

    def process_cached_messages(self):
        while len(self.cached_messages) > 0:
            # Get first message from cache
            message = self.cached_messages[0]
            # Decode Message Body from Base64 to binary
            base64_data = message['MessageBody'].encode('utf-8')
            # Get filename from metadata
            metadata = message['MessageMetadata']
            if 'filename' not in metadata:
                logging.error("message does not contain filename; required for processing, skipping...")
                continue
            # Save the file into the temporary folder
            os.makedirs(self.processing_dir, exist_ok=True)
            file_path = Path(self.processing_dir).joinpath(metadata['filename'])
            with open(file_path, "wb") as file:
                binary_content = base64.decodebytes(base64_data)
                file.write(binary_content)
            # Process the message
            result_data = self.diarize_audio(file_path)
            # delete the temp file after processing
            os.remove(file_path)

            # Remove message from cache, AFTER being processed
            self.cached_messages.pop(0)

            # Build Result
            result = {
                "MessageID": message['MessageID'],
                "MessageMetadata": message['MessageMetadata'],
                "MessageBody": result_data,
            }
            result_json = json.dumps(result)

            # Publish to result queue
            logging.info("Sending result for message ID '{}': {}".format(message['MessageID'], result_json))
            result_sent = False
            while not result_sent:
                try:
                    self.pushing_connection = pika.BlockingConnection(pika.ConnectionParameters(
                        host=self.rabbitmq_host,
                        port=self.rabbitmq_port,
                        credentials=pika.credentials.PlainCredentials(username=self.rabbitmq_user, password=self.rabbitmq_pass),
                        heartbeat=30
                    ))
                    self.pushing_channel_ref = self.pushing_connection.channel()
                    self.pushing_channel_ref.queue_declare(queue=self.push_channel, durable=True)
                    self.pushing_channel_ref.basic_publish(exchange='', routing_key=self.push_channel, body=result_json)
                    result_sent = True
                    self.pushing_channel_ref.close()
                    self.pushing_connection.close()
                except Exception as e:
                    if not result_sent:
                        logging.error("Failed to send result: {0}".format(str(e)))
                        logging.error("Retrying in 10 seconds...")
                        time.sleep(10)
                        continue
                    else:
                        logging.warning("Exception after sending result: {0}".format(str(e)))

    def diarize_audio(
            self,
            audio,
    ):
        # TODO: Add all functionality from base method in diarize_multi.py
        segments, info = self.whisper_model.transcribe(
            audio,
            beam_size=5,
            word_timestamps=True,
            suppress_tokens=None,
            vad_filter=True,
        )
        whisper_results = []
        word_timestamps = []
        for segment in segments:
            whisper_results.append(segment._asdict())

        if len(word_timestamps) == 0:
            for segment in whisper_results:
                for word in segment["words"]:
                    word_timestamps.append({"word": word[2].strip(), "start": word[0], "end": word[1]})

        wsm = get_words_mapping(word_timestamps)
        ssm = get_sentences(wsm)
        return ssm

    def shutdown(self):
        if self.polling_channel_ref is not None:
            self.polling_channel_ref.stop_consuming()
        if self.polling_connection is not None:
            self.polling_connection.close()
        if self.cache_thread is not None:
            self.cache_thread.active = False
            self.cache_thread.join()


class CacheProcessingThread(threading.Thread):

    def __init__(self, worker: DiarizeWorker):
        # execute the base constructor
        threading.Thread.__init__(self)
        # store the reference
        self.worker = worker
        # Internal vars
        self.active = False

    def run(self):
        if self.worker is not None:
            logging.info("Starting Cache Processing Thread")
            self.active = True
            while self.active:
                self.worker.process_cached_messages()
                # Sleep if not processing anything
                time.sleep(0.01)
        else:
            raise RuntimeError("RabbitMQ Worker not initialized")


if __name__ == "__main__":
    global _polling_connection, _polling_channel, _pushing_connection

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

    # Parse from arguments
    parser = argparse.ArgumentParser()

    # MQ Parameters
    parser.add_argument("-u", "--rabbitmq_user", type=str, help="username to establish broker connection")
    parser.add_argument("-p", "--rabbitmq_pass", type=str, help="password to establish broker connection")
    parser.add_argument("-rh", "--rabbitmq_host", type=str, help="host of the rabbitmq server")
    parser.add_argument("-rp", "--rabbitmq_port", type=int, default=5672, help="port of the rabbitmq server")
    parser.add_argument("-pl", "--poll_channel", type=str, default="diarize_tasks", help="name if the rabbitmq channel to subscribe to")
    parser.add_argument("-pu", "--push_channel", type=str, default="diarize_results", help="name if the rabbitmq channel to push results to")

    # Worker Parameters
    parser.add_argument("-pd", "--processing_dir", type=str, default='/tmp/', help="path to the processing directory to store temporary files")
    parser.add_argument("-cs", "--cache_size", type=int, default=1, help="amount of messages to cache while processing")
    parser.add_argument("--whisper-model", dest="model_name", default="medium.en", help="name of the Whisper model to use")
    parser.add_argument("--device", dest="device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="specifies device to execute this worker on")

    # Run
    args = parser.parse_args()
    worker = DiarizeWorker(**vars(args))
    try:
        worker.run()
    except KeyboardInterrupt:
        logging.info("Manual Interrupt!")
        # Try clean shutdown
        worker.shutdown()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
