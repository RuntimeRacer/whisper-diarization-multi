import argparse
import logging
import sys
import time
import torch
import subprocess

from logging import Formatter

if __name__ == "__main__":

    # Init logger
    Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

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
    parser.add_argument("-pd", "--processing_dir", type=str, default='~/diarize_worker_tmp/', help="path to the processing directory to store temporary files")
    parser.add_argument("-cs", "--cache_size", type=int, default=1, help="amount of messages to cache while processing")
    parser.add_argument("-m", "--model", type=str, default="medium.en", help="name of the Whisper model to use")

    # Spawner params
    parser.add_argument("-t", "--threads", type=int, default=1, help="amount of threads to spawn per device")
    parser.add_argument("-l", "--logdir", type=str, default='/var/log/diarize_workers/', help="path to the processing directory to store temporary files")

    # Get args from parser
    args = parser.parse_args()

    # Evaluate devices
    devices = []
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for dId in range(torch.cuda.device_count()):
            devices.append('cuda:{0}'.format(dId))
    else:
        # Only use 1 worker Thread per CPU device; torch will employ everything else to work towards the ML task
        devices.append("cpu")
        args.threads = 1

    # Spawn processes for each device
    process_handles = []
    for dId, device in devices:
        for tId in range(args.threads):
            worker_args = [
                "python3",
                "diarize_worker.py",
                "-u",
                args.rabbitmq_user,
                "-p",
                args.rabbitmq_pass,
                "-rh",
                args.rabbitmq_host,
                "-rp",
                args.rabbitmq_port,
                "-pl",
                args.poll_channel,
                "-pu",
                args.push_channel,
                "-pd",
                args.processing_dir,
                "-cs",
                args.cache_size,
                "--whisper-model",
                args.model,
                ">",
                "{0}worker-{1}_thread-{2}".format(args.logdir, dId, tId)
            ]

            handle = subprocess.Popen(worker_args)
            process_handles.append(handle)

    # Run continous control loop until we abort the script
    logging.info("Launched {0} worker tasks on {1} devices".format(len(process_handles), len(devices)))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Manual Interrupt!")
        # Try to kill all workers
        for handle in process_handles:
            try:
                handle.kill()
            except Exception as e:
                logging.warning("Failed to kill worker process: {0}".format(str(e)))
                pass

        # Shutdown the script
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)





