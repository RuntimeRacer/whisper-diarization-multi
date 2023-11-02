<h1 align="center">Speaker Diarization Using OpenAI Whisper</h1>
<p align="center">
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/stargazers">
    <img src="https://img.shields.io/github/stars/MahmoudAshraf97/whisper-diarization.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/issues">
        <img src="https://img.shields.io/github/issues/MahmoudAshraf97/whisper-diarization.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/MahmoudAshraf97/whisper-diarization.svg"
             alt="GitHub license">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2FMahmoudAshraf97%2Fwhisper-diarization">
  <img src="https://img.shields.io/twitter/url/https/github.com/MahmoudAshraf97/whisper-diarization.svg?style=social" alt="Twitter">
  </a> 
  </a>
  <a href="https://colab.research.google.com/github/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
 
</p>
<h3 align="center">- Optimized for batch processing - </h3>

## Speaker Diarization Multiprocessing Fork
I created this fork because I needed to Diarize and label large datasets for LLM-Training.
Unfortunately, the base repo, and also an existing batch processing fork do not support directory-based processing, and
the other batch processing fork is largely outdated, too.

So I decided to extend the base repo with the following functionality:

- Directory-based batch processing: Diarize all files in a specified directory.
- Multi-GPU processing: Run parallel processes on multiple GPUs if availiable.
- Multi-Threaded processing: Run multiple processing threads per GPU if VRAM capacity is sufficient.
- Minimize I/O load and blocking calls

### Changes of this Fork in Detail

- Added a new execution script, called `diarize_multi.py`
- All models used during the transcription process will be kept in VRAM to avoid loading times.
- Changed some Script parameters:
  - `-d / --audio-dir`: points to folder with target audio files. Finds all files matching `-p`, inside, also in subdirectories.
  - `-o / --output-dir`: points to a target folder for the output. Output directory tree will maintain the folder structure of `-d`.
  - `-p / --pattern`: Pattern of files to search for. It is not checked if it's a valid format or pre-converted, except for what `faster-whisper` is doing.
So be careful to not specify a too broad pattern for whole directory trees which could include meta-files in `.json /.csv / .tsv` format etc. 
  - `--no-nemo`: Disables NeMo for Speaker Diarization and relies completely on Whisper for Transcription.
  - `--no-punctuation`: Disables punctuation restauration and relies completely on Whisper for Transcription.
  - `--devices`: Allows to specify multiple Devices for transcription. For each device, a separate handler with `-t` processing threads will be launched.
  - `-t / --threads`: number of processing threads to use per device
  - `-ct / --compute-type`: data type to use for loading the models
  - `-s / --split-audio`: Split Audio files on voice activity and speaker instead of generating an SRT file.
  - `-st / --sample-rate`: Target sample rate for splitted output files (if split enabled). Set to -1 to disable conversion.


### Benchmarks for Multiprocessing
I did some benchmark runs with a small set of japanese audio (515 short samples) on my AI training machine.
To speed things up a little more, plus the fact that I didn't see too much benefit running NeMo on top of / next to
whisper, I decided to run this benchmark only on whisper's VAD and transcription capabilities.

The machine specs:

```
AMD EPYC 7352 24-Core Processor (48 Threads) @ 3.20 Ghz
256 GB SAMSUNG ECC DDR4-3200
6x Nvidia RTX 3090 @ PCI-E 4.0 x8
```
Command used for Benchmark (cuda device IDs and thread count changed accordingly):
```
python diarize_multi.py -d ~/test/ja_short_samples -o ~/test/ja_short_samples_transcribed -p "*.flac" -sd --no-stem --whisper-model large-v2 --devices "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4" -t 2 -s --no-nemo --no-punctuation
```

Benchmark Results (number before 'x' is number of GPUs, number after 'x' is number of Threads per GPU):
```
515 Short samples (4-28 Seconds japanese audio)

1x1: 0:06:31
1x2: 0:04:49
1x3: 0:04:11
1x4: 0:04:08

2x1: 0:04:11
2x2: 0:03:49
2x3: 0:03:39

3x1: 0:03:37
3x2: 0:03:30
3x3: 0:03:45

4x1: 0:03:27
4x2: 0:03:38
4x3: 0:03:38

5x1: 0:03:23
5x2: 0:03:42
5x3: 0:03:37
5x4: 0:03:40

6x1: 0:03:30
6x2: 0:03:38
```
From the above results, there seems to be a "peak" of parallelization benefit to occur at around 4-8 threads globally.
The best result was achieved using 5 GPUs with one thread per device (5x1). However, the benefit was really low compared to
the results of 6x1, and 4x1 or 3x2. 

Overall, it seemed like the speedup reached a limit in this region. I did not run a cross-checking benchmark
with splitting the sample sets per python process and run them independently; however, I assume that the decrease
in performance speedup when adding additional devices is caused by Pythons global interpreter lock (GIL).
Maybe I will do more investigations / optimizations here when I do another iteration on this codebase.

## Original Readme
Speaker Diarization pipeline based on OpenAI Whisper
I'd like to thank [@m-bain](https://github.com/m-bain) for Wav2Vec2 forced alignment, [@mu4farooqi](https://github.com/mu4farooqi) for punctuation realignment algorithm

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star the project on github (see top-right corner) if you appreciate my contribution to the community!**

## What is it
This repository combines Whisper ASR capabilities with Voice Activity Detection (VAD) and Speaker Embedding to identify the speaker for each sentence in the transcription generated by Whisper. First, the vocals are extracted from the audio to increase the speaker embedding accuracy, then the transcription is generated using Whisper, then the timestamps are corrected and aligned using WhisperX to help minimize diarization error due to time shift. The audio is then passed into MarbleNet for VAD and segmentation to exclude silences, TitaNet is then used to extract speaker embeddings to identify the speaker for each segment, the result is then associated with the timestamps generated by WhisperX to detect the speaker for each word based on timestamps and then realigned using punctuation models to compensate for minor time shifts.


Whisper, WhisperX and NeMo parameters are coded into diarize.py and helpers.py, I will add the CLI arguments to change them later
## Installation
`FFMPEG` and `Cython` are needed as prerquisites to install the requirements
```
pip install cython
```
or
```
sudo apt update && sudo apt install cython3
```
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
```
pip install -r requirements.txt
```
## Usage 

```
python diarize.py -a AUDIO_FILE_NAME
```

If your system has enough VRAM (>=10GB), you can use `diarize_parallel.py` instead, the difference is that it runs NeMo in parallel with Whisper, this can be benifecial in some cases and the result is the same since the two models are nondependent on each other. This is still experimental, so expect errors and sharp edges. Your feedback is welcome.

## Command Line Options

- `-a AUDIO_FILE_NAME`: The name of the audio file to be processed
- `--no-stem`: Disables source separation
- `--whisper-model`: The model to be used for ASR, default is `medium.en`
- `--suppress_numerals`: Transcribes numbers in their pronounced letters instead of digits, improves alignment accuracy

## Known Limitations
- Overlapping speakers are yet to be addressed, a possible approach would be to separate the audio file and isolate only one speaker, then feed it into the pipeline but this will need much more computation
- There might be some errors, please raise an issue if you encounter any.

## Future Improvements
- Implement a maximum length per sentence for SRT
- Improve Batch Processing

## Acknowledgements
Special Thanks for [@adamjonas](https://github.com/adamjonas) for supporting this project
This work is based on [OpenAI's Whisper](https://github.com/openai/whisper) , [Faster Whisper](https://github.com/guillaumekln/faster-whisper) , [Nvidia NeMo](https://github.com/NVIDIA/NeMo) , and [Facebook's Demucs](https://github.com/facebookresearch/demucs)
