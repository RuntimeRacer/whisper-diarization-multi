import os
import wget
from pathlib import Path
from omegaconf import OmegaConf
import json
import shutil
import nltk
import subprocess
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]
wav2vec2_langs = list(DEFAULT_ALIGN_MODELS_TORCH.keys()) + list(
    DEFAULT_ALIGN_MODELS_HF.keys()
)


def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting, telephonic, or general based on domain type of the audio file
    CONFIG_LOCAL_DIRECTORY = "nemo_msdd_configs"
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    MODEL_CONFIG_PATH = os.path.join(CONFIG_LOCAL_DIRECTORY, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG_PATH):
        os.makedirs(CONFIG_LOCAL_DIRECTORY, exist_ok=True)
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG_PATH = wget.download(CONFIG_URL, MODEL_CONFIG_PATH)

    config = OmegaConf.load(MODEL_CONFIG_PATH)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config


def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["word"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping


def get_words_mapping(wrd_ts):
    wrd_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["word"],
        )
        wrd_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we}
        )
    return wrd_mapping


sentence_ending_punctuations = ".?!"


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list)
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )


def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping, max_words_in_sentence=50
):
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_sentences(word_mapping):
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    snts = []

    snt = None

    for wrd_dict in word_mapping:
        wrd = wrd_dict["word"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if not snt:
            snt = {
                "start_time": s,
                "end_time": e,
                "text": "",
            }

        if sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "

    if snt:
        snts.append(snt)

    return snts


def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    previous_speaker = sentences_speaker_mapping[0]["speaker"]
    text = sentences_speaker_mapping[0]["text"]

    for sentence_dict in sentences_speaker_mapping[1:]:
        sp = sentence_dict["speaker"]
        sentence = sentence_dict["text"]

        if sp != previous_speaker:
            f.write(f"{previous_speaker}: {text}\n\n")
            text = sentence
            previous_speaker = sp
        else:
            text += " " + sentence


def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert milliseconds >= 0, "non-negative timestamp expected"

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_srt(transcript, file):
    """
    Write a transcript to a file in SRT format.

    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def split_by_vad_and_speaker(base_file, base_dir, output_dir, transcript_data, sample_rate):
    # get base file name
    base_name = os.path.splitext(base_file)[0]
    base_name = base_name.replace(base_dir, output_dir)

    # Group all Segments by speaker
    speakers = {}
    for _, segment in enumerate(transcript_data):
        if segment['speaker'] not in speakers:
            speakers[segment['speaker']] = []
        speakers[segment['speaker']].append(segment)

    # For each speaker, create splitted output audio and transcripts
    for spk_idx, spk_key in enumerate(speakers.keys()):
        speaker_utterances = speakers[spk_key]
        for utt_idx, utt_segment in enumerate(speaker_utterances):
            utt_audio_name = "{0}_{1}_{2}.flac".format(base_name, spk_idx, utt_idx)
            utt_transcript_name = "{0}_{1}_{2}.txt".format(base_name, spk_idx, utt_idx)

            # Create utterance file if it doesen't exist
            dest_utterance_path = Path(utt_transcript_name)
            os.makedirs(os.path.dirname(dest_utterance_path), exist_ok=True)
            if not dest_utterance_path.is_file():
                with open(dest_utterance_path, "w", encoding="utf-8-sig") as out:
                    out.write(utt_segment["text"].strip())

            # if the file already exists, skip conversion
            dest_audio_path = Path(utt_audio_name)
            if dest_audio_path.is_file():
                continue

            # Process using FFMPEG
            if sample_rate > 0:
                convert_args = [
                    "/usr/bin/ffmpeg",
                    "-y",
                    "-loglevel",
                    "fatal",
                    "-i",
                    str(base_file),
                    "-ss",
                    format_timestamp(utt_segment['start_time'], always_include_hours=True),
                    "-to",
                    format_timestamp(utt_segment['end_time'], always_include_hours=True),
                    "-ar",
                    str(sample_rate),
                    "-threads",
                    str(1),
                    str(utt_audio_name)
                ]
                subprocess.Popen(convert_args)


def save_transcript(base_file, base_dir, output_dir, sentences, sample_rate):
    # get base file name
    base_name = os.path.splitext(base_file)[0]
    base_name = base_name.replace(base_dir, output_dir)

    for sentence_idx, sentence_segment in enumerate(sentences):
        snt_audio_name = "{0}_{1}.flac".format(base_name, sentence_idx)
        snt_transcript_name = "{0}_{1}_transcript.txt".format(base_name, sentence_idx)

        # Create utterance file if it doesen't exist
        dest_utterance_path = Path(snt_transcript_name)
        os.makedirs(os.path.dirname(dest_utterance_path), exist_ok=True)
        if not dest_utterance_path.is_file():
            with open(dest_utterance_path, "w", encoding="utf-8-sig") as out:
                out.write(sentence_segment["text"].strip())

        # if the file already exists, skip conversion
        dest_audio_path = Path(snt_audio_name)
        if dest_audio_path.is_file():
            continue

        # Process using FFMPEG
        if sample_rate > 0:
            convert_args = [
                "/usr/bin/ffmpeg",
                "-y",
                "-loglevel",
                "fatal",
                "-i",
                str(base_file),
                "-ss",
                format_timestamp(sentence_segment['start_time'], always_include_hours=True),
                "-to",
                format_timestamp(sentence_segment['end_time'], always_include_hours=True),
                "-ar",
                str(sample_rate),
                "-threads",
                str(1),
                str(snt_audio_name)
            ]
            subprocess.Popen(convert_args)


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = [
        -1,
    ]
    for token, token_id in tokenizer.get_vocab().items():
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def _get_next_start_timestamp(word_timestamps, current_word_index):
    # if current word is the last word
    if current_word_index == len(word_timestamps) - 1:
        return word_timestamps[current_word_index]["start"]

    next_word_index = current_word_index + 1
    while current_word_index < len(word_timestamps) - 1:
        if word_timestamps[next_word_index].get("start") is None:
            # if next word doesn't have a start timestamp
            # merge it with the current word and delete it
            word_timestamps[current_word_index]["word"] += (
                " " + word_timestamps[next_word_index]["word"]
            )

            word_timestamps[next_word_index]["word"] = None
            next_word_index += 1

        else:
            return word_timestamps[next_word_index]["start"]


def filter_missing_timestamps(word_timestamps):
    # handle the first and last word
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = 0
        word_timestamps[0]["end"] = _get_next_start_timestamp(word_timestamps, 0)

    result = [
        word_timestamps[0],
    ]

    for i, ws in enumerate(word_timestamps[1:], start=1):
        # if ws doesn't have a start and end
        # use the previous end as start and next start as end
        if ws.get("start") is None and ws.get("word") is not None:
            ws["start"] = word_timestamps[i - 1]["end"]
            ws["end"] = _get_next_start_timestamp(word_timestamps, i)

        if ws["word"] is not None:
            result.append(ws)
    return result


def cleanup(path: str):
    """path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))
