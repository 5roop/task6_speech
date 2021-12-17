# %%
# 
from itertools import groupby
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import soundfile as sf

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, )
##############
# load model & audio and run audio through model
##############



tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "/home/peterr/macocu/task6_speech/", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)



model_name = '/home/peterr/macocu/task6_speech/7_/checkpoint-5200/'

model = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
def get_transcript(audio_filepath:str):
    speech, sample_rate = sf.read(audio_filepath)
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values.cuda()

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower()

    ##############
    # this is where the logic starts to get the start and end timestamp for each word
    ##############
    words = [w for w in transcription.split(' ') if len(w) > 0]
    predicted_ids = predicted_ids[0].tolist()
    duration_sec = input_values.shape[1] / sample_rate


    ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
    # remove entries which are just "padding" (i.e. no characers are recognized)
    ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
    # now split the ids into groups of ids where each group represents a word
    split_ids_w_time = [list(group) for k, group
                        in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                        if not k]

    assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

    word_start_times = []
    word_end_times = []
    for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
        _times = [_time for _time, _id in cur_ids_w_time]
        word_start_times.append(min(_times))
        word_end_times.append(max(_times))

    return words, word_start_times, word_end_times
# %%
import pandas as pd

df = pd.read_csv("12_new_files.csv")
df["words"] = ""
df["words_start_times"] = ""
df["words_end_times"] = ""

# %%

l = df.shape[0]
for row in df.iterrows():
    logging.info(f"\rWorking on file {row[0]:05} of {l}.")
    r = row[1]
    w, ws, we = get_transcript(r["path"])
    row[1]["words"] = str(w)
    row[1]["words_start_times"] = str(ws)
    row[1]["words_end_times"] = str(we)
    logging.info(f"Got transcript: {w}")
    df.to_csv("13_results.csv", index=False)
# %%
