# %%
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from datasets import load_dataset, load_metric, Audio
import datasets
import pandas as pd
import numpy as np
df = pd.read_csv("15_new_DS.csv")

LIM = int(df.shape[0] * 0.8)
common_voice_train_df = df.loc[0:LIM, :].copy()
common_voice_test_df = df.loc[LIM:, :].copy()


common_voice_train_df.reset_index(drop=True, inplace=True)
common_voice_test_df.reset_index(drop=True, inplace=True)


def load_audio(path):
    return datasets.Audio(sampling_rate=16000).decode_example(path)


# Adding audio
common_voice_train_df.loc[:, "audio"] = common_voice_train_df.path.apply(
    load_audio)
common_voice_test_df.loc[:,
                         "audio"] = common_voice_test_df.path.apply(load_audio)

# Initiating a dataset from pandas dataframe
common_voice_train_dataset = datasets.Dataset.from_pandas(
    common_voice_train_df)
common_voice_test_dataset = datasets.Dataset.from_pandas(common_voice_test_df)


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


common_voice_train_mapped = common_voice_train_dataset.map(
    prepare_dataset, remove_columns=common_voice_train_dataset.column_names)
common_voice_test_mapped = common_voice_test_dataset.map(
    prepare_dataset, remove_columns=common_voice_test_dataset.column_names)

# %%
repo_name = "16_"
wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


max_input_length_in_sec = 20
common_voice_train_mapped = common_voice_train_mapped.filter(
    lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_extractor()


training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    num_train_epochs=8,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=1,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train_mapped,
    eval_dataset=common_voice_test_mapped,
    tokenizer=processor.feature_extractor,
)

trainer.train()
