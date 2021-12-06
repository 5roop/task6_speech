# TASK 6: speech to text experiments

An [example](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) was reconstructed as an excercize. Some observations:
* A lot of workarounds were implemented to make the model work and not all of the aspects are as of yet clear to me
* Training time for 30 epochs was about 2h.

Discuss which evaluation metric to use (WER? PER?)

# Addendum 2021-12-03T08:18:03

The virtual machine does not have the necessary tools installed to convert the flac files to wav. I therefore migrated all the files to my computer and converted them there. The disk only has 14G of free space, meaning that in case the wav dataset is bigger than that, I will need to delete the original flacs.

I noticed that I also had the chance to downsample the original 44100 Hz sampling rate to the recommended 16000 Hz. In this conversion filesize dropped to about 33% of the original size, meaning that we will probably be ok regarding disk space on the virtual machine.

For posterity: file inspection was done with `soxi file.wav`. Conversion was performed using `ffmpeg -i 00023289.flac -ar 16000 brisi.wav`

After copying files back to VM, some toying was necessary to get it to load properly, but finally it was discovered how to do it.

The vocabulary found in the normalized transcripts is as follows: `{' ', '*', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ü', 'ć', 'č', 'đ', 'š', 'ž'}`. I will not transform non-Croatian characters now, but I will remove special characters (e.g. `*`, `:`...)

I noticed that the substitutions take a long time. It might be better to do it in `pandas` in the future.

I performed a train-test split randomly without setting the random seed. Train size was set to 0.8, meaning that 18430 instances were in the training split and 4608 were in the test split.


# Addendum 2021-12-03T14:04:13

The model would not train due to CUDA Out of memory errors. My usual trick `import torch;torch.cuda.empty_cache()` did not work and the `nvidia-smi` output showed there was still a lot of memory allocated to my processes. Fortunately this error is so ubiquitous that I found another approach:
```
from numba import cuda
cuda.select_device(0)
cuda.close()
cuda.select_device(0)
```

This successfully released CUDA memory. Training is done via the Trainer module, but for some unknown reason every time the training starts, something stalls the process for about 10-15 minutes, and then the training either starts or crashes, meaning that debugging is time-consuming. It also means that the README will probably be bloated again, because it gives me time to complain and log all difficulties and attempted fixes.

# Addendum 2021-12-03T17:07:06

It had been discovered that none of the tricks prevent the training from crashing. To explore further I first dropped the number of `per_device_train_batch_size` to 4, but to no avail. In the next step I only read 10k instances. It did not work. With the reduced dataset I proceeded to further reduce the batch size to 1. This worked. I therefore increased the batch size to 2 and tried again. If this works, the dataset will be expanded to use full data. It did not.

# Addendum 2021-12-06T07:40:36
To get it to work I clipped the audios as demonstrated in the example:
```python
max_input_length_in_sec = 5.0
train = train.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
```
The batch size was 2 and the first training episode ran OK with 2.5 it/s. Unfortunately the evaluation was way slower and crashed at the very end:
```ValueError: number of ground truth inputs (24) and hypothesis inputs (0) must match```

I opened a new notebook and repeated the pipeline again in case I missed some steps. It seemed the training was even slower this time somehow... I clipped all instances at 20s and set the train batch size at 2. The problem now arises at the evaluation stage. For some reason there is a ValueError raised at the point of evaluation. The weird thing is that the error gets raised at the very end of the evaluation. Probably this means that the speech2text works and only the evaluation crashes.

I tried rerunning the training pipeline with CER metric disabled, with a reduced dataset that only reads 2k files. That seemed to work, so I proceeded with training on the full dataset. For now I still have the length clipped at 5s.

Current speed is about 0.2 it/s when training and 0.5 it/s when evaluating. Per device train batch size was increased to 16.

# Addendum 2021-12-06T13:28:12

First and second evaluations inspire hope: WER 0.43 and 0.30. Total training time seem to will have been about 3h.

To do: implement filtering in the dataset construction part to allow for more elegant choice of input lengths.

File sizes in kB are distributed as follows:

```
                            Distribution of filesizes                         
    ┌────────────────────────────────────────────────────────────────────────┐
4422┤ ▐█▌                                                                    │
    │ ▐█▌                                                                    │
3685┤ ▐███▌                                                                  │
    │█████▌                                                                  │
    │█████▌                                                                  │
2948┤█████▙▄                                                                 │
    │███████                                                                 │
    │███████▄▄                                                               │
2211┤█████████                                                               │
    │█████████                                                               │
    │██████████▌                                                             │
1474┤██████████▌                                                             │
    │████████████▌                                                           │
    │████████████▙▄▖                                                         │
 737┤████████████████                                                        │
    │██████████████████                                                      │
   0┤███████████████████████▙▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄                ▗▄│
    └┬─────┬──────┬──────┬──────┬───────┬──────┬──────┬──────┬───────┬───────┘
    18.8 293.0  658.0  1023.0 1388.0  1753.0 2118.0 2483.0 2848.0  3213.0     
```

1200kB means length of about 31s, which means I can probably discard everything after 1000kB and hardly change the dataset size and hopefully keep the pipeline from crashing.

So far the automatic checkpoint deletion works like a charm.