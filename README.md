# TASK 6: speech to text experiments

An [example](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) was reconstructed as an excercize. Some observations:
* A lot of workarounds were implemented to make the model work and not all of the aspects are as of yet clear to me
* Training time for 30 epochs was about 2h.

Discuss which evaluation metric to use (WER? PER?)

# Addendum 2021-12-03T08:18:03

The virtual machine does not have the necessary tools installed to convert the flac files to wav. I therefore migrated all the files to my computer and converted them there. The disk only has 14G of free space, meaning that in case the wav dataset is bigger than that, I will need to delete the original flacs.

I noticed that I also had the chance to downsample the original 44100 Hz sampling rate to the recommended 16000 Hz. In this conversion filesize dropped to about 33% of the original size, meaning that we will probably be ok regarding disk space on the virtual machine.

For posterity: file inspection was done with `soxi file.wav`. Conversion was performed using `ffmpeg -i 00023289.flac -ar 16000 brisi.wav`