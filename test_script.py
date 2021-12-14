import pandas as pd
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import gc



def splitter(file, MIN=5, MAX=10):
    def is_ok(pauses, duration, MIN=10, MAX=20):
        pauses = [0, *pauses, duration]
        durations = [MIN * 1000 <= e - s <= MAX * 1000 for s, e in zip(pauses[:-1], pauses[1:])]
        return all(durations)

    def _splitter(pauses, duration, MIN=10, MAX=20):
        from itertools import combinations, chain
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        all_combinations = list(powerset(pauses))
        logging.info(f"Testing {len(all_combinations)} combinations....")
        for i, combination in enumerate(all_combinations):
            if i & 1024 == 0:
                gc.collect()
            if is_ok(combination, duration, MIN=MIN, MAX=MAX):
                return list(combination)
        logging.info(f"No solution found so that {MIN=}s <= duration <= {MAX=}s.")
        return None
    audio = AudioSegment.from_wav(file)
    duration = audio.duration_seconds * 1000
    logging.debug(f"Duration: {duration/1000} s")
    res = None
    for silence in [1000, 800, 500, 300, 200]:
        logging.info(f"Testing silence {silence}")
        detected_silences = detect_silence(audio, min_silence_len=silence, silence_thresh=-40) # In seconds
        if detected_silences == []:
            logging.info(f"No silences detected")
            continue
        nr_of_silences = len(detected_silences)
        logging.info(f"Got {nr_of_silences} silences.")
        centroids = np.array(detected_silences).mean(axis=1)
        centroids = centroids.tolist()

        res = _splitter(centroids, duration, MIN=MIN, MAX=MAX)
        if res != None:
            logging.info(f" Success! Found splitting: {res}")
            break
    if res == None:
        raise Exception("No splitting was found.")
    centroids = np.array(res).tolist()
    cuts = [0, *centroids, duration]

    return cuts
    
res = splitter('/home/peterr/macocu/task6_speech/data/00017378.flac.wav', MIN=5, MAX=10)