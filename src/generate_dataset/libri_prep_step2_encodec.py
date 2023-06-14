import os
import pandas as pd
import torch
import multiprocessing as mp
import torchaudio
from generate_dataset.encodec_wrapper import EncodecWrapper

codec = EncodecWrapper()

def worker_function(item, out_path):
    """
    do some work, write results to output
    """
    save_path = item.replace('.flac', '.pt')
    waveform, sample_rate = torchaudio.load(item)
    _, indices, _ = codec(waveform, return_encoded = True)
    torch.save(indices, out_path)


if __name__ == '__main__':

    manifest_audio_path = "../../audio_dataset/LibriSpeech"
    manifests = []
    out_encodec_path = "../../audio_dataset/Encodec_LibriSpeech"

    pool = mp.Pool(16)
    jobs = []

    for manifest in manifests:
        data = pd.read_csv()

        # data = pd.read_csv('output_list.txt', sep=" ", header=None)
        # data.columns = ["a", "b", "c", "etc."]

        for item in range(10000):
            job = pool.apply_async(worker_function, (item,))
            jobs.append(job)

        for job in jobs:
            job.get()

    pool.close()
    pool.join()