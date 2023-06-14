import os
import pandas as pd
import torch
import multiprocessing as mp
import torchaudio
from encodec_wrapper import EncodecWrapper

codec = EncodecWrapper()

def worker_function(item, out_path):
    """
    do some work, write results to output
    """
    waveform, sample_rate = torchaudio.load(item)
    _, indices, _ = codec(waveform, return_encoded = True)
    torch.save(indices, out_path)

if __name__ == '__main__':

    manifest_audio_path = "../../audio_dataset/LibriSpeech/"
    manifests = []
    out_encodec_path = "../../audio_dataset/Encodec_LibriSpeech/"

    pool = mp.Pool(16)
    jobs = []

    for manifest in manifests:
        file = pd.read_csv(manifest, sep='\t')
        file.columns = ['file_path','transcript', 'transcript_formatted']
        file['out_path'] = file['file_path']
        file['out_path'] = file['out_path'].str.replace('flac','pt')
        file['out_path'] = out_encodec_path + file['out_path']
        file['file_path'] = manifest_audio_path + file['file_path']
        # data = pd.read_csv('output_list.txt', sep=" ", header=None)
        # data.columns = ["a", "b", "c", "etc."]
        for index, row in file.iterrows():
            job = pool.apply_async(worker_function, (row['file_path'],row['out_path'],))
            jobs.append(job)

        for job in jobs:
            job.get()

    pool.close()
    pool.join()