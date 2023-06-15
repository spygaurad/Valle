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
    audio_length =  waveform.shape[1] / sample_rate
    if audio_length > 14.0:
        return False
    else:
        _, indices, _ = codec(waveform, return_encoded = True)
    # print(len(indices[0]))
    # if len(indices[0]) > 
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok = True)
        torch.save(indices, out_path)
        return True


if __name__ == '__main__':

    manifest_audio_path = "/home/wiseyak/suraj/everything_text_valle/Valle/audio_dataset/LibriSpeech/"
    manifests = ['dev-clean']
    out_encodec_path = "/home/wiseyak/suraj/everything_text_valle/Valle/audio_dataset/Encodec_LibriSpeech/"

    pool = mp.Pool(16)
    jobs = []

    for manifest in manifests:
        file = pd.read_csv('/home/wiseyak/suraj/everything_text_valle/Valle/audio_dataset/' + manifest + '-transcript.txt', sep='\t')
        file.columns = ['file_path','transcript', 'transcript_formatted']
        file['out_path'] = file['file_path']
        file['out_path'] = file['out_path'].str.replace('flac','pt')
        file['out_path'] = out_encodec_path + file['out_path']
        file['file_path'] = manifest_audio_path + file['file_path']
        # data = pd.read_csv('output_list.txt', sep=" ", header=None)
        # data.columns = ["a", "b", "c", "etc."]
        encodec_df = file.copy
        for index, row in file.iterrows():
            encodec_applied = worker_function(row['file_path'],row['out_path'])
            if not encodec_applied:
                encodec_df.drop(index, inplace=True)
        encodec_df.to_csv('/home/wiseyak/suraj/everything_text_valle/Valle/audio_dataset/' + manifest + '-encodec-transcript.txt', index=False)
