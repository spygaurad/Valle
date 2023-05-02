import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.nn import DataParallel
from transformers import ReformerModel

from src.encoding.PositionalEncoding import PositionalEncoding
from src.model.handwriting.text_encoder import TextEncoder
from src.model.handwriting.text_decoder import TextDecoder
from src.task.task import Task


class Txt2Txt_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Txt2Txt_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        self.reformer_model = ReformerModel.from_pretrained("google/reformer-enwik8")
        for param in self.reformer_model.parameters():
            param.requires_grad = False

        # char_embedding = nn.Embedding(char_model.n_chars, config.char_embedding_dim)
        char_embedding = nn.Embedding(338, config.char_embedding_dim)
        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.text_encoder = DataParallel(TextEncoder(config))
        self.text_decoder = DataParallel(TextDecoder(338, char_embedding, pos_encoding, config, device))

    def forward(self, batch_output):

        batch_output.device = self.device
        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        # Text to Text
        txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                            batch_output['txt_txt_pad_mask'],
                                            self.reformer_model)
        txt2txt_dec_txt = self.text_decoder(txt2txt_enc_txt,
                                            batch_output['txt_txt_tgt_in'],
                                            batch_output['txt_txt_pad_mask'],
                                            batch_output['txt_txt_tgt_in_pad_mask'],
                                            batch_output['txt_txt_tgt_in_mask'])
        return txt2txt_dec_txt

    def evaluate(self, batch_output):

        batch_output.device = self.device
        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                            batch_output['txt_txt_pad_mask'],
                                            self.reformer_model)
        txt2txt_dec_txt = self.text_decoder(txt2txt_enc_txt,
                                            batch_output['txt_txt_tgt_in'],
                                            batch_output['txt_txt_pad_mask'],
                                            batch_output['txt_txt_tgt_in_pad_mask'],
                                            batch_output['txt_txt_tgt_in_mask'])
        return txt2txt_dec_txt

    def generate(self, batch_output):

        batch_output.device = self.device
        self.reformer_model = self.reformer_model.eval()
        self.reformer_model = self.reformer_model.to(self.device)

        txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                            batch_output['txt_txt_pad_mask'],
                                            self.reformer_model)
        txt2txt_dec_txt = self.text_decoder.module.generate(txt2txt_enc_txt,
                                                            batch_output['txt_txt_pad_mask'],
                                                            self.char_model.char2index["SOS"],
                                                            self.char_model.char2index["EOS"],
                                                            self.config.max_char_len)

        return txt2txt_dec_txt


class Txt2Txt(Task):
    def __init__(self, train_set, val_set, test_set, gen_set, char_model,  config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)

    def build_model(self):
        return Txt2Txt_Model(self.char_model, self.config, self.device)

    def loss_function(self):
        return nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['PAD'])

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def get_scaler(self):
        pass

    def get_scheduler(self):
        pass

    def train_model(self):

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            self.model.train()
            self.current_epoch = epoch

            total_loss = []

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    txt2txt_dec_txt = self.model(batch_output)
                    loss = self.criterion(txt2txt_dec_txt.view(-1, self.char_model.n_chars),
                                          batch_output['txt_txt_tgt_out'].contiguous().view(-1))

                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()

                    total_loss.append(loss.item())

                    bar.update(index)

            print("Train Loss: ", np.average(total_loss))
            if epoch % self.config.model_eval_epoch == 0:

                self.exp_track["train_total_loss"].log(np.average(total_loss))
                self.eval_model()
                self.save_model()
                self.gen_model()

    def eval_model(self):

        self.model.eval()

        total_loss = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):
                    txt2txt_dec_txt = self.model.evaluate(batch_output)
                    txt2txt_loss = self.criterion(txt2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                  batch_output['txt_txt_tgt_out'].contiguous().view(-1))
                    total_loss.append(txt2txt_loss.item())

                    bar.update(index)

        self.exp_track["val_total_loss"].log(np.average(total_loss))
        print("Val Loss: ", np.average(total_loss))

    def gen_model(self):

        def save_txt(file_path, data, txt_title):
            txt = "".join(data)
            f = open(file_path, "w")
            f.write("".join(txt))
            f.close()

            self.exp_track[txt_title].log(txt)

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():

            for index, batch_output in enumerate(self.gen_set, 1):

                if batch_output['label'] == 0:

                    output = self.model.generate(batch_output)

                    real_output = self.char_model.indexes2characters(batch_output['txt_txt_tgt_out'].cpu().numpy()[0])
                    save_txt(f"{self.config.gen_epoch_path}/txt2txt_in_{index}.txt", real_output, 'real_txt2txt')

                    predicted_output = self.char_model.indexes2characters(output[1:])
                    save_txt(f"{self.config.gen_epoch_path}/txt2txt_out_{index}.txt",
                             predicted_output, 'predicted_txt2txt')

    def test_model(self):
        pass
