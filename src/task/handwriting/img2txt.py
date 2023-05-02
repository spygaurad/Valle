import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.nn import DataParallel

from src.encoding.PositionalEncoding import PositionalEncoding
from src.model.handwriting.image_encoder import ImageEncoder
from src.model.handwriting.text_decoder import TextDecoder
from src.task.task import Task


class Img2Txt_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Img2Txt_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        char_embedding = nn.Embedding(char_model.n_chars, config.char_embedding_dim)
        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.text_decoder = DataParallel(TextDecoder(char_model.n_chars, char_embedding, pos_encoding, config, device))

        self.image_encoder = DataParallel(ImageEncoder(pos_encoding, config))

    def forward(self, batch_output):

        batch_output.device = self.device

        # Image to Text
        img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
        img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                            batch_output['img_txt_txt_tgt_in'],
                                            batch_output['img_txt_txt_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_mask'])

        return img2txt_dec_txt

    def evaluate(self, batch_output):

        batch_output.device = self.device

        try:
            img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
            img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                                batch_output['img_txt_txt_tgt_in'],
                                                batch_output['img_txt_txt_pad_mask'],
                                                batch_output['img_txt_txt_tgt_in_pad_mask'],
                                                batch_output['img_txt_txt_tgt_in_mask'])
            return img2txt_dec_txt

        except RuntimeError:
            raise RuntimeWarning("Data is unavailable for img2txt task.")

    def generate(self, batch_output):

        batch_output.device = self.device

        img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
        img2txt_dec_txt = self.text_decoder.module.generate(img2txt_enc_char,
                                                            None,
                                                            self.char_model.char2index["SOS"],
                                                            self.char_model.char2index["EOS"],
                                                            self.config.max_char_len)

        return img2txt_dec_txt


class Img2Txt(Task):

    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)

    def build_model(self):
        return Img2Txt_Model(self.char_model,  self.config, self.device)

    def loss_function(self):
        return nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['PAD'])

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_model(self):

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            self.model.train()
            self.current_epoch = epoch

            total_loss = []

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    img2txt_dec_txt = self.model(batch_output)

                    img2txt_loss = self.criterion(img2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                  batch_output['img_txt_txt_tgt_out'].contiguous().view(-1))

                    self.optimizer.zero_grad()
                    loss = img2txt_loss
                    loss.backward()

                    self.optimizer.step()

                    total_loss.append(loss.item())

                    bar.update(index)

            print("train_total_loss", np.average(total_loss))

            if epoch % self.config.model_eval_epoch == 0:

                for i in range(self.config.eval_gen_data_length):
                    print("----------------------------------------------")
                    print("For index: ", i)

                    expected_output = self.char_model.indexes2characters(
                        batch_output['img_txt_txt_tgt_out'].cpu().numpy()[i])
                    print("Expected output: ", "".join(expected_output))

                    predicted_output = self.model.text_decoder.module.softmax(img2txt_dec_txt)
                    _, predicted_output = predicted_output.max(dim=-1)
                    predicted_output = self.char_model.indexes2characters(predicted_output.cpu().numpy()[i])
                    print("Predicted output: ", "".join(predicted_output))
                    print("----------------------------------------------")

                self.exp_track.log_metric("train_total_loss", np.average(total_loss))

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

                    try:
                        img2txt_dec_txt = self.model.evaluate(batch_output)
                        img2txt_loss = self.criterion(img2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                      batch_output['img_txt_txt_tgt_out'].contiguous().view(-1))
                        total_loss.append(img2txt_loss.item())
                    except RuntimeWarning:
                        pass

                    bar.update(index)

        self.exp_track.log_metric("val_total_loss", np.average(total_loss))

    def gen_model(self):

        def save_txt(file_path, data, txt_title):
            txt = "".join(data)
            f = open(file_path, "w")
            f.write("".join(txt))
            f.close()

            self.exp_track.log_text(txt_title, txt)

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():

            for index, batch_output in enumerate(self.gen_set, 1):

                if batch_output['label'] == 2:

                    output = self.model.generate(batch_output)

                    real_output = self.char_model.indexes2characters(
                        batch_output['img_txt_txt_tgt_out'].cpu().numpy()[0])
                    save_txt(f"{self.config.gen_epoch_path}/img2txt_in_{index}.txt", real_output, 'real_img2txt')

                    predicted_output = self.char_model.indexes2characters(output[1:])
                    save_txt(f"{self.config.gen_epoch_path}/img2txt_out_{index}.txt",
                             predicted_output, 'predicted_img2txt')

    def test_model(self):
        pass
