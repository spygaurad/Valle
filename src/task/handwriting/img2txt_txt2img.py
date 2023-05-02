import cv2
import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.nn import DataParallel

from src.encoding.PositionalEncoding import PositionalEncoding
from src.model.handwriting.image_encoder import ImageEncoder
from src.model.handwriting.image_decoder import ImageDecoder
from src.model.handwriting.text_encoder import TextEncoder
from src.model.handwriting.text_decoder import TextDecoder
from src.task.task import Task


class Img2Txt_Txt2Img_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Img2Txt_Txt2Img_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        char_embedding = nn.Embedding(char_model.n_chars, config.char_embedding_dim)
        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.text_encoder = DataParallel(TextEncoder(char_embedding, pos_encoding, config))
        self.text_decoder = DataParallel(TextDecoder(char_model.n_chars, char_embedding, pos_encoding, config, device))

        self.image_encoder = DataParallel(ImageEncoder(pos_encoding, config))
        self.image_decoder = DataParallel(ImageDecoder(config))

    def forward(self, batch_output):

        batch_output.device = self.device

        # Image to Text
        img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
        img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                            batch_output['img_txt_txt_tgt_in'],
                                            batch_output['img_txt_txt_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_mask'])

        # Text to Image
        txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                            batch_output['img_txt_txt_pad_mask'])
        txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

        return img2txt_dec_txt, txt2img_dec_img

    def evaluate(self, batch_output, type):

        batch_output.device = self.device

        if type == "img2txt":
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

        elif type == "txt2img":
            try:
                txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                    batch_output['img_txt_txt_pad_mask'])
                txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

                return txt2img_dec_img

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for txt2img task.")

    def generate(self, batch_output, type):

        batch_output.device = self.device

        if type == "img2txt":
            img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
            img2txt_dec_txt = self.text_decoder.module.generate(img2txt_enc_char,
                                                                None,
                                                                self.char_model.char2index["SOS"],
                                                                self.char_model.char2index["EOS"],
                                                                self.config.max_char_len)

            return img2txt_dec_txt

        elif type == "txt2img":
            txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                batch_output['img_txt_txt_pad_mask'])
            txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

            return txt2img_dec_img


class Img2Txt_Txt2Img(Task):

    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)

    def build_model(self):
        return Img2Txt_Txt2Img_Model(self.char_model, self.config, self.device)

    def loss_function(self):
        txt = nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['PAD'])
        img = nn.MSELoss()

        return {'txt': txt,
                'img': img}

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_model(self):

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            self.model.train()
            self.current_epoch = epoch

            total_img2txt_loss = []
            total_txt2img_loss = []

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    img2txt_dec_txt, txt2img_dec_img = self.model(batch_output)

                    img2txt_loss = self.criterion['txt'](img2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                         batch_output['img_txt_txt_tgt_out'].contiguous().view(-1))

                    txt2img_loss = self.criterion['img'](txt2img_dec_img, batch_output['img_txt_img'])

                    self.optimizer.zero_grad()
                    loss = img2txt_loss + txt2img_loss
                    loss.backward()

                    self.optimizer.step()

                    total_img2txt_loss.append(img2txt_loss.item())
                    total_txt2img_loss.append(txt2img_loss.item())

                    bar.update(index)

            total_loss = total_img2txt_loss + total_txt2img_loss

            print("train_img2txt_loss", np.average(total_img2txt_loss))
            print("train_txt2img_loss", np.average(total_txt2img_loss))
            print("train_total_loss", np.average(total_loss))

            if epoch % self.config.model_eval_epoch == 0:

                self.exp_track.log_metric("train_img2txt_loss", np.average(total_img2txt_loss))
                self.exp_track.log_metric("train_txt2img_loss", np.average(total_txt2img_loss))
                self.exp_track.log_metric("train_total_loss", np.average(total_loss))

                self.eval_model()
                self.save_model()
                self.gen_model()

    def eval_model(self):

        self.model.eval()

        total_img2txt_loss = []
        total_txt2img_loss = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):

                    try:
                        img2txt_dec_txt = self.model.evaluate(batch_output,  "img2txt")
                        img2txt_loss = self.criterion['txt'](img2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                             batch_output['img_txt_txt_tgt_out'].contiguous().view(-1))
                        total_img2txt_loss.append(img2txt_loss.item())
                    except RuntimeWarning:
                        pass

                    try:
                        txt2img_dec_img = self.model.evaluate(batch_output, "txt2img")
                        txt2img_loss = self.criterion['img'](txt2img_dec_img, batch_output['img_txt_img'])
                        total_txt2img_loss.append(txt2img_loss.item())
                    except RuntimeWarning:
                        pass

                    bar.update(index)

            total_loss = total_img2txt_loss + total_txt2img_loss

        self.exp_track.log_metric("val_img2txt_loss", np.average(total_img2txt_loss))
        self.exp_track.log_metric("val_txt2img_loss", np.average(total_txt2img_loss))
        self.exp_track.log_metric("val_total_loss", np.average(total_loss))

    def gen_model(self):

        def save_txt(file_path, data, txt_title):
            txt = "".join(data)
            f = open(file_path, "w")
            f.write("".join(txt))
            f.close()

            self.exp_track.log_text(txt_title, txt)

        def save_img(img_path, img, img_title):
            cv2.imwrite(img_path, img)

            self.exp_track.log_image(img_title, img.squeeze())

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():

            for index, batch_output in enumerate(self.gen_set, 1):

                if batch_output['label'] == 2:

                    output = self.model.generate(batch_output, "img2txt")

                    real_output = self.char_model.indexes2characters(
                        batch_output['img_txt_txt_tgt_out'].cpu().numpy()[0])
                    save_txt(f"{self.config.gen_epoch_path}/img2txt_in_{index}.txt", real_output, 'real_img2txt')

                    predicted_output = self.char_model.indexes2characters(output[1:])
                    save_txt(f"{self.config.gen_epoch_path}/img2txt_out_{index}.txt",
                             predicted_output, 'predicted_img2txt')

                    output = self.model.generate(batch_output, "txt2img")

                    save_img(f"{self.config.gen_epoch_path}/txt2img_in_{index}.png",
                             255 - batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy(), 'real_txt2img')

                    save_img(f"{self.config.gen_epoch_path}/txt2img_out_{index}.png",
                             255 - output.squeeze(0).squeeze(0).cpu().numpy(), 'predicted_txt2img')

    def test_model(self):
        pass
