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
from src.task.task import Task


class Txt2Img_Img2Img_ImgTxt_Enc_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Txt2Img_Img2Img_ImgTxt_Enc_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        char_embedding = nn.Embedding(char_model.n_chars, config.char_embedding_dim)
        pos_encoding = PositionalEncoding(config.char_embedding_dim)

        self.text_encoder = DataParallel(TextEncoder(char_embedding, pos_encoding, config))
        self.image_encoder = DataParallel(ImageEncoder(pos_encoding, config))
        self.image_decoder = DataParallel(ImageDecoder(config))

    def forward(self, batch_output):

        batch_output.device = self.device

        # Image to Image
        img2img_enc_txt = self.image_encoder(batch_output['img_img'])
        img2img_dec_img = self.image_decoder(img2img_enc_txt)

        # Text to Image
        txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                            batch_output['img_txt_txt_pad_mask'])
        txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

        # Image Text Encoder
        img_enc_txt = self.image_encoder(batch_output['img_txt_img'])

        txt_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                        batch_output['img_txt_txt_pad_mask'])

        return img2img_dec_img,  txt2img_dec_img, img_enc_txt, txt_enc_txt

    def evaluate(self, batch_output, type):

        batch_output.device = self.device

        if type == "img2img":
            try:
                img2img_enc_txt = self.image_encoder(batch_output['img_img'])
                img2img_dec_img = self.image_decoder(img2img_enc_txt)

                return img2img_dec_img

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for img2img task.")

        elif type == "txt2img":
            try:
                txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                    batch_output['img_txt_txt_pad_mask'])
                txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

                return txt2img_dec_img

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for txt2img task.")

        elif type == "imgtxt_enc":
            try:
                img_enc_txt = self.image_encoder(batch_output['img_txt_img'])
                txt_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                batch_output['img_txt_txt_pad_mask'])

                return img_enc_txt, txt_enc_txt

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for imgtxt_enc task.")

    def generate(self, batch_output, type):

        batch_output.device = self.device

        if type == "img2img":
            img2img_enc_char = self.image_encoder(batch_output['img_img'])
            img2img_dec_img = self.image_decoder(img2img_enc_char)

            return img2img_dec_img

        elif type == "txt2img":
            txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                batch_output['img_txt_txt_pad_mask'])
            txt2img_dec_img = self.image_decoder(txt2img_enc_txt)

            return txt2img_dec_img


class Txt2Img_Img2Img_ImgTxt_Enc(Task):

    def __init__(self, train_set, val_set, test_set, gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, gen_set, char_model, config, device, exp_track)

    def build_model(self):
        return Txt2Img_Img2Img_ImgTxt_Enc_Model(self.char_model, self.config, self.device)

    def loss_function(self):
        return nn.MSELoss()

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_model(self):

        for epoch in range(self.current_epoch, self.config.epoch + 1):

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")

            self.model.train()
            self.current_epoch = epoch

            total_loss = []
            total_img2img_loss = []
            total_txt2img_loss = []
            total_imgtxt_enc_loss = []
            total_imgtxt_enc_sim = []

            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    img2img_dec_img, txt2img_dec_img, img_enc_txt, txt_enc_txt = self.model(batch_output)

                    img2img_loss = self.criterion(img2img_dec_img, batch_output['img_img'])
                    txt2img_loss = self.criterion(txt2img_dec_img, batch_output['img_txt_img'])
                    imgtxt_enc_loss = self.criterion(img_enc_txt, txt_enc_txt)

                    loss = img2img_loss + txt2img_loss + imgtxt_enc_loss

                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()

                    imgtxt_enc_sim = torch.nn.CosineSimilarity()(img_enc_txt, txt_enc_txt)

                    total_img2img_loss.append(img2img_loss.item())
                    total_txt2img_loss.append(txt2img_loss.item())
                    total_imgtxt_enc_loss.append(imgtxt_enc_loss.item())
                    total_imgtxt_enc_sim.append(torch.mean(imgtxt_enc_sim).item())

                    bar.update(index)

            total_loss = total_img2img_loss + total_txt2img_loss + total_imgtxt_enc_loss

            print("train_img2img_loss", np.average(total_img2img_loss))
            print("train_txt2img_loss", np.average(total_txt2img_loss))
            print("train_imgtxt_enc_loss", np.average(total_imgtxt_enc_loss))
            print("train_total_loss", np.average(total_loss))
            print("total_imgtxt_enc_sim", np.average(total_imgtxt_enc_sim))

            if epoch % self.config.model_eval_epoch == 0:

                self.exp_track.log_metric("train_img2img_loss", np.average(total_img2img_loss))
                self.exp_track.log_metric("train_txt2img_loss", np.average(total_txt2img_loss))
                self.exp_track.log_metric("train_imgtxt_enc_loss", np.average(total_imgtxt_enc_loss))
                self.exp_track.log_metric("train_total_loss", np.average(total_loss))
                self.exp_track.log_metric("total_imgtxt_enc_sim", np.average(total_imgtxt_enc_sim))

                self.eval_model()
                self.save_model()
                self.gen_model()

    def eval_model(self):

        self.model.eval()

        total_loss = []
        total_img2img_loss = []
        total_txt2img_loss = []
        total_imgtxt_enc_loss = []
        total_imgtxt_enc_sim = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):

                    try:
                        img2img_dec_img = self.model.evaluate(batch_output, "img2img")
                        img2img_loss = self.criterion(img2img_dec_img, batch_output['img_img'])
                        total_img2img_loss.append(img2img_loss.item())
                    except RuntimeWarning:
                        pass

                    try:
                        txt2img_dec_img = self.model.evaluate(batch_output, "txt2img")
                        txt2img_loss = self.criterion(txt2img_dec_img, batch_output['img_txt_img'])
                        total_txt2img_loss.append(txt2img_loss.item())
                    except RuntimeWarning:
                        pass

                    try:
                        img_enc_txt, txt_enc_txt = self.model.evaluate(batch_output, "imgtxt_enc")
                        imgtxt_enc_loss = self.criterion(img_enc_txt, txt_enc_txt)
                        imgtxt_enc_sim = torch.nn.CosineSimilarity()(img_enc_txt, txt_enc_txt)
                        total_imgtxt_enc_loss.append(imgtxt_enc_loss.item())
                        total_imgtxt_enc_sim.append(torch.mean(imgtxt_enc_sim).item())

                    except RuntimeWarning:
                        pass

                    bar.update(index)

            total_loss = total_img2img_loss + total_txt2img_loss + total_imgtxt_enc_loss

        self.exp_track.log_metric("val_img2img_loss", np.average(total_img2img_loss))
        self.exp_track.log_metric("val_txt2img_loss", np.average(total_txt2img_loss))
        self.exp_track.log_metric("val_imgtxt_enc_loss", np.average(total_imgtxt_enc_loss))
        self.exp_track.log_metric("val_total_loss", np.average(total_loss))
        self.exp_track.log_metric("val_imgtxt_enc_sim", np.average(total_imgtxt_enc_sim))

    def gen_model(self):

        def save_img(img_path, img, img_title):
            cv2.imwrite(img_path, img)

            self.exp_track.log_image(img_title, img.squeeze())

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():

            for index, batch_output in enumerate(self.gen_set, 1):

                if batch_output['label'] == 1:

                    output = self.model.generate(batch_output, "img2img")

                    save_img(f"{self.config.gen_epoch_path}/img2img_in_{index}.png",
                             255 - batch_output['img_img'].squeeze(0).squeeze(0).cpu().numpy(), 'real_img2img')

                    save_img(f"{self.config.gen_epoch_path}/img2img_out_{index}.png",
                             255 - output.squeeze(0).squeeze(0).cpu().numpy(), 'predicted_img2img')

                elif batch_output['label'] == 2:

                    output = self.model.generate(batch_output, "txt2img")

                    save_img(f"{self.config.gen_epoch_path}/txt2img_in_{index}.png",
                             255 - batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy(), 'real_txt2img')

                    save_img(f"{self.config.gen_epoch_path}/txt2img_out_{index}.png",
                             255 - output.squeeze(0).squeeze(0).cpu().numpy(), 'predicted_txt2img')

    def test_model(self):
        pass
