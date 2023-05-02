import cv2
import numpy as np
import progressbar
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from transformers import ReformerModel
from tensorboardX import SummaryWriter

import torchvision

from src.encoding.PositionalEncoding import PositionalEncoding
from src.model.handwriting.image_encoder import ImageEncoder
from src.model.handwriting.image_decoder import ImageDecoder
from src.model.handwriting.text_encoder import TextEncoder
from src.model.handwriting.text_decoder import TextDecoder
from src.task.task import Task
from src.soft_dtw import SoftDTW


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class StylePredictor(nn.Module):
    def __init__(self, total_styles, device):
        super(StylePredictor, self).__init__()
        self.resnet_head = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(torchvision.models.resnet18(pretrained=True).children())[1:-1]))
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.resnet_head(x)
        x = x.squeeze()
        x = self.linear(x)
        return x


class Handwriting_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Handwriting_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        self.reformer_model = None 

        """
        self.reformer_model = ReformerModel.from_pretrained("google/reformer-enwik8")
        for param in self.reformer_model.parameters():
            param.requires_grad = False
        self.reformer_model = DataParallel(self.reformer_model)
        """



        # reformer_n_chars = 258
        reformer_n_chars = self.char_model.n_chars
        char_embedding = nn.Embedding(reformer_n_chars, config.char_embedding_dim)
        pos_encoding_char_dim = PositionalEncoding(config.char_embedding_dim)
        # pos_encoding = PositionalEncoding(2048)
        pos_encoding = PositionalEncoding(512)

        self.text_encoder = DataParallel(TextEncoder(char_embedding, pos_encoding, config))
        # self.text_encoder = DataParallel(TextEncoder(config))
        self.text_decoder = DataParallel(TextDecoder(reformer_n_chars, char_embedding, pos_encoding_char_dim, config, device))

        self.image_encoder = DataParallel(ImageEncoder(pos_encoding_char_dim, config))
        # self.image_decoder = DataParallel(ImageDecoder(config))
        self.image_decoder = DataParallel(ImageDecoder(char_model.n_chars, char_embedding, pos_encoding, config, device))

        self.style_predictor = DataParallel(StylePredictor(total_styles=10,device=device))

        print(f"Number of parameters in text encoder: {count_parameters(self.text_encoder)}")
        print(f"Number of parameters in text decoder: {count_parameters(self.text_decoder)}")
        print(f"Number of parameters in image encoder: {count_parameters(self.image_encoder)}")
        print(f"Number of parameters in image decoder: {count_parameters(self.image_decoder)}")
        print(f"Number of parameters in style predictor: {count_parameters(self.style_predictor)}")

    def forward(self, batch_output):

        batch_output.device = self.device

        # self.reformer_model = self.reformer_model.eval()
        # self.reformer_model = self.reformer_model.to(self.device)

        # Text to Text
        txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                            batch_output['txt_txt_pad_mask'], self.reformer_model)
        txt2txt_dec_txt = self.text_decoder(txt2txt_enc_txt,
                                            batch_output['txt_txt_tgt_in'],
                                            batch_output['txt_txt_pad_mask'],
                                            batch_output['txt_txt_tgt_in_pad_mask'],
                                            batch_output['txt_txt_tgt_in_mask'])
        # Image to Image
        img2img_enc_char = self.image_encoder(batch_output['img_img'])
        # p = np.random.random()
        # if np.random.random() < 0.1:
        #     img2img_enc_char = self.image_encoder(batch_output['img_img'])
        # else:
        #     img2img_enc_char = self.image_encoder(batch_output['img_img'] + (torch.randn_like(batch_output['img_img'].to(self.device)) + torch.randn_like(batch_output['img_img'].to(self.device)) + torch.randn_like(batch_output['img_img'].to(self.device))))
        img2img_dec_style_vec, img2img_dec_img = self.image_decoder(img2img_enc_char, batch_output['img_img_style_img'])
        img2img_dec_style_vec_back = self.image_decoder(style=img2img_dec_img, return_style_only=True)
        img2img_dec_style_prediction_real = self.style_predictor(batch_output['img_img'])
        img2img_dec_style_prediction_gen = self.style_predictor(img2img_dec_img)


        # Image to Text
        img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
        img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                            batch_output['img_txt_txt_tgt_in'],
                                            None,
                                            batch_output['img_txt_txt_tgt_in_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_mask'])

        # Text to Image
        txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                            batch_output['img_txt_txt_pad_mask'], self.reformer_model)
        txt2img_dec_style_vec, txt2img_dec_img = self.image_decoder(txt2img_enc_txt, batch_output['img_txt_style_img'])

        txt2img_dec_style_vec_back = self.image_decoder(style=txt2img_dec_img, return_style_only=True)
        txt2img_dec_style_prediction_real = self.style_predictor(batch_output['img_txt_img'])
        txt2img_dec_style_prediction_gen = self.style_predictor(txt2img_dec_img)

        # Image to Text backtranslated
        img2txt_enc_char_back = self.image_encoder(txt2img_dec_img)
        img2txt_dec_txt_back = self.text_decoder(img2txt_enc_char_back,
                                            batch_output['img_txt_txt_tgt_in'],
                                            None,
                                            batch_output['img_txt_txt_tgt_in_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_mask'])



        return (img2txt_enc_char,
                txt2img_enc_txt,
                txt2txt_dec_txt,
                img2img_dec_img,
                img2txt_dec_txt,
                txt2img_dec_img,
                img2txt_dec_txt_back,
                img2img_dec_style_vec,
                img2img_dec_style_vec_back,
                img2img_dec_style_prediction_real,
                img2img_dec_style_prediction_gen,
                txt2img_dec_style_vec,
                txt2img_dec_style_vec_back,
                txt2img_dec_style_prediction_real,
                txt2img_dec_style_prediction_gen
                )

    def evaluate(self, batch_output, type):

        batch_output.device = self.device

        if type == "txt2txt":
            try:
                txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                                    batch_output['txt_txt_pad_mask'], self.reformer_model)
                txt2txt_dec_txt = self.text_decoder(txt2txt_enc_txt,
                                                    batch_output['txt_txt_tgt_in'],
                                                    batch_output['txt_txt_pad_mask'],
                                                    batch_output['txt_txt_tgt_in_pad_mask'],
                                                    batch_output['txt_txt_tgt_in_mask'])
                return txt2txt_dec_txt

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for txt2txt task.")

        elif type == "img2img":
            try:
                img2img_enc_char = self.image_encoder(batch_output['img_img'])
                _, img2img_dec_img = self.image_decoder(img2img_enc_char, batch_output['img_img_style_img'])

                return img2img_dec_img

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for img2img task.")

        elif type == "img2txt":
            try:
                img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
                img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                                    batch_output['img_txt_txt_tgt_in'],
                                                    None,
                                                    batch_output['img_txt_txt_tgt_in_pad_mask'],
                                                    batch_output['img_txt_txt_tgt_in_mask'])
                return img2txt_dec_txt

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for img2txt task.")

        elif type == "txt2img":
            try:
                txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                    batch_output['img_txt_txt_pad_mask'], self.reformer_model)
                style, txt2img_dec_img = self.image_decoder(txt2img_enc_txt, batch_output['img_txt_style_img'])

                return txt2img_dec_img

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for txt2img task.")

    def generate(self, batch_output, type):

        batch_output.device = self.device

        if type == "txt2txt":
            txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                                batch_output['txt_txt_pad_mask'], self.reformer_model)
            txt2txt_dec_txt = self.text_decoder.module.generate(txt2txt_enc_txt,
                                                                batch_output['txt_txt_pad_mask'],
                                                                self.char_model.char2index["SOS"],
                                                                self.char_model.char2index["EOS"],
                                                                self.config.max_char_len)

            return txt2txt_dec_txt

        elif type == "img2img":
            img2img_enc_char = self.image_encoder(batch_output['img_img'])
            _, img2img_dec_img = self.image_decoder(img2img_enc_char, batch_output['img_img_style_img'])

            return img2img_dec_img

        elif type == "img2txt":
            img2txt_enc_char = self.image_encoder(batch_output['img_txt_img'])
            img2txt_dec_txt = self.text_decoder.module.generate(img2txt_enc_char,
                                                                None,
                                                                self.char_model.char2index["SOS"],
                                                                self.char_model.char2index["EOS"],
                                                                self.config.max_char_len)

            return img2txt_dec_txt

        elif type == "txt2img":
            txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                batch_output['img_txt_txt_pad_mask'], self.reformer_model)
            style, txt2img_dec_img = self.image_decoder(txt2img_enc_txt, batch_output['img_txt_style_img'])

            return txt2img_dec_img


class Handwriting(Task):

    def __init__(self, train_set, val_set, test_set, train_gen_set, val_gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, val_gen_set, char_model, config, device, exp_track)

        self.train_gen_set = train_gen_set
        self.val_gen_set = val_gen_set

        self.train_batch_size = self.config.batch_size / 3
        self.val_batch_size = self.config.batch_size
        self.writer = SummaryWriter(self.config.tensorboard_path)

        print(f"Train Batch Size: {self.train_batch_size}")
        print(f"Val Batch Size: {self.val_batch_size}")

        self.img_comparison_loss = 'mse'
        self.use_scheduler = False
        self.warmup_epochs = 0 
        self.output_dump_len = 10


    def build_model(self):
        return Handwriting_Model(self.char_model, self.config, self.device)

    def get_scheduler(self):
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 8000, eta_min=0.0001/2.0)
        return scheduler

    def get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision_training)
        # scaler = None
        return scaler

    def loss_function(self):
        txt = nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['PAD'])
        img = nn.MSELoss(reduction='sum')
        font = nn.CrossEntropyLoss()
        alignment = nn.CosineSimilarity(dim=2)
        style = nn.L1Loss(reduction='sum')
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)

        return {'txt': txt,
                'img': img,
                'style_cross': font,
                'alignment': alignment,
                'style_l1': style,
                'sdtw': sdtw}

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_model(self):


        print("Handwriting task")
        
        for epoch in range(self.current_epoch, self.config.epoch + 1):

            # torch.cuda.empty_cache()

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")
            for param_group in self.optimizer.param_groups:

                if epoch <= self.warmup_epochs:
                    param_group['lr'] = (epoch/self.warmup_epochs) * self.config.lr

                current_lr = param_group['lr']
                print(f"Current LR: {current_lr}")

            self.model.train()
            self.current_epoch = epoch

            total_txt2txt_loss = []
            total_img2img_loss = []
            total_img2txt_loss = []
            total_txt2img_loss = []
            total_img2txt_back_loss = []
            total_alignment_loss = []
            total_txt2img_style_vec_loss = []
            total_img2img_style_vec_loss = []
            total_txt2img_real_style_loss = []
            total_txt2img_generated_style_loss = []
            total_img2img_real_style_loss = []
            total_img2img_generated_style_loss = []


            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision_training):

                        img2txt_enc_char, txt2img_enc_txt, txt2txt_dec_txt, img2img_dec_img, img2txt_dec_txt, txt2img_dec_img, img2txt_dec_txt_back, img2img_dec_style_vec, img2img_dec_style_vec_back, img2img_dec_style_prediction_real, img2img_dec_style_prediction_gen, txt2img_dec_style_vec, txt2img_dec_style_vec_back, txt2img_dec_style_prediction_real, txt2img_dec_style_prediction_gen = self.model(batch_output)

                        txt2txt_loss = self.criterion['txt'](txt2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                             batch_output['txt_txt_tgt_out'].contiguous().view(-1))

                        if self.img_comparison_loss == 'mse':
                            img2img_loss = self.criterion['img'](img2img_dec_img, batch_output['img_img'])
                            img2img_loss /= self.train_batch_size

                            txt2img_loss = self.criterion['img'](txt2img_dec_img, batch_output['img_txt_img'])
                            txt2img_loss /= self.train_batch_size
                        elif self.img_comparison_loss == 'sdtw':
                            img2img_loss = self.criterion['sdtw'](img2img_dec_img.squeeze(1), batch_output['img_img'].squeeze(1))
                            img2img_loss = img2img_loss.mean()

                            txt2img_loss = self.criterion['sdtw'](txt2img_dec_img.squeeze(1), batch_output['img_txt_img'].squeeze(1))
                            txt2img_loss = txt2img_loss.mean()

                        img2txt_loss = self.criterion['txt'](img2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                             batch_output['img_txt_txt_tgt_out'].contiguous().view(-1))


                        img2txt_back_loss = self.criterion['txt'](img2txt_dec_txt_back.view(-1, self.char_model.n_chars),
                                                             batch_output['img_txt_txt_tgt_out'].contiguous().view(-1))


                        txt2img_dec_style_vec_back = txt2img_dec_style_vec_back.repeat(1, 4, 1)
                        img2img_dec_style_vec_back = img2img_dec_style_vec_back.repeat(1, 4, 1)

                        txt2img_style_vec_loss = self.criterion['style_l1'](txt2img_dec_style_vec_back, txt2img_dec_style_vec)
                        txt2img_style_vec_loss /= self.train_batch_size

                        img2img_style_vec_loss = self.criterion['style_l1'](img2img_dec_style_vec_back, img2img_dec_style_vec)
                        img2img_style_vec_loss /= self.train_batch_size

                        txt2img_real_style_loss = self.criterion['style_cross'](txt2img_dec_style_prediction_real, batch_output['img_txt_font'])
                        txt2img_generated_style_loss = self.criterion['style_cross'](txt2img_dec_style_prediction_gen, batch_output['img_txt_font'])

                        img2img_real_style_loss = self.criterion['style_cross'](img2img_dec_style_prediction_real, batch_output['img_font'])
                        img2img_generated_style_loss = self.criterion['style_cross'](img2img_dec_style_prediction_gen, batch_output['img_font'])

                        mask_label = torch.ones(128, requires_grad=False)

                        alignment_loss = -1.0 * torch.mean(self.criterion['alignment'](img2txt_enc_char, txt2img_enc_txt))


                        if (index+1)%2 == 0:
                            loss = 200 * txt2txt_loss + img2img_loss + 200 * img2txt_loss + txt2img_loss + img2txt_back_loss + 1000.0 * alignment_loss + txt2img_style_vec_loss + img2img_style_vec_loss + txt2img_real_style_loss + img2img_real_style_loss + txt2img_generated_style_loss + img2img_generated_style_loss
                        else:
                            loss = 200 * txt2txt_loss + img2img_loss + 200 * img2txt_loss + txt2img_loss + img2txt_back_loss + 1000.0 * alignment_loss + txt2img_style_vec_loss + img2img_style_vec_loss + txt2img_real_style_loss + img2img_real_style_loss

                    # loss.backward()
                    # self.optimizer.step()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()



                    total_txt2txt_loss.append(txt2txt_loss.item())
                    total_img2img_loss.append(img2img_loss.item())
                    total_img2txt_loss.append(img2txt_loss.item())
                    total_txt2img_loss.append(txt2img_loss.item())
                    total_alignment_loss.append(alignment_loss.item())
                    total_img2txt_back_loss.append(img2txt_back_loss.item())
                    total_txt2img_style_vec_loss.append(txt2img_style_vec_loss.item())
                    total_img2img_style_vec_loss.append(img2img_style_vec_loss.item())
                    total_img2img_real_style_loss.append(img2img_real_style_loss.item())
                    total_img2img_generated_style_loss.append(img2img_generated_style_loss.item())
                    total_txt2img_real_style_loss.append(txt2img_real_style_loss.item())
                    total_txt2img_generated_style_loss.append(txt2img_generated_style_loss.item())

                    bar.update(index)
                    # torch.cuda.empty_cache()

            if self.use_scheduler:
                self.scheduler.step()

            total_loss = total_img2img_loss + total_img2txt_loss + total_txt2img_loss + total_img2txt_back_loss + total_alignment_loss + total_img2img_style_vec_loss + total_txt2img_style_vec_loss + total_img2img_real_style_loss + total_img2img_generated_style_loss + total_txt2img_real_style_loss + total_txt2img_generated_style_loss

            train_txt2txt_loss = np.average(total_txt2txt_loss)
            train_txt2img_loss = np.average(total_txt2img_loss)
            train_img2txt_loss = np.average(total_img2txt_loss)
            train_img2img_loss = np.average(total_img2img_loss)
            train_alignment_loss = np.average(total_alignment_loss)
            train_img2txt_back_loss = np.average(total_img2txt_back_loss)
            train_txt2img_style_vec_loss = np.average(total_txt2img_style_vec_loss)
            train_img2img_style_vec_loss = np.average(total_img2img_style_vec_loss)
            train_txt2img_real_style_loss = np.average(total_txt2img_real_style_loss)
            train_txt2img_generated_style_loss = np.average(total_txt2img_generated_style_loss)
            train_img2img_real_style_loss = np.average(total_img2img_real_style_loss)
            train_img2img_generated_style_loss = np.average(total_img2img_generated_style_loss)
            train_loss = np.average(total_loss)
            
            print("train_txt2txt_loss", (train_txt2txt_loss))
            print("train_img2img_loss", (train_img2img_loss))
            print("train_img2txt_loss", (train_img2txt_loss))
            print("train_txt2img_loss", (train_txt2img_loss))
            print("train_alignment_loss", (train_alignment_loss))
            print("train_img2txt_back_loss", (train_img2txt_back_loss))
            print("train_txt2img_style_vec_loss", (train_txt2img_style_vec_loss))
            print("train_img2img_style_vec_loss", (train_img2img_style_vec_loss))
            print("train_txt2img_real_style_loss", (train_txt2img_real_style_loss))
            print("train_txt2img_generated_style_loss", (train_txt2img_generated_style_loss))
            print("train_img2img_real_style_loss", (train_img2img_real_style_loss))
            print("train_img2img_generated_style_loss", (train_img2img_generated_style_loss))
            print("train_total_loss", (train_loss))

            if np.isnan(train_loss):
                print("Traning halted due to nan loss")
                break
            
            self.writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)
            self.writer.add_scalars('Txt2Txt Loss', {'train_txt2txt_loss': train_txt2txt_loss}, epoch)
            self.writer.add_scalars('Txt2Img Loss', {'train_txt2img_loss': train_txt2img_loss}, epoch)
            self.writer.add_scalars('Img2Txt Loss', {'train_img2txt_loss': train_img2txt_loss}, epoch)
            self.writer.add_scalars('Img2Img Loss', {'train_img2img_loss': train_img2img_loss}, epoch)
            self.writer.add_scalars('Alignment Loss', {'train_alignment_loss': train_alignment_loss}, epoch)
            self.writer.add_scalars('Img2Txt Back Loss', {'train_img2txt_back_loss': train_img2txt_back_loss}, epoch)
            self.writer.add_scalars('Img2Img Style Vec Loss', {'train_img2img_style_vec_loss': train_img2img_style_vec_loss}, epoch)
            self.writer.add_scalars('Img2Txt Style Vec Loss', {'train_txt2img_style_vec_loss': train_txt2img_style_vec_loss}, epoch)
            self.writer.add_scalars('Img2Txt Real Style Loss', {'train_txt2img_real_style_loss': train_txt2img_real_style_loss}, epoch)
            self.writer.add_scalars('Img2Txt Gen Style Loss', {'train_txt2img_generated_style_loss': train_txt2img_generated_style_loss}, epoch)
            self.writer.add_scalars('Img2Img Real Style Loss', {'train_img2img_real_style_loss': train_img2img_real_style_loss}, epoch)
            self.writer.add_scalars('Img2Img Gen Style Loss', {'train_img2img_generated_style_loss': train_img2img_generated_style_loss}, epoch)

            if epoch % self.config.model_eval_epoch == 0:
                torch.cuda.empty_cache()

                if self.exp_track is not None:

                    self.exp_track.log_metric("train_img2img_loss", np.average(total_img2img_loss))
                    self.exp_track.log_metric("train_img2txt_loss", np.average(total_img2txt_loss))
                    self.exp_track.log_metric("train_txt2img_loss", np.average(total_txt2img_loss))
                    self.exp_track.log_metric("train_total_loss", np.average(total_loss))

                val_loss, val_txt2txt_loss, val_txt2img_loss, val_img2txt_loss, val_img2img_loss = self.eval_model()
                self.writer.add_scalars('Loss', {'val_loss': val_loss}, epoch)
                self.writer.add_scalars('Txt2Txt Loss', {'val_txt2txt_loss': val_txt2txt_loss}, epoch)
                self.writer.add_scalars('Txt2Img Loss', {'val_txt2img_loss': val_txt2img_loss}, epoch)
                self.writer.add_scalars('Img2Txt Loss', {'val_img2txt_loss': val_img2txt_loss}, epoch)
                self.writer.add_scalars('Img2Img Loss', {'val_img2img_loss': val_img2img_loss}, epoch)
                self.save_model()
                self.gen_model()

    def eval_model(self):

        self.model.eval()

        total_txt2txt_loss = []
        total_img2img_loss = []
        total_img2txt_loss = []
        total_txt2img_loss = []

        with torch.no_grad():

            print("\nEvaluating..............")
            with progressbar.ProgressBar(max_value=len(self.val_set)) as bar:

                for index, batch_output in enumerate(self.val_set):

                    try:
                        txt2txt_dec_txt = self.model.evaluate(batch_output, "txt2txt")
                        txt2txt_loss = self.criterion['txt'](txt2txt_dec_txt.view(-1, self.char_model.n_chars),
                                                             batch_output['txt_txt_tgt_out'].contiguous().view(-1))
                        total_txt2txt_loss.append(txt2txt_loss.item())
                    except RuntimeWarning:
                        pass

                    try:
                        img2img_dec_img = self.model.evaluate(batch_output, "img2img")

                        img2img_loss = self.criterion['img'](img2img_dec_img, batch_output['img_img'])
                        img2img_loss /= self.val_batch_size

                        total_img2img_loss.append(img2img_loss.item())

                    except RuntimeWarning:
                        pass

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
                        txt2img_loss /= self.val_batch_size
                        total_txt2img_loss.append(txt2img_loss.item())
                    except RuntimeWarning:
                        pass

                    bar.update(index)

            total_loss = total_txt2txt_loss + total_img2img_loss + total_img2txt_loss + total_txt2img_loss

        val_loss = np.average(total_loss)
        val_txt2txt_loss = np.average(total_txt2txt_loss)
        val_txt2img_loss = np.average(total_txt2img_loss)
        val_img2txt_loss = np.average(total_img2txt_loss)
        val_img2img_loss = np.average(total_img2img_loss)

        if self.exp_track is not None:
            self.exp_track.log_metric("val_img2img_loss", np.average(total_img2img_loss))
            self.exp_track.log_metric("val_img2txt_loss", np.average(total_img2txt_loss))
            self.exp_track.log_metric("val_txt2img_loss", np.average(total_txt2img_loss))
            self.exp_track.log_metric("val_total_loss", np.average(total_loss))

        print("val_txt2txt_loss", (val_txt2txt_loss))
        print("val_img2img_loss", (val_img2img_loss))
        print("val_img2txt_loss", (val_img2txt_loss))
        print("val_txt2img_loss", (val_txt2img_loss))
        print("val_total_loss", (val_loss))

        return val_loss, val_txt2txt_loss, val_txt2img_loss, val_img2txt_loss, val_img2img_loss


    def gen_model(self):

        def save_txt(file_path, data, txt_title):
            txt = "".join(data)
            f = open(file_path, "w")
            f.write("".join(txt))
            f.close()

            if self.exp_track is not None:
                self.exp_track.log_text(txt_title, txt)

        def save_img(img_path, img, img_title):
            cv2.imwrite(img_path, img)

            if self.exp_track is not None:

                self.exp_track.log_image(img_title, img.squeeze())

        def gen_output(gen_set, mode="test"):

            with torch.no_grad():

                counter = {"label_1":0, "label_2":0, "label_0":0}
                test_dump_len = self.output_dump_len

                for index, batch_output in enumerate(gen_set, 1):

                    if batch_output['label'] == 0 and counter["label_0"] < test_dump_len:

                        output = self.model.generate(batch_output, "txt2txt")

                        real_output = self.char_model.indexes2characters(batch_output['txt_txt_tgt_out'].cpu().numpy()[0])
                        save_txt(f"{self.config.gen_epoch_path}/{mode}_txt2txt_in_{index}.txt", real_output, 'real_txt2txt')

                        predicted_output = self.char_model.indexes2characters(output[1:])
                        save_txt(f"{self.config.gen_epoch_path}/{mode}_txt2txt_out_{index}.txt",
                                 predicted_output, 'predicted_txt2txt')
                        counter["label_0"] += 1

                    if batch_output['label'] == 1 and counter["label_1"] < test_dump_len:

                        output = self.model.generate(batch_output, "img2img")

                        save_img(f"{self.config.gen_epoch_path}/{mode}_img2img_in_{index}.png",
                                 255 * batch_output['img_img'].squeeze(0).squeeze(0).cpu().numpy(), 'real_img2img')

                        save_img(f"{self.config.gen_epoch_path}/{mode}_img2img_out_{index}.png",
                                 255 * output.squeeze(0).squeeze(0).cpu().numpy(), 'predicted_img2img')

                        counter["label_1"] += 1

                    elif batch_output['label'] == 2 and counter["label_2"] < test_dump_len:

                        output = self.model.generate(batch_output, "img2txt")

                        real_output = self.char_model.indexes2characters(
                            batch_output['img_txt_txt_tgt_out'].cpu().numpy()[0])
                        save_txt(f"{self.config.gen_epoch_path}/{mode}_img2txt_in_{index}.txt", real_output, 'real_img2txt')

                        predicted_output = self.char_model.indexes2characters(output[1:])
                        save_txt(f"{self.config.gen_epoch_path}/{mode}_img2txt_out_{index}.txt",
                                 predicted_output, 'predicted_img2txt')

                        output = self.model.generate(batch_output, "txt2img")

                        save_img(f"{self.config.gen_epoch_path}/{mode}_txt2img_in_{index}.png",
                                 255 * batch_output['img_txt_img'].squeeze(0).squeeze(0).cpu().numpy(), 'real_txt2img')

                        save_img(f"{self.config.gen_epoch_path}/{mode}_txt2img_out_{index}.png",
                                 255 * output.squeeze(0).squeeze(0).cpu().numpy(), 'predicted_txt2img')
                        counter["label_2"] += 1

                    if counter["label_0"] + counter["label_1"] + counter["label_2"] > 3 * test_dump_len:
                        break


        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        gen_output(self.train_gen_set, mode="train")
        gen_output(self.val_gen_set, mode="val")
        gen_output(self.test_set, mode="test")



    def test_model(self):
        pass
