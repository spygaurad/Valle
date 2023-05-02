import cv2
import numpy as np
import pandas as pd
import progressbar
import re
import time
import gc
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from transformers import ReformerModel
from tensorboardX import SummaryWriter

import torch.distributed as dist

import torchvision
import editdistance

from torchmetrics.functional import char_error_rate, word_error_rate


import src.vqvae_codebook as vqvae_codebook

from src.encoding.PositionalEncoding import PositionalEncoding, LearnablePositionalEncoding
from src.model.handwriting.image_encoder import ImageEncoder, StackMixImageEncoder
from src.model.handwriting.image_decoder import ImageDecoder
from src.model.handwriting.text_encoder import TextEncoder
from src.model.handwriting.text_decoder import TextDecoder
from src.task.task import Task
from src.soft_dtw import SoftDTW
# cv2.set_num_threads(0)
from itertools import chain

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

        print(self.optimizers)

    def __getitem__(self, key):
        return self.optimizers[key]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dict):
        for index, op in enumerate(self.optimizers):
            op.load_state_dict(state_dict[index])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class StylePredictor(nn.Module):
    def __init__(self, total_styles, device):
        super(StylePredictor, self).__init__()
        # """
        self.resnet_head = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(torchvision.models.resnet18(pretrained=True).children())[1:-1]))
        self.linear = nn.Linear(512, 10)
        """
        self.resnet_head = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(torchvision.models.resnet18(pretrained=True).children())[1:6]))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(128, 10)
        # """

    def forward(self, x):
        x = self.resnet_head(x)
        x = x.squeeze()
        # x = self.avg_pool(x).squeeze(2).squeeze(2)
        x = self.linear(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """
        self.resnet_head = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(torchvision.models.resnet18(pretrained=True).children())[1:-1]))
        self.linear = nn.Linear(512, 10)
        """
        self.resnet_head = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(torchvision.models.resnet18(pretrained=True).children())[1:6]))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        # """

    def forward(self, x):
        x = self.resnet_head(x)
        # x = x.squeeze()
        x = self.avg_pool(x).squeeze(2).squeeze(2)
        x = self.sigmoid(self.linear(x))
        return x

class ImgCharModel:
    def __init__(self):
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 1027

        self.char2index['ISOS'] = 1024
        self.char2index['IEOS'] = 1025
        
        self.index2char[1024] = 'ISOS'
        self.index2char[1025] = 'IEOS'

class StyleCharModel:
    def __init__(self):
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 1026

        self.char2index['ISOS'] = 1024
        self.char2index['IEOS'] = 1025
        
        self.index2char[1024] = 'ISOS'
        self.index2char[1025] = 'IEOS'


class Handwriting_Model(nn.Module):
    def __init__(self, char_model, config, device):
        super(Handwriting_Model, self).__init__()

        self.char_model = char_model
        self.config = config
        self.device = device

        self.random_masking = self.config.random_masking
        self.distillation = self.config.distillation
        self.aux_ctc = self.config.aux_ctc
        print("Random Masking:", self.random_masking)
        print("Distillation:", self.distillation)
        print("Aux CTC:", self.aux_ctc)

        self.reformer_model = None 
        # self.

        """
        self.reformer_model = ReformerModel.from_pretrained("google/reformer-enwik8")
        for param in self.reformer_model.parameters():
            param.requires_grad = False
        self.reformer_model = DataParallel(self.reformer_model)
        """



        # reformer_n_chars = 258
        reformer_n_chars = self.char_model.n_chars
        #out_vocab_size = 514
        out_vocab_size = 1027


        char_embedding = nn.Embedding(reformer_n_chars, config.char_embedding_dim)
        img_embedding  = nn.Embedding(out_vocab_size, config.char_embedding_dim)
        self.img_embedding = img_embedding
        # pos_encoding_char_dim = LearnablePositionalEncoding(config.char_embedding_dim)
        # pos_encoding = PositionalEncoding(2048)
        text_encoder_pos_encoding = PositionalEncoding(config.char_embedding_dim)
        text_pos_encoding = PositionalEncoding(config.char_embedding_dim)
        audio_pos_encoding = PositionalEncoding(config.char_embedding_dim)

        img_char_model = ImgCharModel()
        img_sos_token = img_char_model.char2index['ISOS']
        self.img_sos_token = torch.LongTensor([[img_sos_token]])
        img_eos_token = img_char_model.char2index['IEOS']
        self.img_eos_token = torch.LongTensor([[img_eos_token]])

        style_char_model = None
        style_sos_token = None
        self.style_sos_token = None
        style_eos_token = None
        self.style_eos_token = None


        
       
        # self.img_init_embedding = nn.Embedding(img_char_model.n_chars, config.char_embedding_dim)
        self.style_init_embedding = None

        self.text_encoder = TextEncoder(char_embedding, text_encoder_pos_encoding, config)
        # self.text_encoder = DataParallel(TextEncoder(config))

        # self.image_decoder = DataParallel(ImageDecoder(config))
        # self.image_decoder = DataParallel(ImageDecoder(char_model.n_chars, char_embedding, pos_encoding, config, device))

        # self.style_predictor = DataParallel(StylePredictor(total_styles=10,device=device))


        #For OCR loss
        # self.image_encoder = ImageEncoder(pos_encoding_char_dim, config, device, out_vocab_size)
        # self.image_encoder = TextEncoder(img_embedding, pos_encoding, config)
        # self.codebook = vqvae_codebook.get_model()
        # checkpoint = torch.load("latest_model.pth", map_location=self.device)
        # self.codebook.load_state_dict(checkpoint["model"])

        # for params in self.codebook.parameters():
        #     params.requires_grad = False
        # self.image_encoder = DataParallel(StackMixImageEncoder(pos_encoding_char_dim, config, device, out_vocab_size))
        self.text_decoder = TextDecoder(out_vocab_size, char_embedding, img_embedding, text_pos_encoding,audio_pos_encoding, config, device)


        # print(f"Number of parameters in text encoder: {count_parameters(self.text_encoder)}")
        # print(f"Number of parameters in image decoder: {count_parameters(self.image_decoder)}")
        # print(f"Number of parameters in style predictor: {count_parameters(self.style_predictor)}")
        print(f"Number of parameters in image encoder: {count_parameters(self.text_encoder)}")
        print(f"Number of parameters in text decoder: {count_parameters(self.text_decoder)}")

    def forward(self, batch_output):
        # print("Device:", self.device)

        # batch_output.device = f"cuda:{self.device}"
        batch_output.device = self.device


        # Text to Image
        """
        txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                            batch_output['img_txt_txt_pad_mask'], self.reformer_model)
        txt2img_dec_style_vec, txt2img_dec_img = self.image_decoder(txt2img_enc_txt, batch_output['img_txt_style_img'])

        txt2img_dec_style_vec_back = self.image_decoder(style=txt2img_dec_img, return_style_only=True)
        txt2img_dec_style_prediction_real = self.style_predictor(batch_output['img_txt_img'])
        txt2img_dec_style_prediction_gen = self.style_predictor(txt2img_dec_img)
        """

        # Image to Text backtranslated
        """
        img2txt_enc_char_back = self.image_encoder(txt2img_dec_img)
        img2txt_dec_txt_back = self.text_decoder(img2txt_enc_char_back,
                                            batch_output['img_txt_txt_tgt_in'],
                                            None,
                                            batch_output['img_txt_txt_tgt_in_pad_mask'],
                                            batch_output['img_txt_txt_tgt_in_mask'])
        """

        txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt_tgt'],
                                            batch_output['img_txt_txt_tgt_pad_mask'], self.reformer_model)

        #Image to Text
        # img2txt_enc_char, aux_features = self.image_encoder(batch_output['img_txt_img'], aux_ctc=self.aux_ctc)
        #img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
        #                                    batch_output['img_txt_txt_tgt_in'],
        #                                    None,
        #                                    batch_output['img_txt_txt_tgt_in_pad_mask'],
        #                                    batch_output['img_txt_txt_tgt_in_mask'])
        # output_codes, output_embedding= self.codebook.get_encodings(batch_output['img_txt_img'])
        #  = [...,...,1]

        output_codes = batch_output['audio'].long()
        # print('*'*10,output_codes.shape)
        # output_codes = output_codes[:,:,:1].squeeze(2)
        bs, seq_len = output_codes.shape
        # print('*'*100)
        # print(bs,seq_len)
        # print('@'*100)
        # output_embedding = output_embedding.view(bs, seq_len, -1)
        img2txt_enc_char = None

        # img_sos_token_idx = self.img_sos_token.to(self.device).repeat(bs,1)
        # img_eos_token_idx = self.img_eos_token.to(self.device).repeat(bs,1)

        # print(img_sos_token_idx.shape)
        # print(output_codes.shape)
        # output_codes = torch.cat([img_sos_token_idx, output_codes, img_eos_token_idx], axis=1)
        # img_sos_token_emb = self.img_init_embedding(img_sos_token_idx)
        # img_eos_token_emb = self.img_init_embedding(img_eos_token_idx)
        # img_sos_token_emb = self.img_embedding(img_sos_token_idx)
        # img_eos_token_emb = self.img_embedding(img_eos_token_idx)
        # output_embedding = torch.cat([img_sos_token_emb, output_embedding, img_eos_token_emb], axis=1)
        # output_embedding_in = output_embedding[:,:,:-1]
        # output_embededing_tgt = output_embedding[:,:,1:]

        # we donot want to predict the PAD after EOS token
        output_codes_in = output_codes[:, :-1]

        # codebook for the audio. This is to just verify whether original codebook is frozen or changes in training. It should be frozen.
        output_codes_tgt = output_codes[:, 1:]

        # print("Target Codes:", output_codes.shape)
        # print("Target Embedding:", output_embedding.shape)

        style_codes, style_embedding= None,None
        style_embedding = None
        style_sos_token_idx = None
        style_eos_token_idx = None
        style_codes = None
        style_sos_token_emb = None
        style_eos_token_emb = None
        style_embedding = None
        # print("Style Embedding:", style_embedding.shape)
        # print("Style Codes:", style_codes.shape)




        img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                            batch_output['img_txt_txt_tgt_in'],
                                            batch_output['src_mask'],
                                            torch.cat((batch_output['img_txt_txt_tgt_pad_mask'], batch_output['audio_pad_mask']), axis = -1),
                                            random_masking=self.random_masking,
                                            distillation=self.distillation,
                                            aux_ctc=self.aux_ctc,
                                            output_codec=output_codes_in,
                                            style_codec=style_codes,
                                            text_encodec = txt2img_enc_txt)



        return (
                img2txt_dec_txt,
                output_codes_tgt,
                style_codes
                )

    def evaluate(self, batch_output, type):

        batch_output.device = self.device
        # batch_output.device = f"cuda:{self.device}"

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
                # img2img_enc_char = self.image_encoder(batch_output['img_img'])
                _, img2img_dec_img = self.image_decoder(img2img_enc_char, batch_output['img_img_style_img'])

                return img2img_dec_img

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for img2img task.")

        elif type == "img2txt":
            try:
                txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt_tgt'],
                                                    batch_output['img_txt_txt_tgt_pad_mask'], self.reformer_model)
                # img2txt_enc_char, aux_features = self.image_encoder(batch_output['img_txt_img'], aux_ctc=self.aux_ctc)
                img2txt_enc_char = None
                # output_codes, output_embedding= self.codebook.get_encodings(batch_output['img_txt_img'])
                output_codes = batch_output['audio'].long()
                bs, seq_len = output_codes.shape
                # output_embedding = output_embedding.view(bs, seq_len, -1)
                '''
                img_sos_token_idx = self.img_sos_token.to(self.device).repeat(bs,1)
                img_eos_token_idx = self.img_eos_token.to(self.device).repeat(bs,1)
                output_codes = torch.cat([img_sos_token_idx, output_codes, img_eos_token_idx], axis=1)
                '''
                # img_sos_token_emb = self.img_init_embedding(img_sos_token_idx)
                # img_eos_token_emb = self.img_init_embedding(img_eos_token_idx)
                '''
                img_sos_token_emb = self.img_embedding(img_sos_token_idx)
                img_eos_token_emb = self.img_embedding(img_eos_token_idx)
                '''
                # output_embedding = torch.cat([img_sos_token_emb, output_embedding, img_eos_token_emb], axis=1)
                # output_embedding_in = output_embedding[:,:,:-1]
                # output_embededing_tgt = output_embedding[:,:,1:]
                output_codes_in = output_codes[:, :-1]
                output_codes_tgt = output_codes[:, 1:]

                style_codes, style_embedding= None, None
                style_embedding = None
                style_sos_token_idx = None
                style_eos_token_idx = None
                style_codes = None
                style_sos_token_emb = None
                style_eos_token_emb = None
                style_embedding = None

                # img2txt_enc_char, aux_features = self.image_encoder(batch_output['img_txt_img'], aux_ctc=self.aux_ctc)
                img2txt_enc_char = None
                img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
                                                    batch_output['img_txt_txt_tgt_in'],
                                                    batch_output['src_mask'],
                                                    torch.cat((batch_output['img_txt_txt_tgt_pad_mask'], batch_output['audio_pad_mask']), axis = -1),
                                                    random_masking=self.random_masking,
                                                    distillation=self.distillation,
                                                    aux_ctc=self.aux_ctc,
                                                    output_codec=output_codes_in,
                                                    style_codec=style_codes,
                                                    text_encodec = txt2img_enc_txt)




                return img2txt_dec_txt, output_codes_tgt, style_codes

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for img2txt task.")

        elif type == "txt2img":
            try:
                txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                    batch_output['img_txt_txt_pad_mask'], self.reformer_model)
                style, txt2img_dec_img = self.image_decoder(txt2img_enc_txt, batch_output['img_txt_style_img'])

                # img2txt_enc_char_back = self.image_encoder(batch_output['img_txt_img'])
                img2txt_enc_char =None
                img2txt_dec_txt_back, _ = self.text_decoder(img2txt_enc_char_back,
                                                    batch_output['img_txt_txt_tgt_in'],
                                                    None,
                                                    batch_output['img_txt_txt_tgt_in_pad_mask'],
                                                    batch_output['img_txt_txt_tgt_in_mask'])

                return txt2img_dec_img, img2txt_dec_txt_back

            except RuntimeError:
                raise RuntimeWarning("Data is unavailable for txt2img task.")
                
    def beam_search(self, batch_output, type, beam_size=1, gt=None, lm_model=None, **kwargs):
        if type == "img2txt":
            beam_penalty = kwargs.pop("beam_penalty", 0.5)
            lm_weight = kwargs.pop("lm_weight", 0.1)
            char_model = kwargs.pop("char_model", None)
            # img2txt_enc_char, _ = self.image_encoder(batch_output['img_txt_img'])
            # img2txt_dec_txt = self.text_decoder.module.beam_search(img2txt_enc_char,
            #                                                     self.char_model.char2index["TSOS"],
            #                                                     self.char_model.char2index["TEOS"],
            #                                                     self.config.max_char_len,
            #                                                     beam_size=beam_size,
            #                                                     gt=gt)
            img2txt_dec_txt = self.text_decoder.module.beam_search_batch(img2txt_enc_char,
                                                                self.char_model.char2index["TSOS"],
                                                                self.char_model.char2index["TEOS"],
                                                                self.config.max_char_len,
                                                                beam_size=beam_size,
                                                                gt=gt,
                                                                lm_model=lm_model,
                                                                beam_penalty=beam_penalty,
                                                                lm_weight=lm_weight,
                                                                char_model=char_model,
                                                                tgt=batch_output['img_txt_txt_tgt_out'])
            return img2txt_dec_txt



    def generate(self, batch_output, type):

        batch_output.device = self.device
        # batch_output.device = f"cuda:{self.device}"

        if type == "txt2txt":
            txt2txt_enc_txt = self.text_encoder(batch_output['txt_txt'],
                                                batch_output['txt_txt_pad_mask'], self.reformer_model)
            txt2txt_dec_txt = self.text_decoder.module.generate(txt2txt_enc_txt,
                                                                batch_output['txt_txt_pad_mask'],
                                                                self.char_model.char2index["TSOS"],
                                                                self.char_model.char2index["TEOS"],
                                                                self.config.max_char_len)

            return txt2txt_dec_txt

        elif type == "img2img":
            # img2img_enc_char = self.image_encoder(batch_output['img_img'])
            _, img2img_dec_img = self.image_decoder(img2img_enc_char, batch_output['img_img_style_img'])

            return img2img_dec_img

        elif type == "img2txt":
            # img2txt_enc_char, _ = self.image_encoder(batch_output['img_txt_img'])
            # img2txt_dec_txt = self.text_decoder.module.generate(img2txt_enc_char,
            #                                                     self.char_model.char2index["TSOS"],
            #                                                     self.char_model.char2index["TEOS"],
            #                                                     self.config.max_char_len)
            # img2txt_dec_txt = self.text_decoder.generate_batch(img2txt_enc_char,
            #                                                     self.char_model.char2index["TSOS"],
            #                                                     self.char_model.char2index["TEOS"],
            #                                                     self.config.max_char_len)

            txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt_tgt'],
                                                batch_output['img_txt_txt_tgt_pad_mask'], self.reformer_model)
            # img2txt_enc_char, aux_features = self.image_encoder(batch_output['img_txt_img'], aux_ctc=self.aux_ctc)

            # output_codes, output_embedding= self.codebook.get_encodings(batch_output['img_txt_img'])
            # print("Inside generate:")
            # print(output_codes.shape)
            # print(output_embedding.shape)
            # bs, seq_len = output_codes.shape
            # output_embedding = output_embedding.view(bs, seq_len, -1)
            output_codes = batch_output['audio'].long()

            '''
            img_sos_token_idx = self.img_sos_token.to(self.device).repeat(bs,1)
            img_eos_token_idx = self.img_eos_token.to(self.device).repeat(bs,1)
            output_codes = torch.cat([img_sos_token_idx, output_codes, img_eos_token_idx], axis=1)
            # img_sos_token_emb = self.img_init_embedding(img_sos_token_idx)
            # img_eos_token_emb = self.img_init_embedding(img_eos_token_idx)
            img_sos_token_emb = self.img_embedding(img_sos_token_idx)
            img_eos_token_emb = self.img_embedding(img_eos_token_idx)
            # output_embedding = torch.cat([img_sos_token_emb, output_embedding, img_eos_token_emb], axis=1)
            # output_embedding_in = output_embedding[:,:,:-1]
            # output_embededing_tgt = output_embedding[:,:,1:]

            '''
            output_codes_in = output_codes[:, :-1]
            output_codes_tgt = output_codes[:, 1:]
            # print("**********************")
            # print(output_codes.shape)
            # print(output_embedding.shape)


            style_codes, style_embedding= None, None
            style_embedding = None
            style_sos_token_idx = None
            style_eos_token_idx = None
            style_codes = None
            style_sos_token_emb = None
            style_eos_token_emb = None
            style_embedding = None

            # img2txt_enc_char, aux_features = self.image_encoder(batch_output['img_txt_img'], aux_ctc=self.aux_ctc)
            img2txt_enc_char = None
            # img2txt_dec_txt_teacher_forcing = self.text_decoder(img2txt_enc_char,
            #                                     batch_output['img_txt_txt_tgt_in'],
            #                                     batch_output['src_mask'],
            #                                     batch_output['img_txt_txt_tgt_in_pad_mask'],
            #                                     random_masking=self.random_masking,
            #                                     distillation=self.distillation,
            #                                     aux_ctc=self.aux_ctc,
            #                                     output_codec=output_codes_in,
            #                                     style_codec=style_codes,
            #                                     text_encodec = txt2img_enc_txt)
            # img2txt_dec_txt_teacher_forcing = nn.Softmax(dim=-1)(img2txt_dec_txt_teacher_forcing)
            # values, indices = img2txt_dec_txt_teacher_forcing.max(dim=-1)
            # img2txt_dec_txt_teacher_forcing = indices.unsqueeze(-1)



            # img2txt_dec_txt_single = self.text_decoder.generate(img2txt_enc_char,
            #                                                     512,
            #                                                     513,
            #                                                     512,
            #                                                     style_codec=style_codes,
            #                                                     text_encodec = txt2img_enc_txt)
            # print("####################################################################")
            # print("Teacher Forcing:")
            # print(img2txt_dec_txt_teacher_forcing.squeeze().cpu().numpy().tolist())
            # print(len(img2txt_dec_txt_teacher_forcing.squeeze().cpu().numpy().tolist()))
            # print("********************************************************************")

            # print("Greedy decoding single:")
            # print(img2txt_dec_txt_single)
            # print(len(img2txt_dec_txt_single))
            # print("********************************************************************")
            # batch_output['img_txt_txt_tgt'],
            #                                             batch_output['img_txt_txt_tgt_pad_mask']

            img2txt_dec_txt = self.text_decoder.generate_batch(img2txt_enc_char,
                                                                # torch.cat((batch_output['img_txt_txt_tgt_pad_mask'], batch_output['audio_pad_mask']), axis = -1),
                                                                batch_output['img_txt_txt_tgt_pad_mask'],
                                                                1025,
                                                                1026,
                                                                256,
                                                                style_codec=style_codes,
                                                                text_encodec = txt2img_enc_txt)


            # print("Greedy decoding batch:")
            # print(img2txt_dec_txt.squeeze().cpu().numpy().tolist())
            # print(len(img2txt_dec_txt.squeeze().cpu().numpy().tolist()))
            # print(img2txt_dec_txt_single == img2txt_dec_txt.squeeze().cpu().numpy().tolist())
            # print("********************************************************************")
            # print("Target:")
            # print(output_codes_tgt.squeeze().cpu().numpy().tolist())
            # print("********************************************************************")
            # print("Target Input:")
            # print(output_codes_in.squeeze().cpu().numpy().tolist())
            # print("********************************************************************")
            # print("Whole output:")
            # print(output_codes.squeeze().cpu().numpy().tolist())
            # assert img2txt_dec_txt_single == img2txt_dec_txt.squeeze().cpu().numpy().tolist()
            # print("####################################################################")


            # img2txt_dec_txt = self.text_decoder(img2txt_enc_char,
            #                                     batch_output['img_txt_txt_tgt_in'],
            #                                     batch_output['src_mask'],
            #                                     batch_output['img_txt_txt_tgt_in_pad_mask'],
            #                                     random_masking=self.random_masking,
            #                                     distillation=self.distillation,
            #                                     aux_ctc=self.aux_ctc,
            #                                     output_codec=output_codes_in,
            #                                     style_codec=style_codes,
            #                                     text_encodec = txt2img_enc_txt)

            # print("************************")
            # print(output_codes.shape)
            # print(output_embedding.shape)
            # print("************************")
            # print("Inside generate:")


            return img2txt_dec_txt, output_codes, style_codes

        elif type == "txt2img":
            txt2img_enc_txt = self.text_encoder(batch_output['img_txt_txt'],
                                                batch_output['img_txt_txt_pad_mask'], self.reformer_model)
            style, txt2img_dec_img = self.image_decoder(txt2img_enc_txt, batch_output['img_txt_style_img'])

            return txt2img_dec_img

def get_ctc_clip_loss(char_model, preds, target, input_len, target_len):
    # prepare data to compute loss matrix using ctc loss
    """
    idea is to compare ctc loss by comparing
    i1, i1, i1, i2, i2, i2, i3, i3, i3 with t1, t2, t3, t1, t2, t3, t1, t2, t3
    losses would be: ctc1_1, ctc1_2, ctc1_3, ctc2_1, ctc2_2, ctc2_3, ctc3_1, ctc3_2, ctc3_3
    reshape ctc losses as
    ctc1_1, ctc1_2, ctc1_3
    ctc2_1, ctc2_2, ctc2_3
    ctc3_1, ctc3_2, ctc3_3
    use CLIP like loss using this matrix
    """

    from torch.nn import CTCLoss
    import torch.nn.functional as F


    batch_size = preds.shape[1]
    preds = preds.repeat_interleave(batch_size, dim=1)
    input_len = torch.tensor(input_len).repeat_interleave(batch_size)

    target = target.repeat(batch_size,1)
    target_len = torch.tensor(target_len).repeat(batch_size)

    loss_ctc = CTCLoss(blank=char_model.vocab_size, reduction="none", zero_infinity=True)

    loss_all = loss_ctc(preds, target, input_len, target_len)

    logits = -1 * loss_all.view(batch_size, batch_size)# * np.exp(0.8)
    labels = torch.arange(batch_size).cuda()
    
    loss1 = F.cross_entropy(F.normalize(logits), labels)
    loss2 = F.cross_entropy(F.normalize(logits.T), labels)
    loss = (loss1 + loss2)/2

    return loss

class Handwriting(Task):

    def __init__(self, train_set, val_set, test_set, train_gen_set, val_gen_set, test_gen_set, char_model, config, device, exp_track):
        super().__init__(train_set, val_set, test_set, val_gen_set, char_model, config, device, exp_track)

        self.train_gen_set = train_gen_set
        self.val_gen_set = val_gen_set
        self.test_gen_set = test_gen_set

        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.writer = SummaryWriter(self.config.tensorboard_path)

        print("Txt2Img With Style")

        print(f"Train Batch Size: {self.train_batch_size}")
        print(f"Val Batch Size: {self.val_batch_size}")

        self.clip_ctc_loss = False
        print(f"Clip CTC Loss: {self.clip_ctc_loss}")
        self.img_comparison_loss = 'mse'
        self.gan_mode = False
        self.use_scheduler = True
        self.warmup_epochs = 0
        self.output_dump_len = 20
        self.prev_val_loss = np.inf

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def build_model(self):
        return Handwriting_Model(self.char_model, self.config, self.device)
        
        # pytorch2 compiled
        # return torch.compile(Handwriting_Model(self.char_model, self.config, self.device))


    def get_scheduler(self):
        # scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 500, eta_min=0.0005/10.0)
        # scheduler = None
        # print(f"Length train set: {len(self.train_set)}")
        return None
        scheduler_c = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer[0],
            epochs=self.config.epoch,
            steps_per_epoch=len(self.train_set),
            max_lr = 0.001,
            pct_start = 0.1,
            anneal_strategy = 'cos',
            final_div_factor = 10**5
        )

        scheduler_t = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer[1],
            epochs=self.config.epoch,
            steps_per_epoch=len(self.train_set),
            max_lr = 0.001/2,
            pct_start = 0.1,
            anneal_strategy = 'cos',
            final_div_factor = 10**5
        )

        return scheduler_c, scheduler_t

    def get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision_training)
        # scaler = None
        return scaler

    def loss_function(self):
        txt = nn.CrossEntropyLoss(ignore_index=self.char_model.char2index['IPAD'], label_smoothing=0.1)
        # txt = nn.CrossEntropyLoss(reduction='sum')
        # txt = nn.CrossEntropyLoss()
        # txt = nn.CrossEntropyLoss()
        img = nn.MSELoss(reduction='sum')
        font = nn.CrossEntropyLoss()
        alignment = nn.CosineSimilarity(dim=2)
        style = nn.L1Loss(reduction='sum')
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        gan = nn.BCELoss()
        ctc = nn.CTCLoss(blank=self.char_model.vocab_size, reduction="sum")#zero_infinity=True)

        return {'txt': txt,
                'img': img,
                'style_cross': font,
                'alignment': alignment,
                'style_l1': style,
                'sdtw': sdtw,
                'gan': gan,
                'ctc': ctc}

    def get_optimizer(self):
        # return torch.optim.Adam(self.model.parameters(), lr=self.config.lr, amsgrad=True)
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-2)
        # return torch.optim.AdamW(self.model.text_decoder.parameters(), lr=self.config.lr, weight_decay=0.1)
        # ctc_optimizer = torch.optim.AdamW(self.model.image_encoder.parameters(), lr=self.config.lr*2, weight_decay=0.1)
        # ce_optimizer = torch.optim.AdamW(self.model.text_decoder.parameters(), lr=self.config.lr, weight_decay=0.1)
        # ctc_optimizer = torch.optim.AdamW(self.model.image_encoder.parameters(), lr=0.0002, weight_decay=1e-2)
        # # ce_optimizer = torch.optim.AdamW(self.model.text_decoder.parameters(), lr=0.0002, weight_decay=1e-2)
        # return MultipleOptimizer(ctc_optimizer, ce_optimizer)

        # return torch.optim.AdamW([
        #     {'params': self.model.text_encoder.parameters(),'lr': self.config.lr},
        #     {'params': self.model.text_decoder.parameters(), 'lr': self.config.lr}
        #     ], lr=self.config.lr * 10e-5, weight_decay=0.1)

    def calculate_wer(self, target, predicted,audio_path, raw_prob=False, return_raw=False, ctc_mode=False):
        wer = []
        raw_texts = []

        with torch.no_grad():
            if raw_prob:
                predicted = torch.argmax(predicted, dim=-1)
            bs = target.shape[0]
            token = 1026
            for i in range(bs):
                str_target =  target[i].cpu().numpy().tolist()
                a_path =  audio_path[i]

                # print(str_target)

                if token in str_target:
                    str_target_first_pad_index = str_target.index(token)
                else:
                    str_target_first_pad_index = len(str_target)

                str_target = (str_target[:str_target_first_pad_index])

                str_predicted = predicted[i].cpu().numpy().tolist()
                # print(str_predicted)
                if token in str_predicted:
                    str_predicted_first_pad_index = str_predicted.index(token)
                else:
                    str_predicted_first_pad_index = len(str_predicted)
                str_predicted = (str_predicted[:str_predicted_first_pad_index])

                raw_texts.append((str_target, str_predicted, a_path))
                 
                wer.append(editdistance.eval(str_target, str_predicted)/(len(str_target)))

            # wer = [editdistance.eval(target[i].cpu().numpy().tolist(),predicted[i].cpu().numpy().tolist()) for i in range(bs)]

            if return_raw:
                return raw_texts

            non_zeros = np.count_nonzero(wer)
            total = len(wer)
            acc = (total - non_zeros)/total
            # print("Accuracy:", acc)

            wer = np.average(wer)
            return wer, acc

    def train_model(self):
        device_count = torch.cuda.device_count()
        print("Let's use", device_count, "GPUs!\n")
        print('GPU id:', self.device)
        # self.model = self.model.to(self.device)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)
        # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])

        # for p in self.model.image_encoder.parameters():
        #     p.requires_grad = False

        # model_D = DataParallel(Discriminator())
        # model_D.to(self.device)
        # optimizer_D = torch.optim.AdamW(model_D.parameters(), lr=self.config.lr)

        # return torch.compile(Handwriting_Model(self.char_model, self.config, self.device))
        # self.model = torch.compile(self.model)

        print("Handwriting task")

        """
        self.eval_model()
        print('\n')

        for i in range(0, 10):
            print(i)
            self.eval_model(True, i)
            print('\n')
        # """

        # self.assert_greedy_beam_decode(self.val_gen_set)
        # self.assert_greedy_beam_decode(self.test_set)

        # self.eval_model(save_model=False, dataset=self.train_set)
        print("*****************************************************************")
        # self.compute_greedy_wer2(self.test_set, "test", search_type="beam", beam_size=10)
        # self.gen_model()
        # self.compute_greedy_wer2(self.test_set, "test", search_type="beam", beam_size=1)
        # self.compute_greedy_wer2(self.test_set, "test", search_type="beam", beam_size=16)
        # self.test_model()
        # self.compute_greedy_wer2(self.test_set, "test")
        print("*****************************************************************")
        # self.eval_model(save_model=False, dataset=self.test_set)

        print("****************************************************")
        # self.eval_model()
        # self.gen_model()
        # self.gen_model()
        # self.gen_model()
        # import sys
        # sys.exit()
        # self.test_model(mode="val")
        # self.test_model(mode="test")
        print("****************************************************")

        # print("Completed...")
        # import sys
        # sys.exit(0)
        
        # print(self.model.image_encoder)
        # print(self.model.text_decoder)
        # print(self.optimizer)
        # sys.exit(0)
        # self.test_model()
        print("*****************************************************************")
        # self.eval_model(save_model=False)
        # self.compute_greedy_wer2(self.val_set, "val")
        print("*****************************************************************")
        # self.test_model()
        # self.gen_model()
        
        for epoch in range(self.current_epoch, self.config.epoch + 1):

            # torch.cuda.empty_cache()

            print("\nTraining..............")
            print(f"Epoch: {epoch} ")
            # for param_group in self.optimizer[0].param_groups:

            #     #if epoch <= self.warmup_epochs:
            #     #    param_group['lr'] = (epoch/self.warmup_epochs) * self.config.lr

            #     # param_group['lr'] = self.config.lr/10

            #     current_lr = param_group['lr']
            #     # print(f"Param Group: {param_group}")
            #     print(f"Current CNN LR: {current_lr}")

            # for param_group in self.optimizer[1].param_groups:
            #     current_lr = param_group['lr']
            #     print(f"Current Transformer LR: {current_lr}")

            self.model.train()
            self.current_epoch = epoch

            total_txt2img_loss = []
            total_txt2img_style_vec_loss = []
            total_txt2img_real_style_loss = []
            total_txt2img_generated_style_loss = []
            total_discriminator_loss = []
            total_img2txt_loss = []
            total_img2txt_back_loss = []
            train_wer = []
            distillation_loss = []
            train_acc = []
            train_ctc_loss = []
            train_ctc_cer = []
            train_ctc_acc = []
            train_raw_texts = []
            train_encoder_raw_texts = []
            print("Mixed precision:", self.use_mixed_precision_training)

            # self.train_set.sampler.set_epoch(epoch)

            num_acc_steps = 64
            self.optimizer.zero_grad()



            with progressbar.ProgressBar(max_value=len(self.train_set)) as bar:

                for index, batch_output in enumerate(self.train_set):


                    # self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision_training):

                        img2txt_dec_txt, output_codes, style_codes = self.model(batch_output)
                        batch_output.device = self.device
                        # print(img2txt_dec_txt.shape)
                        # print(output_codes.shape)
                        img2txt_loss = self.criterion['txt'](img2txt_dec_txt.view(-1, 1027), output_codes.contiguous().view(-1).to(self.device))
                        loss = img2txt_loss/num_acc_steps


                    raw_text = self.calculate_wer(output_codes, img2txt_dec_txt,batch_output['audio_path'], raw_prob=True, return_raw=True)
                    train_raw_texts.extend(raw_text)

                    # loss.backward()
                    # self.optimizer.step()

                    self.scaler.scale(loss).backward()
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()

                    if (index+1)%num_acc_steps == 0 or (index+1)%len(self.train_set) == 0:
                    # self.scaler.step(self.optimizer[0])
                    # self.scaler.step(self.optimizer[1])
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    total_img2txt_loss.append(img2txt_loss.item())

                    bar.update(index)

                    # if self.use_scheduler:
                    #     self.scheduler[0].step()
                    #     self.scheduler[1].step()

            train_df = pd.DataFrame(train_raw_texts, columns=["target", "predicted","audio_path"])
            train_df["edit_distance"] = train_df.apply(lambda row: editdistance.eval(row["target"], row["predicted"]), axis=1)
            train_cer = train_df["edit_distance"].sum()/(train_df.shape[0] * 1024)
            # train_cer = char_error_rate(preds=train_df.predicted.values.tolist(), target=train_df.target.values.tolist())
            # train_wer = word_error_rate(preds=train_df.predicted.values.tolist(), target=train_df.target.values.tolist())
            train_acc = (train_df["target"] == train_df["predicted"]).sum()/train_df.shape[0]

            total_loss = total_img2txt_loss

            train_loss = np.average(total_loss)
            train_cer = np.average(train_cer)
            # train_wer = np.average(train_wer)
            train_acc = np.average(train_acc)
            train_img2txt_loss = np.average(total_img2txt_loss)
            
            if distillation_loss:
                distillation_loss = np.average(distillation_loss)
                print("train distillation_loss", (distillation_loss))
                self.writer.add_scalars('Distillation Loss', {'train_distillation_loss': distillation_loss}, epoch)
            if train_ctc_loss:
                train_ctc_loss = np.average(train_ctc_loss)
                print("train ctc_loss", (train_ctc_loss))
                self.writer.add_scalars('CTC Loss', {'train_ctc_loss': train_ctc_loss}, epoch)


            print("train_img2txt_loss", (train_img2txt_loss))
            # print("train_img2txt_back_loss", (train_img2txt_back_loss))
            print("train_total_loss", (train_loss))
            print("train_cer", (train_cer))
            # print("train_wer", (train_wer))
            print("train_acc", (train_acc))

            if np.isnan(train_loss):
                print("Traning halted due to nan loss")
                break
            
            self.writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)
            self.writer.add_scalars('Img2Txt Loss', {'train_img2txt_loss': train_img2txt_loss}, epoch)
            self.writer.add_scalars('Img2Txt CER', {'train_img2txt_cer': train_cer}, epoch)

            # dist.barrier()
            if epoch % self.config.model_eval_epoch == 0:# and self.device == 0:
                # self.gen_model()
                # import sys
                # sys.exit()
                # gc.collect()
                # torch.cuda.empty_cache()

                self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
                self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

                train_df.to_csv(f"{self.config.gen_epoch_path}/train.csv", index=False)

                if self.exp_track is not None:

                    self.exp_track.log_metric("train_img2img_loss", np.average(total_img2img_loss))
                    self.exp_track.log_metric("train_img2txt_loss", np.average(total_img2txt_loss))
                    self.exp_track.log_metric("train_txt2img_loss", np.average(total_txt2img_loss))
                    self.exp_track.log_metric("train_total_loss", np.average(total_loss))

                val_loss, val_cer, greedy_val_cer = self.test_model(mode="val")
                self.writer.add_scalars('Loss', {'val_loss': val_loss}, epoch)
                # self.writer.add_scalars('Txt2Img Loss', {'val_txt2img_loss': val_txt2img_loss}, epoch)
                # self.writer.add_scalars('Img2Txt Loss', {'val_img2txt_loss': val_img2txt_loss}, epoch)
                self.writer.add_scalars('Img2Txt CER', {'val_cer': val_cer}, epoch)
                self.writer.add_scalars('Greedy CER', {'val_greedy_cer': greedy_val_cer}, epoch)
                # self.writer.add_scalars('Img2Txt Back Loss', {'val_img2txt_back_loss': val_img2txt_back_loss}, epoch)

                # if self.device == 0:
                self.save_model()

                if greedy_val_cer < self.prev_val_loss:
                    print("Best till now:")
                    # if self.device == 0:
                    self.save_model(best=True)
                    self.prev_val_loss = greedy_val_cer

                self.test_model(mode="test")

                if epoch % 50 == 0:
                    self.gen_model()


    def assert_greedy_beam_decode(self, dataset):

        def convert_tensor_to_string(input_tensor):
            try:
                out_txt = self.char_model.indexes2characters(input_tensor.cpu().numpy()[0])
            except Exception as e:
                # print(e)
                out_txt = self.char_model.indexes2characters(input_tensor)
            out_txt = "".join(out_txt)
            return out_txt

        self.model.eval()
        with torch.no_grad():
            with progressbar.ProgressBar(max_value=len(dataset)) as bar:
                for index, batch_output in enumerate(dataset):
                    gt = batch_output["img_txt_txt_tgt_out"]

                    eos_index = (batch_output["img_txt_txt_tgt"][0]==2).nonzero(as_tuple=True)[0]
                    real_output = convert_tensor_to_string(batch_output['img_txt_txt_tgt'][:eos_index+1])

                    print("************************************")
                    print("Ground Truth:")
                    print(real_output)
                    # print(batch_output['img_txt_txt_tgt'][0][:eos_index+1].cpu().numpy())
                    print("\n")

                    greedy_output = self.model.generate(batch_output, "img2txt")

                    print("Greedy Output:")
                    # print(greedy_output)
                    print(convert_tensor_to_string(greedy_output))
                    print("\n")


                    beam_size=1
                    beam_output, beam_score, gt_output, gt_score = self.model.beam_search(batch_output, "img2txt", beam_size, gt)
                    print(f"Beam Decoding with size:{beam_size}")
                    for bo, bs in zip(beam_output, beam_score):
                        print(convert_tensor_to_string(bo))
                        print(bs)
                    # print(beam_output)
                    # print(convert_tensor_to_string(beam_output))
                    # print(beam_score)
                    print("GT:")
                    # print(gt_output)
                    print(convert_tensor_to_string(gt_output))
                    print(gt_score)
                    print("\n")

                    beam_size=20
                    beam_output, beam_score, gt_output, gt_score = self.model.beam_search(batch_output, "img2txt", beam_size, gt)
                    print(f"Beam Decoding with size:{beam_size}")
                    for i in np.argsort(beam_score):
                        bo = beam_output[i]
                        bs = beam_score[i]
                        print(convert_tensor_to_string(bo))
                        print(bs)
                    # print(beam_output)
                    # print(convert_tensor_to_string(beam_output))
                    # print(beam_score)
                    print("GT:")
                    # print(gt_output)
                    print(convert_tensor_to_string(gt_output))
                    print(gt_score)
                    print("\n")


                    if index>=25:
                        break
                    print("************************************")

    def compute_greedy_wer(self, dataset, mode="val", search_type="greedy", beam_size=None):
        greedy_wer = []
        greedy_acc = []
        self.model.eval()
        print(f"Search type: {search_type}")
        with torch.no_grad():
            with progressbar.ProgressBar(max_value=len(dataset)) as bar:
                for index, batch_output in enumerate(dataset):

                    if search_type == "greedy":
                        output = self.model.generate(batch_output, "img2txt")
                    else:
                        output, score = self.model.beam_search(batch_output, "img2txt", beam_size)

                    output = output[1:]
                    output = torch.tensor(output).unsqueeze(0)
                    wer, acc = self.calculate_wer(batch_output['img_txt_txt_tgt_out'], output,batch_output['audio_path'], raw_prob=False)
                    greedy_wer.append(wer)
                    greedy_acc.append(acc)

                    bar.update(index)

                print(f"Greedy {mode} wer:", np.average(greedy_wer))
                print(f"Greedy {mode} acc:", np.average(greedy_acc))
                return np.average(greedy_wer)

    @staticmethod
    def compute_modified_wer_cer(predicted, references):
        def format_string_for_wer(str):
            str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
            str = re.sub('([ \n])+', " ", str).strip()
            return str


        def edit_wer_from_list(truth, pred):
            edit = 0
            for pred, gt in zip(pred, truth):
                gt = format_string_for_wer(gt)
                pred = format_string_for_wer(pred)
                gt = gt.split(" ")
                pred = pred.split(" ")
                edit += editdistance.eval(gt, pred)
            return edit

        def nb_words_from_list(list_gt):
            len_ = 0
            for gt in list_gt:
                gt = format_string_for_wer(gt)
                gt = gt.split(" ")
                len_ += len(gt)
            return len_


        def nb_chars_from_list(list_gt):
            return sum([len(t) for t in list_gt])

        predicted = [re.sub("( )+", ' ', t).strip(" ") for t in predicted]
        cer_wo_norm = [editdistance.eval(u, v) for u,v in zip(predicted, references)]
        cer_norm =  nb_chars_from_list(references)
        cer = sum(cer_wo_norm)/cer_norm

        wer_wo_norm = edit_wer_from_list(predicted, references)
        wer_norm = nb_words_from_list(references)
        wer = wer_wo_norm/wer_norm

        print("CER Updated:", cer)
        print("WER Updated:", wer)

    def compute_greedy_wer2(self, dataset, mode="val", search_type="greedy", beam_size=None):
        raw_texts = []
        self.model.eval()
        print(f"Search type: {search_type}")
        if search_type == "beam":
            print(f"Beam Size: {beam_size}")
        with torch.no_grad():
            with progressbar.ProgressBar(max_value=len(dataset)) as bar:
                for index, batch_output in enumerate(dataset):

                    if search_type == "greedy":
                        output, output_codes, _ = self.model.module.generate(batch_output, "img2txt")
                        output_codes_tgt = output_codes[:, 1:-1]
                    else:
                        output, score = self.model.beam_search(batch_output, "img2txt", beam_size)
                        output = output.unsqueeze(0)

                    # output = output[1:]
                    # output = torch.tensor(output).unsqueeze(0)

                    # print("*************************************************")
                    # print("Greedy Output:", output.shape)
                    # print(output)
                    output = output[:, 1:]
                    # print("Greedy Output:", output.shape)
                    # print(output)
                    # print("*************************************************")
                    raw_text = self.calculate_wer(output_codes_tgt, output, batch_output['audio_path'],raw_prob=False, return_raw=True)
                    raw_texts.extend(raw_text)

                    bar.update(index)

                df = pd.DataFrame(raw_texts, columns=["target", "predicted","audio_path"])

                self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
                self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

                df.to_csv(f"{self.config.gen_epoch_path}/{mode}.csv", index=False)

                df["edit_distance"] = df.apply(lambda row: editdistance.eval(row["target"], row["predicted"]), axis=1)
                greedy_cer = df["edit_distance"].sum()/(df.shape[0] * 1024)
                greedy_acc = (df["target"] == df["predicted"]).sum()/df.shape[0]

                print(f"Greedy {mode} cer:", np.average(greedy_cer))
                print(f"Greedy {mode} acc:", np.average(greedy_acc))

                return greedy_cer

    def test_model(self, mode="test"):

        self.model.eval()


        # mode = "test"
        # dataset = self.test_gen_set
        dataset = self.test_set
        if mode=="val":
            dataset = self.val_set
        raw_texts = []
        encoder_raw_texts = []
        total_img2txt_loss = []

        with torch.no_grad():

            print(f"\n{mode} evaluating..............")

            with progressbar.ProgressBar(max_value=len(dataset)) as bar:

                for index, batch_output in enumerate(dataset):
                    try:
                        img2txt_dec_txt, output_codes, style_codes = self.model.module.evaluate(batch_output,  "img2txt")
                        # img2txt_dec_txt, output_codes = self.model(batch_output)
                        img2txt_loss = self.criterion['txt'](img2txt_dec_txt.view(-1, 1027), output_codes.contiguous().view(-1).to(self.device))
                        total_img2txt_loss.append(img2txt_loss.item())

                        raw_text = self.calculate_wer(output_codes, img2txt_dec_txt,batch_output['audio_path'], raw_prob=True, return_raw=True)
                        raw_texts.extend(raw_text)


                    except RuntimeWarning:
                        pass

                    bar.update(index)

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(raw_texts, columns=["target", "predicted","audio_path"])
        df.to_csv(f"{self.config.gen_epoch_path}/{mode}_transformer_tf.csv", index=False)
        df["edit_distance"] = df.apply(lambda row: editdistance.eval(row["target"], row["predicted"]), axis=1)
        cer = df["edit_distance"].sum()/(df.shape[0] * 1027)
        acc = (df["target"] == df["predicted"]).sum()/df.shape[0]

        print(f"{mode}_cer", (cer))
        # print(f"{mode}_wer", (wer))
        print(f"{mode}_acc", (acc))

        # greedy_cer = self.compute_greedy_wer2(dataset, mode)
        if self.current_epoch%self.config.model_eval_epoch==0:
            greedy_cer = self.compute_greedy_wer2(dataset, mode)
        else:
            greedy_cer = np.inf

        return np.average(total_img2txt_loss), cer, greedy_cer


    def eval_model(self, one_by_one=False, index=None, save_model=True, dataset=None):

        if dataset is None:
            dataset=self.val_set

        self.model.eval()

        total_txt2txt_loss = []
        total_img2img_loss = []
        total_img2txt_loss = []
        total_img2txt_back_loss = []
        total_txt2img_loss = []
        val_wer = []
        val_acc = []
        val_raw_texts = []
        val_encoder_raw_texts = []
        greedy_val_wer = []

        with torch.no_grad():

            print("\nEvaluating..............")

            with progressbar.ProgressBar(max_value=len(dataset)) as bar:

                for index, batch_output in enumerate(dataset):

                    """
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
                     """

                    try:
                        img2txt_dec_txt, aux_features = self.model.module.evaluate(batch_output,  "img2txt")
                        img2txt_loss = self.criterion['txt'](img2txt_dec_txt.view(-1, self.char_model.vocab_size),
                                                             batch_output['img_txt_txt_tgt_out'].contiguous().view(-1).to(self.device))
                        total_img2txt_loss.append(img2txt_loss.item())

                        raw_text = self.calculate_wer(batch_output['img_txt_txt_tgt_out'], img2txt_dec_txt,batch_output['audio_path'], raw_prob=True, return_raw=True)
                        val_raw_texts.extend(raw_text)

                        encoder_raw_text = self.calculate_wer(batch_output['img_txt_txt_tgt'], aux_features.permute(1,0,2),batch_output['audio_path'], raw_prob=True, return_raw=True, ctc_mode=True)
                        val_encoder_raw_texts.extend(encoder_raw_text)

                       
                    except RuntimeWarning:
                        pass

                    """

                    try:
                        if one_by_one:
                            if batch_output['img_txt_font'] == index:
                                txt2img_dec_img = self.model.evaluate(batch_output, "txt2img")
                                txt2img_loss = self.criterion['img'](txt2img_dec_img, batch_output['img_txt_img'])
                                # txt2img_loss /= self.val_batch_size
                                total_txt2img_loss.append(txt2img_loss.item())
                        else:
                            txt2img_dec_img, img2txt_dec_txt_back = self.model.evaluate(batch_output, "txt2img")
                            txt2img_loss = self.criterion['img'](txt2img_dec_img, batch_output['img_txt_img'])
                            txt2img_loss /= self.val_batch_size
                            total_txt2img_loss.append(txt2img_loss.item())

                       
                    except RuntimeWarning:
                        pass

                    """
                    bar.update(index)

            total_loss =  total_img2txt_loss

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        val_df = pd.DataFrame(val_raw_texts, columns=["target", "predicted","audio_path"])
        val_df.to_csv(f"{self.config.gen_epoch_path}/tf_val.csv", index=False)
        val_cer = char_error_rate(preds=val_df.predicted.values.tolist(), target=val_df.target.values.tolist())
        val_wer = word_error_rate(preds=val_df.predicted.values.tolist(), target=val_df.target.values.tolist())
        val_acc = (val_df["target"] == val_df["predicted"]).sum()/val_df.shape[0]

        val_encoder_df = pd.DataFrame(val_encoder_raw_texts, columns=["target", "predicted","audio_path"])
        val_encoder_df.to_csv(f"{self.config.gen_epoch_path}/val_encoder.csv", index=False)
        val_encoder_cer = char_error_rate(preds=val_encoder_df.predicted.values.tolist(), target=val_encoder_df.target.values.tolist())
        val_encoder_wer = word_error_rate(preds=val_encoder_df.predicted.values.tolist(), target=val_encoder_df.target.values.tolist())
        val_encoder_acc = (val_encoder_df["target"] == val_encoder_df["predicted"]).sum()/val_encoder_df.shape[0]

        val_loss = np.average(total_loss)
        # val_txt2img_loss = np.average(total_txt2img_loss)
        val_cer = np.average(val_cer)
        val_wer = np.average(val_wer)
        val_acc = np.average(val_acc)

        val_encoder_cer = np.average(val_encoder_cer)
        val_encoder_wer = np.average(val_encoder_wer)
        val_encoder_acc = np.average(val_encoder_acc)

        val_img2txt_loss = np.average(total_img2txt_loss)
        # val_img2txt_back_loss = np.average(total_img2txt_back_loss)

        if self.exp_track is not None:
            self.exp_track.log_metric("val_img2img_loss", np.average(total_img2img_loss))
            self.exp_track.log_metric("val_img2txt_loss", np.average(total_img2txt_loss))
            self.exp_track.log_metric("val_txt2img_loss", np.average(total_txt2img_loss))
            self.exp_track.log_metric("val_total_loss", np.average(total_loss))

        # print("val_txt2img_loss", (val_txt2img_loss))
        print("val_img2txt_loss", (val_img2txt_loss))
        # print("val_img2txt_back_loss", (val_img2txt_back_loss))
        print("val_total_loss", (val_loss))
        print("val_cer", (val_cer))
        print("val_wer", (val_wer))
        print("val_acc", (val_acc))
        self.compute_modified_wer_cer(val_df.predicted.values.tolist(), val_df.target.values.tolist())
        print("val_encoder_cer", (val_encoder_cer))
        print("val_encoder_wer", (val_encoder_wer))
        print("val_encoder_acc", (val_encoder_acc))
        self.compute_modified_wer_cer(val_encoder_df.predicted.values.tolist(), val_encoder_df.target.values.tolist())

        greedy_val_cer = self.compute_greedy_wer2(self.val_set, "val")
        # greedy_val_cer = self.compute_greedy_wer2(self.val_gen_set, "val")
        # greedy_val_cer = self.compute_greedy_wer2(self.val_gen_set, "val", search_type="beam", beam_size=10)
        
        if greedy_val_cer < self.prev_val_loss:
            print("Best till now:")
            if save_model:
                if self.device == 0:
                    self.save_model(best=True)
            self.prev_val_loss = greedy_val_cer

            # self.compute_greedy_wer2(self.test_set, "test")
            # self.test_model()
        # self.compute_greedy_wer(self.val_gen_set, "val")

        # print(self.compute_greedy_wer(self.val_gen_set, "val", "beam", 10))
        # print(self.compute_greedy_wer2(self.val_gen_set, "val", "beam", 10))

        self.test_model()
        self.compute_greedy_wer2(self.test_set, "test")
        # gc.collect()
        # torch.cuda.empty_cache()

        return val_loss, val_img2txt_loss, val_cer, greedy_val_cer


    def gen_model(self):
        print("Generation Started")

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
                for index, batch_output in enumerate(gen_set):

                    for inf_mode in ["greedy", "teacher_forcing"]:

                        if inf_mode == "greedy":
                            output, target, style = self.model.module.generate(batch_output, "img2txt")

                            # print("Output:")
                            # print(output)
                            # print(output.shape)
                            # print("Target:")
                            # print(target)
                            # print(target.shape)
                            # print("Style:")
                            # print(style)
                            # print(style.shape)

                            output = output[:, 1:]
                            target = target[:, 1:-1]
                            style = style[:, 1:-1]

                            # for bs in range(output.shape[0]):
                            #     print("Inference Mode:", inf_mode)
                            #     print("BS:", bs)
                            #     print("output:")
                            #     print(output[bs])
                            #     print("target:")
                            #     print(target[bs])


                        elif inf_mode == "teacher_forcing":
                            output, target, style = self.model(batch_output)

                            output = torch.argmax(output, dim=-1)

                            output = output[:, :-1]
                            target = target[:, :-1]
                            style = style[:, 1:-1]

                            # for bs in range(output.shape[0]):
                            #     print("Inference Mode:", inf_mode)
                            #     print("BS:", bs)
                            #     print("output:")
                            #     print(output[bs])
                            #     print("target:")
                            #     print(target[bs])


                        bs = output.shape[0]
                        out_img = []
                        gap_img = np.zeros((8, 1024))

                        for bs_idx in range(bs):
                            try:
                                # print(output[bs_idx].shape)
                                # print(target[bs_idx].shape)
                                generated_img = self.model.module.codebook.get_reconstructed_img(output[bs_idx].unsqueeze(0), False)
                                # print(generated_img.shape)
                                target_img = self.model.module.codebook.get_reconstructed_img(target[bs_idx].unsqueeze(0), False)
                                style_img = self.model.module.codebook.get_reconstructed_img(style[bs_idx].unsqueeze(0), False)
                                # print(target_img.shape)

                                out_img.append(target_img.cpu().squeeze().numpy()*255)
                                out_img.append(style_img.cpu().squeeze().numpy()*255)
                                out_img.append(generated_img.cpu().squeeze().numpy()*255)
                                out_img.append(gap_img)
                            except Exception as e:
                                print("**********************************************")
                                print("##############################################")
                                print("**********************************************")
                                print("From Inside Loop")
                                print(e)
                                print("**********************************************")
                                print("##############################################")
                                print("**********************************************")
                                continue

                        out_img = np.vstack(out_img)
                        save_img(f"{self.config.gen_epoch_path}/{mode}_{inf_mode}_{self.current_epoch}_{index}.png", out_img, 'real_txt2img')
                    if (index+1)%8==0:
                        break

        self.model.eval()

        self.config.gen_epoch_path = self.config.gen_path/str(self.current_epoch)
        self.config.gen_epoch_path.mkdir(parents=True, exist_ok=True)

        # gen_output(self.test_set, mode="test")

        print("\n\n")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        try:
            print("Train Generation")
            gen_output(self.train_gen_set, mode="train")
        except Exception as e:
            print("Train Generation Failed")
            print(e)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n\n")

        try:
            print("Val Generation")
            gen_output(self.val_gen_set, mode="val")
        except Exception as e:
            print("Val Generation Failed")
            print(e)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n\n")

        try:
            print("Test Generation")
            gen_output(self.test_gen_set, mode="test")
            print("Test Generation Failed")
        except Exception as e:
            print(e)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n\n")

        print("Generation Completed")



    # def test_model(self):
    #     pass
