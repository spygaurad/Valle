from collections import defaultdict
import numpy as np

import torch

class BatchOutput:
    def __init__(self, batch):

        self.img_seq_len = 788
        self.txt_seq_len = 258
        self.bs = None
        self.char_model = None

        self.data = defaultdict(list)
        self.add_items(batch)

        self.device = None

    def generate_pad_mask(self, line):
        mask = (line == self.char_model.char2index["PAD"])
        return mask

    def generate_audio_pad_mask(self, line):
        mask = (line == self.char_model.char2index["IPAD"])
        return mask

    def indexes_from_sentence(self, sentence):
        try:
            return [self.char_model.char2index[char] for char in sentence]
        except Exception as e:
            print(e)
            print(sentence)

    def tensor_from_sentence(self, sentence, max_char_len, src):
        try:

            indexes = self.indexes_from_sentence(sentence)
        except Exception as e:
            print(e)
            print(sentence)

        # if src:
        #     sentence_len_diff = max_char_len - len(sentence)
        # else:
        indexes.insert(0, self.char_model.char2index["TSOS"])
        indexes.append(self.char_model.char2index["TEOS"])
        sentence_len_diff = max_char_len - len(sentence) - 2

        if sentence_len_diff > 0:
            for i in range(0, sentence_len_diff):
                indexes.append(self.char_model.char2index["PAD"])

        return torch.tensor(indexes, dtype=torch.long)

    def tensor_from_audio(self,audio, max_aud_length):
        try:
            indexes = audio.to(torch.long)
        except Exception as e:
            print("nothing here :) ")
        sos = self.char_model.char2index['ISOS']
        eos = self.char_model.char2index['IEOS']
        pad = self.char_model.char2index['IPAD']
        indexes = torch.cat((torch.tensor([sos]),indexes))
        indexes = torch.cat((indexes,torch.tensor([eos])))
        # print(len(indexes), indexes.shape)
        # quit()
        difference = max_aud_length - len(indexes)
        if difference > 0:
            for i in range(0, difference):
                indexes = torch.cat((indexes, torch.tensor([pad])))
                # pass

        return indexes.to(torch.long)

    def tensor_from_font_type(self, font_type, max_char_len):
        return torch.tensor(self.font_model.font2index[font_type])

    @staticmethod
    def get_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    @staticmethod
    def get_mask_seq_cat(first_seq_len=128, second_seq_len=128):
        second_mask = torch.triu(torch.full((second_seq_len, second_seq_len), float('-inf')), diagonal=1)
        first_mask = torch.zeros(second_seq_len, first_seq_len)

        bottom_mask = torch.cat([first_mask, second_mask], axis=-1)
        top_mask = bottom_mask[0].unsqueeze(0).repeat(first_seq_len,1)

        mask = torch.cat([top_mask, bottom_mask], axis=0)
        return mask

# 1, 64,1024 (246,8) (246)
    def pad_images(self, imgs):
        # for i in imgs:
        #     print(i.shape)
        #     break
        # print('Img: ', imgs)
        imgs_height = [i.shape[1] for i in imgs] #64
        imgs_width = [i.shape[0] for i in imgs] #1024
        # print("max height:", max(imgs_height))
        # print("max width:", max(imgs_width))
        data_len, height, width = len(imgs), max(imgs_height), max(imgs_width)
        print(data_len, height, width)
        padded_imgs = np.zeros((data_len, width, height))
        # padded_imgs = np.ones((data_len, height, width)) * 255
        for i, hw in enumerate(zip(imgs_height, imgs_width)):

            h,w = hw
            # print('@'*100)
            # print(h,w)
            padded_imgs[i, :w, :h] = imgs[i][:w, :h]

        # np.repeat(mask[..., np.newaxis], 3, axis=-1)
        padded_imgs = torch.from_numpy(padded_imgs).to(torch.float)
        # padded_imgs = padded_imgs.unsqueeze(1)
        # padded_imgs = padded_imgs.repeat(1, 3, 1, 1)

        # print(imgs_width)
        # print(data_len, height, width)
        # print("Collate padded images:", padded_imgs.shape)
        return padded_imgs


    def __getitem__(self, key):
            # pass
        # print('&&&&&&&&&')
        # print(self.data[key])
        # print('(((((((())))))))')
        if key == "audio_path":
            return self.data[key]
            
        value = torch.stack(self.data[key], 0)
        value = value.to(self.device) if self.device else value
        return value
        # pass

    def add_items(self, batch):

        for item in batch:
            for key, value in item.items():
                self.data[key].append(value)


        if self.data['label'][0] == 2:
            imgs_width = [i.shape[0] for i in self.data["img_txt_img"]]
            # max_img_width = max(imgs_width)
            # max_img_width = 1072
            # print("Max width:", max_img_width)

            self.bs = len(imgs_width)
            # print('*'*12,self.bs)
            # self.img_seq_len = int(np.ceil(max_img_width/8)) + 2
            # self.img_seq_len = 750
            # self.txt_seq_len = 250 #max(self.data["img_txt_txt_tgt_len"]).item()

            # length of 1007
            self.data['src_mask'] = [self.get_mask_seq_cat(self.txt_seq_len,self.img_seq_len)] * self.bs
            # self.data['src_mask'] = [self.get_mask_seq_cat(self.img_seq_len, self.txt_seq_len-1)] * self.bs


            # print('Batch Output')
            # print('-'*30)
            # print(len(self.data['src_mask'][0]))
            # print('-'*30)
            self.char_model = self.data["char_model"][0]
# batch_output['img_txt_txt_tgt'],
#                                             batch_output['img_txt_txt_tgt_pad_mask']

            for line in self.data['img_txt_original_txt']:
                # print('¢'*10, line, type(line))
                
                txt_tgt = self.tensor_from_sentence(line, self.txt_seq_len, False)
                # print(txt_tgt.shape, type(txt_tgt))
                # break
                txt_tgt_in = txt_tgt[:-1]  # Decoder input during training
                txt_tgt_in_pad_mask = self.generate_pad_mask(txt_tgt_in)

                txt_tgt_pad_mask = self.generate_pad_mask(txt_tgt)
                first_sequence_pad_mask = torch.zeros(self.img_seq_len, dtype=torch.bool)
                txt_tgt_in_pad_mask = torch.cat([first_sequence_pad_mask, txt_tgt_in_pad_mask])
                txt_tgt_out = txt_tgt[1:]  # Used during loss calculation

                self.data["img_txt_txt_tgt"].append(txt_tgt)
                self.data["img_txt_txt_tgt_pad_mask"].append(txt_tgt_pad_mask)
                self.data["img_txt_txt_tgt_in"].append(txt_tgt_in)
                self.data["img_txt_txt_tgt_in_pad_mask"].append(txt_tgt_in_pad_mask)
                self.data["img_txt_txt_tgt_out"].append(txt_tgt_out)

            for line in self.data['img_txt_img']:
                # print('¢'*10, line.shape, type(line))
                # print(line)

                audio_tgt = self.tensor_from_audio(line, self.img_seq_len)
                audio_pad_mask = self.generate_audio_pad_mask(audio_tgt)
                self.data['audio'].append(audio_tgt)
                self.data['audio_pad_mask'].append(audio_pad_mask)

            '''
            print('Batch Output')
            print('-'*30)
            # print(self.data['audio_pad_mask'])
            # quit()

            print(len(self.data['audio_pad_mask'][0]))
            print(len(self.data['img_txt_txt_tgt_pad_mask'][0]))

            print('-'*30)
            '''

        elif self.data['label'][0] == 0:
            self.txt_seq_len = 130 #max(self.data["img_txt_txt_tgt_len"]).item()
            self.char_model = self.data["char_model"][0]

            for line in self.data['img_txt_original_txt']:
                txt_tgt = self.tensor_from_sentence(line, self.txt_seq_len, False)
                txt_tgt_in = txt_tgt[:-1]  # Decoder input during training
                txt_tgt_in_pad_mask = self.generate_pad_mask(txt_tgt_in)
                txt_tgt_in_ar_mask = self.get_mask(self.txt_seq_len-1)
                txt_tgt_out = txt_tgt[1:]  # Used during loss calculation

                self.data["txt_txt_tgt"].append(txt_tgt)
                self.data["txt_txt_tgt_in"].append(txt_tgt_in)
                self.data["txt_txt_tgt_in_mask"].append(txt_tgt_in_ar_mask)
                self.data["txt_txt_tgt_in_pad_mask"].append(txt_tgt_in_pad_mask)
                self.data["txt_txt_tgt_out"].append(txt_tgt_out)
         



