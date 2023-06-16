import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from src.data.handwriting.batch_sampler import BatchSampler
from src.data.char_model import CharModel
from src.data.font_model import FontModel
from src.data.handwriting.handwriting_dataset import collate_fn
from src.data.handwriting.handwriting_dataset import LibriSpeechDataset, ImgDataset, ImgTxtDataset, TxtDataset, StyleDataset, SyntheticStyleDataset

# torch.multiprocessing.set_sharing_strategy('file_descriptor')

class HandwritingDataloader:
    def __init__(self, config):

        self.config = config

        self.char_model = CharModel()
        self.char_model(self.read_vocab_file(self.config.CHAR_VOCAB_FILE))
        print(self.char_model.index2char)
        print(self.char_model.char2index)
        print(len(self.char_model.index2char))
        print(len(self.char_model.char2index))
        print(self.char_model.char2lm)
        print(self.char_model.n_chars)
        print(self.char_model.vocab_size)

        self.font_model = FontModel()
        self.font_model(self.read_vocab_file(self.config.CHAR_VOCAB_FILE))
        print("*****************************Font Model *************************************")
        # print(self.font_model.font2index)
        # print(self.font_model.n_fonts)
        print("*****************************Font Model *************************************")

        self.txt_data = pd.read_csv(self.config.TXT_DATASET).sample(frac=1)
        self.img_data = pd.read_csv(self.config.IMG_DATASET).sample(frac=1)
        self.img_txt_data = pd.read_csv(self.config.IMG_TXT_DATASET).sample(frac=1)
        # self.img_txt_data = self.img_txt_data[self.img_txt_data.font_type != 'caveat']
        # self.img_txt_data = self.img_txt_data[self.img_txt_data.font_type != 'sacramento']
        try:
            print(self.config.TRAIN_DATASET)
            self.train_data = pd.read_csv(self.config.TRAIN_DATASET).sample(frac=1)#[:16]
            # self.eval_data = pd.read_csv(self.config.EVAL_DATASET)[:]#.sample(frac=1)
            self.eval_data = pd.read_csv(self.config.EVAL_DATASET)#.sample(frac=1)#[:16]
            self.test_data = pd.read_csv(self.config.TEST_DATASET)#.sample(frac=1)
            # self.train_data = pd.read_csv(self.config.TRAIN_DATASET).sample(frac=1)[:15000]
            # self.eval_data = pd.read_csv(self.config.EVAL_DATASET)[:1000]#.sample(frac=1)
            # self.test_data = pd.read_csv(self.config.TEST_DATASET)[:3000]#.sample(frac=1)
        except Exception as e:
            print(e)
            self.train_data = None
            self.eval_data = None
            self.test_data = None

    def read_vocab_file(self, vocab_file):

        with open(vocab_file, "r") as f:
            collection = f.read().splitlines()

        return collection

    def split_data(self, dataset):


        """
        train_len = 2
        test_len = 2
        val_len = 2

        val_test_len = 6
        val_gen_len = 4

        """
        # dataset_len = 5000
        '''
        dataset_len = len(dataset)
        # dataset_len = 1000 
        train_len = int(0.99 * dataset_len)
        val_test_len = dataset_len - train_len
        val_gen_len = int(0.5 * val_test_len)
        test_len = val_test_len - val_gen_len
        val_len = val_gen_len - self.config.eval_gen_data_length
        #"""
        print(train_len, val_len, val_gen_len, test_len)
        # return train_len, val_len, val_gen_len, test_len
        '''
        return 4, 2, 2, 2

        # train_set, val_test = random_split(
        #     dataset,
        #     [train_len, val_test_len],
        #     generator=torch.Generator().manual_seed(42)
        # )

        # val_gen_set, test_set = random_split(
        #     val_test,
        #     [val_gen_len, test_len],
        #     generator=torch.Generator().manual_seed(42)
        # )

        # val_set, gen_set = random_split(
        #     val_gen_set,
        #     [val_len, self.config.eval_gen_data_length],
        #     generator=torch.Generator().manual_seed(42)
        # )

        train_set, val_test = random_split(
            dataset,
            [train_len, val_test_len],
        )

        val_gen_set, test_set = random_split(
            val_test,
            [val_gen_len, test_len],
        )

        val_set, gen_set = random_split(
            val_gen_set,
            [val_len, self.config.eval_gen_data_length],
        )

        return train_set, val_set, test_set, gen_set

    def load_data(self):
        """
        data_len = 8

        # txt_dataset = TxtDataset(self.txt_data[:data_len], self.char_model, self.config.max_char_len)
        # img_dataset = ImgDataset(self.img_data[:data_len], self.font_model, self.config.max_char_len, self.config.IMAGE_DIR)
        img_txt_dataset = ImgTxtDataset(self.img_txt_data[:data_len], self.char_model, self.font_model,
                                        self.config.max_char_len, self.config.IMAGE_DIR)
        """

        # txt_dataset = TxtDataset(self.txt_data[:], self.char_model, self.config.max_char_len)
        # img_dataset = ImgDataset(self.img_data[:], self.font_model, self.config.max_char_len, self.config.IMAGE_DIR)
        # img_txt_dataset = ImgTxtDataset(self.img_txt_data[:], self.char_model, self.font_model,
        #                                 self.config.max_char_len, self.config.IMAGE_DIR)
        # """

        # txt_train_set, txt_val_set, txt_test_set, txt_gen_set = self.split_data(txt_dataset)
        # img_train_set, img_val_set, img_test_set, img_gen_set = self.split_data(img_dataset)
        augment = True
        style_transfer = True
        print(f"Style transfer status: {style_transfer}")
        if self.train_data is not None:
            print("Custom train/val/test split")
            if not style_transfer:
                print("ImageTxt Dataset")
                img_txt_train_set = LibriSpeechDataset(self.train_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, augment)
                img_txt_val_set = LibriSpeechDataset(self.eval_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, False)
                img_txt_gen_set = LibriSpeechDataset(self.eval_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, False)
                img_txt_test_set = LibriSpeechDataset(self.test_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, False)
            else:
                print("Style Dataset")
                df_file_path = "/home/wiseyak/suraj/everything_text_valle/Valle/audio_dataset/"


                # test-other-transcript_csv.csv
                
                # train_file_path = df_file_path + "test-other-transcript_csv.csv"
                # train_file_1 = [
                #                 df_file_path + "train_960-transcript_0_csv.csv",
                #                 df_file_path + "train_960-transcript_1_csv.csv",
                #                 df_file_path + "train_960-transcript_2_csv.csv",
                #                 df_file_path + "train_960-transcript_3_csv.csv",
                #                 # df_file_path + "train_960-transcript_4_csv.csv",
                #                 # df_file_path + "train_960-transcript_5_csv.csv",
                #                 # df_file_path + "train_960-transcript_6_csv.csv",
                #                 # df_file_path + "train_960-transcript_7_csv.csv",
                #                 # df_file_path + "train_960-transcript_8_csv.csv",
                #                 # df_file_path + "train_960-transcript_9_csv.csv",
                #                 # df_file_path + "train_960-transcript_10_csv.csv",
                #                 # df_file_path + "train_960-transcript_11_csv.csv",
                #                 # df_file_path + "train_960-transcript_12_csv.csv",
                #                 # df_file_path + "train_960-transcript_13_csv.csv",
                #                 # df_file_path + "train_960-transcript_14_csv.csv",
                #                 # df_file_path + "train_960-transcript_15_csv.csv"
                #                 # df_file_path + "train_960-transcript_16_csv.csv",
                #                 ]

                # eval_other_path = df_file_path + "dev-other-transcript_csv.csv"
                # test_file_path = df_file_path + "test-clean-transcript_csv.csv"
                # eval_file_path = df_file_path + "dev-clean-transcript_csv.csv"

                # train_file_path = df_file_path + "test-other-transcript_csv.csv"
                train_file_1 = [
                                df_file_path + "train-clean-100-encodec-transcript.txt",
                                # df_file_path + "train_960-transcript_1_csv.csv",
                                # df_file_path + "train_960-transcript_2_csv.csv",
                                # df_file_path + "train_960-transcript_3_csv.csv",
                                # df_file_path + "train_960-transcript_4_csv.csv",
                                # df_file_path + "train_960-transcript_5_csv.csv",
                                # df_file_path + "train_960-transcript_6_csv.csv",
                                # df_file_path + "train_960-transcript_7_csv.csv",
                                # df_file_path + "train_960-transcript_8_csv.csv",
                                # df_file_path + "train_960-transcript_9_csv.csv",
                                # df_file_path + "train_960-transcript_10_csv.csv",
                                # df_file_path + "train_960-transcript_11_csv.csv",
                                # df_file_path + "train_960-transcript_12_csv.csv",
                                # df_file_path + "train_960-transcript_13_csv.csv",
                                # df_file_path + "train_960-transcript_14_csv.csv",
                                # df_file_path + "train_960-transcript_15_csv.csv"
                                # df_file_path + "train_960-transcript_16_csv.csv",
                                ]

                eval_other_path = df_file_path + "dev-other-encodec-transcript.txt"
                test_file_path = df_file_path + "test-clean-encodec-transcript.txt"
                eval_file_path = df_file_path + "dev-clean-encodec-transcript.txt"



                # df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)


                # self.train_data = pd.read_csv(train_file_path, nrows=8)#[:20000]
                # '''
                self.train_data = pd.concat((pd.read_csv(f, sep=',') for f in train_file_1))
                # self.train_data.columns = ['raw_audio','text', 'transcript_formatted','audio']

                self.eval_data = pd.read_csv(test_file_path, sep=',')
                # self.eval_data.columns = ['raw_audio','text', 'transcript_formatted','audio']

                self.test_data = pd.read_csv(eval_other_path, sep=',')
                # self.test_data.columns = ['raw_audio','text', 'transcript_formatted','audio']

                self.eval_gen_data = pd.read_csv(eval_file_path, sep=',')
                # self.eval_gen_data.columns = ['raw_audio','text', 'transcript_formatted','audio']

                self.train_data = self.train_data[self.train_data["transcript"].apply(lambda x: len(x) < 250)]
                self.eval_data = self.eval_data[self.eval_data["transcript"].apply(lambda x: len(x) < 250)]
                self.test_data = self.test_data[self.test_data["transcript"].apply(lambda x: len(x) < 250)]
                self.eval_gen_data = self.eval_gen_data[self.eval_gen_data["transcript"].apply(lambda x: len(x) < 250)]
                # self.train_data = self.train_data[self.train_data["transcript"].apply(lambda x: len(x) < 250)]


                '''
                # small dataset
                self.train_data = pd.concat((pd.read_csv(f, nrows=1) for f in train_file_1))
                self.eval_data = pd.read_csv(test_file_path, nrows=4)
                self.test_data = pd.read_csv(eval_other_path, nrows=4)
                self.eval_gen_data = pd.read_csv(eval_file_path, nrows=4)
                #'''
                
                



                # index_to_remove = []
                # for index, row in self.img_txt_data.iterrows():
                #     if len(row["text"]) > 40 or len(row["text"])< 15:
                #         index_to_remove.append(index)
                # self.img_txt_data = self.img_txt_data.drop(index=index_to_remove)

                # self.img_txt_data = pd.read_fwf(df_file_path)[:200]
                # print(len(self.img_txt_data))
                '''
                train_len, val_len, val_gen_len, test_len = self.split_data(self.img_txt_data)


                self.train_data = self.img_txt_data[:train_len].reset_index(drop=True)

                self.eval_data = self.img_txt_data[train_len: train_len+val_len].reset_index(drop=True)

                # self.eval_gen_data = self.img_txt_data[train_len+val_len:train_len+val_len+val_gen_len].reset_index(drop=True)

                self.test_data = self.img_txt_data[train_len+val_len:].reset_index(drop=True)
                self.eval_gen_data = self.test_data
                '''

                img_txt_train_set = LibriSpeechDataset(self.train_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, augment)
                img_txt_val_set = LibriSpeechDataset(self.eval_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, False)
                img_txt_gen_set = LibriSpeechDataset(self.eval_gen_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, False)
                img_txt_test_set = LibriSpeechDataset(self.test_data[:], self.char_model, self.font_model,
                                                self.config.max_char_len, self.config.IMAGE_DIR, False)

                # img_txt_train_set = StyleDataset(self.train_data[:], self.char_model, self.font_model,
                #                                 self.config.max_char_len, self.config.IMAGE_DIR, augment)
                # img_txt_val_set = StyleDataset(self.eval_data[:], self.char_model, self.font_model,
                #                                 self.config.max_char_len, self.config.IMAGE_DIR, False)
                # img_txt_gen_set = StyleDataset(self.eval_data[:], self.char_model, self.font_model,
                #                                 self.config.max_char_len, self.config.IMAGE_DIR, False)
                # img_txt_test_set = StyleDataset(self.test_data[:], self.char_model, self.font_model,
                #                                 self.config.max_char_len, self.config.IMAGE_DIR, False)
            print("Train Size:", len(img_txt_train_set))
            print("Val Size:", len(img_txt_val_set))
            print("Test Size:", len(img_txt_test_set))
        else:
            print("Automatic train/val/test split")
            # img_txt_dataset = ImgTxtDataset(self.img_txt_data[:], self.char_model, self.font_model,
            #                                 self.config.max_char_len, self.config.IMAGE_DIR)
            # train_len, val_len, val_gen_len, test_len = self.split_data(self.img_txt_data)
            self.train_data = self.img_txt_data[:train_len].reset_index(drop=True)
            self.eval_data = self.img_txt_data[train_len: train_len+val_len].reset_index(drop=True)
            self.eval_gen_data = self.img_txt_data[train_len+val_len:train_len+val_len+val_gen_len].reset_index(drop=True)
            self.test_data = self.img_txt_data[train_len+val_len+val_gen_len:].reset_index(drop=True)

            img_txt_train_set = LibriSpeechDataset(self.train_data[:], self.char_model, self.font_model,
                                            self.config.max_char_len, self.config.IMAGE_DIR, augment)
            img_txt_val_set = LibriSpeechDataset(self.eval_data[:], self.char_model, self.font_model,
                                            self.config.max_char_len, self.config.IMAGE_DIR, False)
            img_txt_gen_set = LibriSpeechDataset(self.test_data[:], self.char_model, self.font_model,
                                            self.config.max_char_len, self.config.IMAGE_DIR, False)
            img_txt_test_set = LibriSpeechDataset(self.test_data[:], self.char_model, self.font_model,
                                            self.config.max_char_len, self.config.IMAGE_DIR, False)

        train_set = ConcatDataset([img_txt_train_set])
        val_set = ConcatDataset([img_txt_val_set])
        test_set = ConcatDataset([img_txt_test_set])
        gen_set = ConcatDataset([img_txt_gen_set])

        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set, shuffle=True) 
        # val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_set, shuffle=True) 
        # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=test_set, shuffle=True) 
        # gen_sampler = torch.utils.data.distributed.DistributedSampler(dataset=gen_set, shuffle=True) 

        # sampler = BatchSampler(train_set,
        #                        self.config.txt_batch_size,
        #                        self.config.img_batch_size,
        #                        self.config.img_txt_batch_size,
        #                        self.config.iter_dataset_index)

        print("Batch", self.config.batch_size)

        # self.config.num_workers = 0


        train_dataloader = DataLoader(train_set, batch_size=self.config.batch_size,
                                      shuffle=False, num_workers=self.config.num_workers, drop_last=False,
                                      collate_fn=collate_fn, pin_memory=True)

        val_dataloader = DataLoader(val_set, batch_size=self.config.batch_size,
                                    shuffle=False, num_workers=self.config.num_workers, drop_last=False,
                                    collate_fn=collate_fn, pin_memory=True)

        test_dataloader = DataLoader(test_set, batch_size=self.config.batch_size,
                                     shuffle=False, num_workers=self.config.num_workers, drop_last=False,
                                     collate_fn=collate_fn, pin_memory=True)

        test_gen_dataloader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=self.config.num_workers, drop_last=False,
                                     collate_fn=collate_fn, pin_memory=True)


        val_gen_dataloader = DataLoader(val_set, batch_size=1,
                                    shuffle=False, num_workers=self.config.num_workers,drop_last=False,
                                    collate_fn=collate_fn, pin_memory=True)

        train_gen_dataloader = DataLoader(train_set, batch_size=1,
                                    shuffle=self.config.shuffle, num_workers=self.config.num_workers, drop_last=False,
                                    collate_fn=collate_fn, pin_memory=True)

        return train_dataloader, val_dataloader, test_dataloader, train_gen_dataloader, val_gen_dataloader, test_gen_dataloader, self.char_model, self.font_model

    def load_small_data(self):

        data_len = 5000
        test_data_len = 100
        bs = 32

        txt_dataset = TxtDataset(self.txt_data[:data_len], self.char_model, self.config.max_char_len)
        img_dataset = ImgDataset(self.img_data[:data_len], self.font_model, self.config.max_char_len, self.config.IMAGE_DIR)
        img_txt_dataset = ImgTxtDataset(self.img_txt_data[:data_len], self.char_model, self.font_model,
                                        self.config.max_char_len, self.config.IMAGE_DIR)

        # test_img_txt_dataset = ImgTxtDataset(self.img_txt_data[-test_data_len:], self.char_model, self.font_model,
        #                                 self.config.max_char_len, self.config.IMAGE_DIR)

        dataset = ConcatDataset([img_txt_dataset])

        sampler = BatchSampler(dataset, bs, bs, bs, 0)
        # self.config.num_workers = 0

        train_dataloader = DataLoader(dataset, batch_size=bs,
                                      shuffle=self.config.shuffle, num_workers=self.config.num_workers,
                                      collate_fn=collate_fn)

        val_dataloader = DataLoader(dataset, batch_size=bs,
                                    shuffle=self.config.shuffle, num_workers=self.config.num_workers,
                                    collate_fn=collate_fn)

        test_dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=self.config.shuffle, num_workers=self.config.num_workers,
                                     collate_fn=collate_fn)

        gen_dataloader = DataLoader(dataset, batch_size=1,
                                    shuffle=self.config.shuffle, num_workers=self.config.num_workers,
                                    collate_fn=collate_fn)

        return train_dataloader, val_dataloader, test_dataloader, gen_dataloader, self.char_model, self.font_model

    def __call__(self, data_type):

        if data_type == 'full':
            return self.load_data()
        elif data_type == 'small':
            return self.load_small_data()
