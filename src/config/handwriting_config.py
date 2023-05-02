from ast import literal_eval
from pathlib import Path

from src.config.config import Config


class HandwritingConfig(Config):

    # DATA_DIR = 'data_50K/handwriting'
    # DATA_DIR = 'dataset_with_style_5K/handwriting'
    # DATA_DIR = 'dataset_with_style_1M/handwriting'
    # DATA_DIR = 'dataset_with_style_1_25M/handwriting'
    # DATA_DIR = 'dataset_with_style_2M/handwriting'
    # DATA_DIR = 'dataset_with_style/handwriting'
    # DATA_DIR = 'ncell_data/handwriting'
    # DATA_DIR = 'mixed_synthetic_nepali_data/handwriting'
    # DATA_DIR = 'dataset_iam_words/handwriting'
    DATA_DIR = 'dataset_iam_lines/handwriting'

    # DATA_DIR = 'dataset_iam_lines_v2/handwriting'
    # DATA_DIR = 'dataset_pretraining_font/handwriting'
    # DATA_DIR = 'dataset_pretraining_handwriting_transformers/handwriting'
    # DATA_DIR = 'dataset_pretraining_handwriting_transformers_v2/handwriting'
    # DATA_DIR = 'dataset_pretraining_handwriting_transformers_iam/handwriting'
    # DATA_DIR = 'dataset_pretraining_stackmix/handwriting'
    # DATA_DIR = 'dataset_pretraining_stackmix_iam/handwriting'
    # DATA_DIR = 'dataset_pretraining/handwriting'

    # DATA_DIR = 'dataset_synthetic_iam_lines/handwriting'
    # DATA_DIR='synthetic_handwritten_hindi_data/handwriting'
    # DATA_DIR='synthetic_handwritten_hindi_ncell_data/handwriting'
    # DATA_DIR = 'synthetic_hindi_data/handwriting'
    # DATA_DIR = 'hindi_data/handwriting'

    # DATASET_DIR = Path(DATA_DIR)/'dataset'
    DATASET_DIR = Path(DATA_DIR)/'dataset_v2'
    # DATASET_DIR = Path(DATA_DIR)/'dataset_style'
    VOCAB_DIR = Path(DATA_DIR)/'vocab'

    # CHAR_VOCAB_FILE = Path(VOCAB_DIR)/'char.txt'
    CHAR_VOCAB_FILE = Path("./")/'combined_char.txt'
    # FONT_VOCAB_FILE = Path(VOCAB_DIR)/'font.txt'
    FONT_VOCAB_FILE = Path(VOCAB_DIR)/'writer.txt'

    # IMAGE_DIR = Path(DATA_DIR)/'image_fxd_ht'
    # IMAGE_DIR = Path(DATA_DIR)/'image'
    IMAGE_DIR = Path(DATA_DIR)/'lines'
    EXP_DIR = Path('out/handwriting')

    TXT_DATASET = Path(DATASET_DIR)/'text.csv'
    IMG_DATASET = Path(DATASET_DIR)/'image.csv'
    IMG_TXT_DATASET = Path(DATASET_DIR)/'image_text.csv'

    TRAIN_DATASET = Path(DATASET_DIR)/'train.csv'
    # TRAIN_DATASET = Path(DATASET_DIR)/'train_val.csv'
    # TRAIN_DATASET = Path(DATASET_DIR)/'train_unknown.csv'
    EVAL_DATASET = Path(DATASET_DIR)/'eval.csv'
    TEST_DATASET = Path(DATASET_DIR)/'test.csv'
    # TEST_DATASET = Path(DATASET_DIR)/'stackmix_train.csv'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mixed_precision_training = kwargs.pop('mixed_precision_training', False)
        self.random_masking = kwargs.pop('random_masking', False)
        self.distillation = kwargs.pop('distillation', False)
        self.aux_ctc = kwargs.pop('aux_ctc', False)

        self.txt_batch_size = kwargs.pop('txt_batch_size', 12)
        self.img_batch_size = kwargs.pop('img_batch_size', 12)
        self.img_txt_batch_size = kwargs.pop('img_txt_batch_size', 12)
        # self.batch_size = self.txt_batch_size + self.img_batch_size + self.img_txt_batch_size
        self.batch_size = self.img_txt_batch_size
        self.iter_dataset_index = kwargs.pop('iter_dataset_index', 0)
        self.font_embedding_dim = kwargs.pop('font_embedding_dim', 64)

        self.resnet_encoder = kwargs.pop('resnet_encoder', {})
        self.resnet_encoder['channels'] = literal_eval(
            self.resnet_encoder['channels']) if 'channels' in self.resnet_encoder else []
        self.resnet_encoder['strides'] = literal_eval(
            self.resnet_encoder['strides']) if 'strides' in self.resnet_encoder else []
        self.resnet_encoder['depths'] = literal_eval(
            self.resnet_encoder['depths']) if 'depths' in self.resnet_encoder else []

        self.resnet_decoder = kwargs.pop('resnet_decoder', {})
        self.resnet_decoder['reshape_size'] = literal_eval(
                self.resnet_decoder['reshape_size']) if 'reshape_size' in self.resnet_decoder else []
        self.resnet_decoder['channels'] = literal_eval(
            self.resnet_decoder['channels']) if 'channels' in self.resnet_decoder else []
        self.resnet_decoder['strides'] = literal_eval(
            self.resnet_decoder['strides']) if 'strides' in self.resnet_decoder else []
        self.resnet_decoder['depths'] = literal_eval(
            self.resnet_decoder['depths']) if 'depths' in self.resnet_decoder else []
        self.resnet_decoder['kernels_size'] = literal_eval(
            self.resnet_decoder['kernels_size']) if 'kernels_size' in self.resnet_decoder else []
        self.resnet_decoder['paddings'] = literal_eval(
            self.resnet_decoder['paddings']) if 'paddings' in self.resnet_decoder else []

        self.max_char_len = kwargs.pop('max_char_len', 128)
        self.bleu_score_ngram = kwargs.pop('bleu_score_ngram', 16)
        self.bleu_score_weights = [1/self.bleu_score_ngram] * self.bleu_score_ngram

    def update_batch_size(self, device_count):
        if device_count:
            self.txt_batch_size *= device_count
            self.img_batch_size *= device_count
            self.img_txt_batch_size *= device_count
            self.batch_size *= device_count
