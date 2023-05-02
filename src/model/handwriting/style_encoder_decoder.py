import torch
import torch.nn as nn
import torchvision
from src.encoding.PositionalEncoding import PositionalEncoding, LearnablePositionalEncoding
from src.model.handwriting.v_image_encoder_blocks import FCN_Encoder_Style, FCN_Encoder_Style2, FCN_Encoder_SE_Style
from src.model.handwriting.image_decoder import FCN_Decoder_Style

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 1, (3, 3)),
)

vgg_style_encoder = nn.Sequential(
    nn.Conv2d(1, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-2
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-3
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-4
    # nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-1
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-2
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-3
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU()  # relu5-4
)

content_encoder = nn.Sequential(
    nn.Conv2d(1, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-2
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-3
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-4
    # nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-1
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-2
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-3
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU()  # relu5-4
)

class ImgCharModel:
    def __init__(self):
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 2

        self.char2index['ISOS'] = 0
        self.char2index['IEOS'] = 1
        
        self.index2char[0] = 'ISOS'
        self.index2char[1] = 'IEOS'

class STImageEncoder(nn.Module):
    def __init__(self, char_model, device):
        super(STImageEncoder, self).__init__()

        self.char_model = char_model
        self.device = device
        params = {"dropout": 0.5, "input_channels": 1}

        self.content_encoder = FCN_Encoder_SE_Style(params)
        self.ada_pool = nn.AdaptiveMaxPool2d((1, None))
        self.char_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.char_model.vocab_size+1)
        )

        img_char_model = ImgCharModel()
        sos_token = img_char_model.char2index['ISOS']
        self.sos_token = torch.LongTensor([[sos_token]])
        eos_token = img_char_model.char2index['IEOS']
        self.eos_token = torch.LongTensor([[eos_token]])
        self.img_embedding = nn.Embedding(img_char_model.n_chars, 256)


    def encode_content(self, input):
       _, result = self.content_encoder(input)
       return result

    def get_char_seq_prediction(self, content_feat):
       content_seq = content_feat
       # print("Content seq", content_seq.shape)
       content_seq = self.ada_pool(content_seq)
       # print("Content seq pool", content_seq.shape)
       content_seq = content_seq.squeeze(dim=-2).permute(0, 2, 1)
       # print("Content seq pool", content_seq.shape)
       bs = content_seq.shape[0]
       sos_token = self.img_embedding(self.sos_token.to(self.device))
       sos_token = sos_token.repeat(bs, 1, 1)
       eos_token = self.img_embedding(self.eos_token.to(self.device))
       eos_token = eos_token.repeat(bs, 1, 1)
       content_seq = torch.cat([sos_token, content_seq, eos_token], axis=1)
       # print("Content seq", content_seq.shape)
       char_seq_prediction = self.char_classifier(content_seq).permute(1, 0, 2).log_softmax(2)
       return char_seq_prediction

    def forward(self, input):
       content_feat =  self.encode_content(input)
       char_seq_prediction = self.get_char_seq_prediction(content_feat)
       return content_feat, char_seq_prediction


class STStyleEncoder(nn.Module):
    def __init__(self, num_writers):
        super(STStyleEncoder, self).__init__()
        print("Writers:", num_writers)

        params = {"dropout": 0.5, "input_channels": 1}

        self.style_encoder = FCN_Encoder_SE_Style(params)
        # self.style_encoder = vgg_style_encoder
        # self.style_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(torchvision.models.resnet18(pretrained=True).children())[1:-1]))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.writer_classifier = nn.Linear(256, num_writers)
        self.writer_classifier= nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_writers)
        )

    def encode_style(self, input):
        result = self.style_encoder(input)
        return result


    def get_style_prediction(self, style_feat):
       # print("Style feat", style_feat.shape)
       # style_feat_global_pool = style_feat
       style_feat_global_pool = self.global_avg_pool(style_feat).squeeze(dim=-1).squeeze(dim=-1)
       # print("Style feats pool:", style_feat_global_pool.shape)
       # style_feat_global_pool = style_feat.view(style_feat.size(0), -1)
       # style_feat_global_pool = style_feat_global_pool.view(style_feat.size(0), -1)
       # print("Style feats pool:", style_feat_global_pool.shape)
       style_prediction = self.writer_classifier(style_feat_global_pool)
       # print(style_prediction.shape)
       return style_prediction


    def forward(self, input):
       style_feats = self.encode_style(input)
       style_prediction =  self.get_style_prediction(style_feats[-1])
       return style_feats, style_prediction


class StyleNet(nn.Module):
    def __init__(self, config, char_model, num_writers, device):
        super(StyleNet,self).__init__()

        self.config = config
        self.char_model = char_model
        self.num_writers = num_writers
        self.device = device

        self.content_encoder = STImageEncoder(self.char_model, self.device)
        self.style_encoder = STStyleEncoder(self.num_writers)
        self.decoder = FCN_Decoder_Style(res_norm = 'in')

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ctc_loss = nn.CTCLoss(blank=self.char_model.vocab_size, reduction="sum", zero_infinity=True)

    def calc_content_loss(self, input, target):
       assert (input.size() == target.size())
       assert (target.requires_grad is False)
       return self.mse_loss(input, target)/input.size()[0]

    def calc_autoencoder_loss(self, input, target):
       assert (input.size() == target.size())
       assert (target.requires_grad is False)
       return self.mse_loss(input, target)/input.size()[0]

    def calc_style_loss(self, input, target):
       assert (input.size() == target.size())
       assert (target.requires_grad is False)
       input_mean, input_std = calc_mean_std(input)
       target_mean, target_std = calc_mean_std(target)
       return (self.mse_loss(input_mean, target_mean) + \
              self.mse_loss(input_std, target_std))/input.size()[0]

    def calc_ce_loss(self, input, target):
       assert (target.requires_grad is False)
       return self.ce_loss(input, target)

    def calc_ctc_loss(self, input, target, input_len, target_len):
       return self.ctc_loss(input, target, input_len, target_len)/input.size()[1]

    def train_encoders(self, content_img, style_img, content_txt, content_txt_len, writers):
        content_feat, char_seq_prediction = self.content_encoder(content_img)
        seq_len, bs, _ = char_seq_prediction.shape
        char_seq_prediction_len = torch.full(size=(bs,), fill_value=seq_len, dtype=torch.long).to(self.device)
        loss_ctc = self.calc_ctc_loss(char_seq_prediction, content_txt, char_seq_prediction_len, content_txt_len)
        predicted_char_seq = torch.argmax(char_seq_prediction.permute(1, 0, 2), dim=-1)

        style_feats, style_prediction = self.style_encoder(style_img)
        loss_ce = self.calc_ce_loss(style_prediction, writers)
        predicted_writers = torch.argmax(style_prediction, dim=-1)

        return loss_ctc, loss_ce, predicted_char_seq, predicted_writers

    def train_decoder(self, content_img, style_img_collection, content_txt, content_txt_len, writers, out_only=False):
        with torch.no_gard():
            content_feat = self.content_encoder.encode_content(content_img)
            style_feats_from_collection = self.style_encoder.encode_style(style_img_collection)

        t = adaptive_instance_normalization(content_feat, style_feats_from_collection[-1])

        out = self.decoder(t)
        if out_only:
            return out

        with torch.no_grad():
            out_style_feats = self.style_encoder.encode_style(out)
            out_content_feat = self.content_encoder.encode_content(out)

        out_content_feat, out_char_seq_prediction = self.content_encoder(out)
        out_seq_len, out_bs, _ = out_char_seq_prediction.shape
        out_char_seq_prediction_len = torch.full(size=(out_bs,), fill_value=out_seq_len, dtype=torch.long).to(self.device)
        out_loss_ctc = self.calc_ctc_loss(out_char_seq_prediction, content_txt, out_char_seq_prediction_len, content_txt_len)
        out_predicted_char_seq = torch.argmax(out_char_seq_prediction.permute(1, 0, 2), dim=-1)

        out_style_feats, out_style_prediction = self.style_encoder(out)
        out_loss_ce = self.calc_ce_loss(out_style_prediction, writers)
        out_predicted_writers = torch.argmax(out_style_prediction, dim=-1)

        loss_c = self.calc_content_loss(out_content_feat, content_feat.detach())
        loss_s = self.calc_style_loss(out_style_feats[0], style_feats[0].detach())
        for i in range(1, 2):
            loss_s += self.calc_style_loss(out_style_feats[i], style_feats[i].detach())

        autoencoder_t = adaptive_instance_normalization(content_feat, content_feat)
        autoencoder_out = self.decoder(autoencoder_t)
        loss_auto = self.calc_autoencoder_loss(autoencoder_out, content_img)


         
    def forward(self, content_img, style_img, style_img_collection, content_txt=None, content_txt_len=None, writers=None, out_only=False, verbose=False):
       # style_feats, style_prediction = self.style_encoder(style_img)
       # loss_ce = self.calc_ce_loss(style_prediction, writers)
       # predicted_writers = torch.argmax(style_prediction, dim=-1)
       # """

       style_feats_from_collection = self.style_encoder.encode_style(style_img_collection)
       t = adaptive_instance_normalization(content_feat, style_feats_from_collection[-1])

       out = self.decoder(t)
       if out_only:
           return out


       # verbose = True
       if verbose:
           print("Content img", content_img.shape)
           print("Content feat", content_feat.shape)
           print("Char seq prediction:", char_seq_prediction.shape)
           print("CTC loss:", loss_ctc)
           print('Style img', style_img.shape)
           print("Style feat:", style_feats[-1].shape)
           print("Style predition:", style_prediction.shape)
           print("CE Loss:", loss_ce)
           print("Style img collection:", style_img_collection.shape)
           print("Style Feats from collection:", style_feats_from_collection[-1].shape)
           print("Content feat:", content_feat.shape)
           print("T:", t.shape)
           print("Out:", out.shape)
           print("Out Content feat", out_content_feat.shape)
           print("Out Char seq prediction:", out_char_seq_prediction.shape)
           print("Out CTC loss:", out_loss_ctc)
           print("Style feat:", out_style_feats[-1].shape)
           print("Style predition:", out_style_prediction.shape)
           print("Out CE Loss:", out_loss_ce)
           print("Autoencoder T:", autoencoder_t.shape)
           print("Autoencoder Out:", autoencoder_out.shape)
           print("Auto Loss:", loss_auto)

       return_dict = {
               "outputs":{
                   "predicted_char_seq": predicted_char_seq, 
                   "predicted_writers": predicted_writers, 
                   "out_img": out, 
                   "out_predicted_char_seq": out_predicted_char_seq,
                   "out_predicted_writers": out_predicted_writers
                   },
               "losses": {
                   "loss_ctc": loss_ctc,
                   "loss_ce": loss_ce,
                   "loss_content": loss_c,
                   "loss_style": loss_s,
                   "out_loss_ctc": out_loss_ctc,
                   "out_loss_ce": out_loss_ce,
                   "loss_auto": loss_auto
                   }
               }
       return return_dict
       """
       return_dict = {
               "outputs":{
                   "predicted_writers": predicted_writers,
                   },
               "losses":{
                   "loss_ce": loss_ce,
                   }
               }
       return return_dict
       #"""


