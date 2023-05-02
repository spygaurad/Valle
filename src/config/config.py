import json


class Config:
    def __init__(self, **kwargs):

        self.batch_size = kwargs.pop('batch_size', 32)
        self.max_char_len = kwargs.pop('max_char_len', 128)
        self.shuffle = kwargs.pop('shuffle', True)
        self.num_workers = kwargs.pop('num_workers', 12)
        self.lr = kwargs.pop('lr', 0.01)
        self.epoch = kwargs.pop('epoch', 1)
        print(f"Number of epochs: {self.epoch}")
        self.char_embedding_dim = kwargs.pop('char_embedding_dim', 64)

        self.transformer_encoder = kwargs.pop('transformer_encoder', {})
        self.transformer_decoder = kwargs.pop('transformer_decoder', {})

        self.model_eval_epoch = kwargs.pop('model_eval_epoch', 5)
        self.eval_gen_data_length = kwargs.pop('eval_gen_data_length', 10)

    @classmethod
    def from_json_file(cls, json_file):

        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)

    def update_batch_size(self, device_count):
        if device_count:
            self.batch_size *= device_count
