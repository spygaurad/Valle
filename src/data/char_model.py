class CharModel:

    def __init__(self):

        self.char2index = {}
        self.index2char = {}
        self.n_chars = 3
        
        self.char2index['PAD'] = 0
        self.char2index['TSOS'] = 1
        self.char2index['TEOS'] = 2

        self.index2char[0] = 'PAD'
        self.index2char[1] = 'TSOS'
        self.index2char[2] = 'TEOS'

        self.char2index['IPAD'] = 1024
        self.char2index['ISOS'] = 1025
        self.char2index['IEOS'] = 1026

        self.index2char[1024] = 'IPAD'
        self.index2char[1025] = 'ISOS'
        self.index2char[1026] = 'IEOS'

        self.char2lm = []
        # self.special_tokens_in_vocab = ['TSOS', 'TEOS']
        # self.special_tokens_out_vocab = ['BLANK', 'PAD']

    def add_char_collection(self, char_collection):
        for char in char_collection:
            self.add_char(char)

        self.vocab_size = self.n_chars

        # for t in self.special_tokens_in_vocab:
        #     self.add_char(t)

        # for ot in self.special_tokens_out_vocab:
        #     self.add_char(ot)

        # self.vocab_size = self.n_chars - len(self.special_tokens_out_vocab)

    def add_char(self, char):

        if char not in self.char2index:
            # char_index = str.encode(char)[0] + 2
            char_index = self.n_chars
            self.char2index[char] = char_index
            self.index2char[char_index] = char
            self.n_chars += 1

    def char2lm_mapping(self, index, char):
        # return index
        if index > 3 and index < 1000:
            new_index = str.encode(char)
            if len(new_index) == 1:
                return new_index[0] + 2
            else:
                print(index, char)
                raise ValueError("Cannot convert character to token")
            
        else:
            return index

    def __call__(self, char_collection):
        self.add_char_collection(char_collection)

        for i, c in self.index2char.items():
            self.char2lm.append(self.char2lm_mapping(i, c))
            if i>=93:
                break


    def indexes2characters(self, indexes, ctc_mode=False):
        characters = []
        if not ctc_mode:
            for i in indexes:
                characters.append(self.index2char[i])
            return characters
        else:
            for j, i in enumerate(indexes):
                if i == self.n_chars:
                    continue
                if characters and indexes[j-1] == i:
                    continue
                characters.append(self.index2char[i])
            return characters

