class FontModel:

    def __init__(self):

        self.font2index = {}
        self.index2font = {}
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 3
        
        self.char2index['IPAD'] = 1024
        self.char2index['ISOS'] = 1025
        self.char2index['IEOS'] = 1026

        self.index2char[1024] = 'IPAD'
        self.index2char[1025] = 'ISOS'
        self.index2char[1026] = 'IEOS'
        self.n_fonts = 0

    def add_font_collection(self, font_collection):
        for font in font_collection:
            self.add_font(font)

    def add_font(self, font):

        if font not in self.font2index:
            self.font2index[font] = self.n_fonts
            self.index2font[self.n_fonts] = font
            self.n_fonts += 1

    def __call__(self, font_collection):
        self.add_font_collection(font_collection)
