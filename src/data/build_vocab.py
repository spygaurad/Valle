import argparse

import pandas as pd


def build_char_vocab_from_corpus(data_file, vocab_file):
    df = pd.read_csv(data_file)
    char_collection = set(df['text'].str.cat(sep=' '))
    char_collection = sorted(list(char_collection))

    with open(vocab_file, "w") as f:
        f.write("\n".join(char_collection))


def build_font_vocab_from_corpus(data_file, vocab_file):
    df = pd.read_csv(data_file) #, dtype=str)
    font_collection = set(df['font_type'].tolist())
    font_collection = sorted(font_collection)
    font_collection = [str(i) for i in font_collection]
    print(font_collection)

    with open(vocab_file, "w") as f:
        f.write("\n".join(font_collection))


def get_args():

    args = argparse.ArgumentParser()

    args.add_argument('--type',
                      required=True,
                      type=str,
                      choices=['char_vocab', 'font_vocab'],
                      help='Extract char vocab or font vocab')

    args.add_argument('--data_file',
                      required=True,
                      type=str,
                      help='Source from which we have to extract vocab')

    args.add_argument('--vocab_file',
                      required=True,
                      type=str,
                      help='File name in which we have to dump vocab')

    return args.parse_args()


def main(args):

    if args.type == 'char_vocab':
        build_char_vocab_from_corpus(args.data_file, args.vocab_file)

    elif args.type == 'font_vocab':
        build_font_vocab_from_corpus(args.data_file, args.vocab_file)


if __name__ == "__main__":
    args = get_args()
    main(args)
