from torchtext.data.metrics import bleu_score

from src.config.handwriting_config import BLEU_SCORE_NGRAM, BLEU_SCORE_WEIGHTS


def calculate_BLEU_score(char_model, reference_corpus, candidate_corpus):

    eos_index = str(char_model.char2index["EOS"])

    reference_corpus = [[[str(i.item()) for i in x]] for x in reference_corpus]
    reference_corpus = [[i[:i.index(eos_index) + 1]] if eos_index in i else [i[:]] for x in reference_corpus for i in x]

    _, candidate_corpus = candidate_corpus.topk(1)
    candidate_corpus = candidate_corpus.squeeze(-1)
    candidate_corpus = [[str(i.item()) for i in x] for x in candidate_corpus]
    candidate_corpus = [x[:x.index(eos_index) + 1] if eos_index in x else x[:] for x in candidate_corpus]

    score = bleu_score(candidate_corpus, reference_corpus, max_n=BLEU_SCORE_NGRAM, weights=BLEU_SCORE_WEIGHTS)

    return score
