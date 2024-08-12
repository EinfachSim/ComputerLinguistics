
#SOURCE: https://fasttext.cc/docs/en/english-vectors.html, last accessed 12.08.2024, 14:11
import io
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    vocab = {}
    word_vectors = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        vocab[tokens[0]] = len(word_vectors)
        word_vectors.append(np.array(list(map(float, tokens[1:]))))
    return vocab, np.array(word_vectors)

def rebuild(common_words, unique_words, vectors, vocab):
    new_vocab = {}
    new_vectors = []
    words = common_words + unique_words
    for word in words:
        word_idx = vocab[word]
        v = vectors[word_idx, :]
        new_vocab[word] = len(new_vectors)
        new_vectors.append(v)
    return new_vocab, np.array(new_vectors), words

def to_txt(words, vectors):
    result = str(len(words)) + " " + str(400) + "\n" 
    for i in range(len(words)):
        w = words[i]
        v = vectors[i]
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        result += w + vstr + "\n"
    return result

def save_to_file(string, path):
    with open(path, "w") as f:
        f.write(string)

model_1_vocab, model_1_vectors = load_vectors("../models/fasttext/afd.vec")
model_2_vocab, model_2_vectors = load_vectors("../models/fasttext/gruene.vec")

words1 = set(model_1_vocab.keys())
words2 = set(model_2_vocab.keys())
corresponding_words = list(words1 & words2)
unique_words_1 = list(words1.difference(corresponding_words))
unique_words_2 = list(words2.difference(corresponding_words))

model_1_vocab, model_1_ordered_vectors, words_1 = rebuild(corresponding_words, unique_words_1, model_1_vectors, model_1_vocab)
model_2_vocab, model_2_ordered_vectors, words_2 = rebuild(corresponding_words, unique_words_2, model_2_vectors, model_2_vocab)

save_to_file(to_txt(words_1, model_1_ordered_vectors), "../models/fasttext/afd.vec")
save_to_file(to_txt(words_2, model_2_ordered_vectors), "../models/fasttext/gruene.vec")
