import fasttext

f = fasttext.load_model("../models/fasttext/gruene_ws3_neg7_dim400_ep700_loss104.bin")
words = f.get_words()
print(str(len(words)) + " " + str(f.get_dimension()))
for w in words:
    v = f.get_word_vector(w)
    vstr = ""
    for vi in v:
        vstr += " " + str(vi)

    print(w + vstr)
