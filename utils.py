from keras.preprocessing.text import Tokenizer
from os import listdir

def load_doc(filename):
    file = open(filename, 'r')
    # Intput and output are seperated by a new line
    # Read the document into one line and split it by new line, exept for the final new line
    text = file.read()[:-1].lower().split('\n')
    # Even numbers are input and odd are output
    x = text[0:][::2]
    y = text[1:][::2]
    file.close()
    return x, y


def load_folders(base, folders):
    
    x, y = [], []
    for folder in folders:
        filenames = listdir(base + folder)
        filenames.sort()
        for filename in filenames:
            nx, ny = load_doc(base + folder + filename)
            x += nx
            y += ny
    return x, y

def create_tokenizer(data):
    t = Tokenizer(filters='\n', lower=True, char_level=True)
    t.fit_on_texts(data)
    return t

def tokenize(sentences, vocab, isTgt):

        tokenized = []

        for line in sentences:

            if isTgt:
                line = "$" + line + "#"
            new_line = [vocab[tok] for tok in line]
        
            tokenized.append(new_line)
        
        return tokenized

def create_padding(tokens, max_len):

        padded_tokens = []

        for line in tokens:
            canvas = [0] * max_len
            canvas[:len(line)] = line
            padded_tokens.append(canvas)
        
        return padded_tokens