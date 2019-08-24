import pandas as pd
import numpy as np
import chess
import os
#os.listdir("/content/drive/My Drive/fics_fen_2_2M_rnn.csv")
df = pd.read_csv("/content/drive/My Drive/fics_and_min1.csv", header=None, sep=";", names=["FEN"])
X_data = df["FEN"]

X_train = X_data[:2048000]
X_test =  X_data[2048000:]

idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)
X_train = X_train[idx]

def fen_to_tensor(inputbatch):
    
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4,
                    pieces_str[4]:5, pieces_str[5]:6,
                    pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4, 
                    pieces_str[10]:-5, pieces_str[11]:-6}

    maxnum = len(inputbatch)
    boardtensor = np.zeros((maxnum, 8, 8,7))
    
    for num, inputstr in enumerate(inputbatch):
        inputliste = inputstr.split()
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputliste[0]):
            if c in pieces:
                boardtensor[num, rownr, colnr, np.abs(pieces_dict[c])-1] = np.sign(pieces_dict[c])
                colnr = colnr + 1
            elif c == '/':  # new row
                rownr = rownr + 1
                colnr = 0
            elif int(c) in valid_spaces:
                colnr = colnr + int(c)
            else:
                raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
        
        if inputliste[1] == "w":
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = 1
        else:
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = -1
  
    return boardtensor


def tensor_to_fen(inputtensor):
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    
    maxnum = len(inputtensor)
    
    outputbatch = []
    for i in range(maxnum):
        fenstr = ""
        for rownr in range(8):
            spaces = 0
            for colnr in range(8):
                for lay in range(6):                    
                    if inputtensor[i,rownr,colnr,lay] == 1:
                        if spaces > 0:
                            fenstr += str(spaces)
                            spaces = 0
                        fenstr += pieces_str[lay]
                        break
                    elif inputtensor[i,rownr,colnr,lay] == -1:
                        if spaces > 0:
                            fenstr += str(spaces)
                            spaces = 0
                        fenstr += pieces_str[lay+6]
                        break
                    if lay == 5:
                        spaces += 1
            if spaces > 0:
                fenstr += str(spaces)
            if rownr < 7:
                fenstr += "/"
        if inputtensor[i,0,0,6] == 1:
            fenstr += " w"
        else:
            fenstr += " b"
        outputbatch.append(fenstr)
    
    return outputbatch


from keras.layers import Input, Dense
from keras.models import Model

encoder_input = Input(shape=(448,))
encoded0 = Dense(300, activation='tanh')(encoder_input)
encoded1 = Dense(200, activation='tanh')(encoded0)
encoded2 = Dense(150, activation='tanh')(encoded1)
encoded3 = Dense(42, activation='tanh')(encoded2)

decoder_input = Input(shape=(42,))
decoded0 = Dense(150, activation='tanh')(encoded3)
decoded1 = Dense(200, activation='tanh')(decoded0)
decoded2 = Dense(300, activation='tanh')(decoded1)
decoded3 = Dense(448, activation='tanh')(decoded2)

autoencoder = Model(encoder_input, decoded3)
autoencoder.summary()

from keras import optimizers
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
autoencoder.compile(optimizer=opt, loss='mean_squared_error')

def myGenerator():
    while 1:
        for i in range(16000): # 16000 * 128 = 2048000 -> # of training samples
            ret = fen_to_tensor(X_train[i*128:(i+1)*128])
            ret = ret.reshape((128, np.prod(ret.shape[1:])))
            yield (ret, ret)

my_generator = myGenerator()
testtensor = fen_to_tensor(X_test)
validdata = testtensor.reshape((len(X_test), np.prod(testtensor.shape[1:])))

history = autoencoder.fit_generator(my_generator, steps_per_epoch = 16000, epochs = 100, verbose=1, 
              validation_data=(validdata,validdata))

input_pos = Input(shape=(448,))

encoder_layer1 = autoencoder.layers[1]
encoder_layer2 = autoencoder.layers[2]
encoder_layer3 = autoencoder.layers[3]
encoder_layer4 = autoencoder.layers[4]

encoded = encoder_layer1(input_pos)
encoded = encoder_layer2(encoded)
encoded = encoder_layer3(encoded)
encoded = encoder_layer4(encoded)

encoder = Model(input_pos, encoded)

encoding_dim = 42  
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer1 = autoencoder.layers[-4]
decoder_layer2 = autoencoder.layers[-3]
decoder_layer3 = autoencoder.layers[-2]
decoder_layer4 = autoencoder.layers[-1]

decoded = decoder_layer1(encoded_input)
decoded = decoder_layer2(decoded)
decoded = decoder_layer3(decoded)
decoded = decoder_layer4(decoded)
# create the decoder model
decoder = Model(encoded_input, decoded)

model_json = autoencoder.to_json()
with open("/content/drive/My Drive/autoencoder_fics_and_min1_dense.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("/content/drive/My Drive/autoencoder_fics_and_min1_dense.h5")

model_json = encoder.to_json()
with open("/content/drive/My Drive/encoder_fics_and_min1_dense.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder.save_weights("/content/drive/My Drive/encoder_fics_and_min1_dense.h5")

model_json = decoder.to_json()
with open("/content/drive/My Drive/decoder_fics_and_min1_dense.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
decoder.save_weights("/content/drive/My Drive/decoder_fics_and_min1_dense.h5")
