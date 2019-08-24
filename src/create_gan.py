import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.models import model_from_json
import pandas as pd
import chess

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

df = pd.read_csv("/home/tvincent/Dropbox/Research/Codes/python_examples/MachineLearning/ChessComposer/mateinonefolder/min1.csv", header=None, sep=";", names=["FEN"])
X_train = df["FEN"]
idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)
X_train = X_train[idx]

json_file = open('/home/tvincent/Dropbox/Research/Codes/python_examples/MachineLearning/ChessComposer/colab_cc_encoder/encoder_dense.json', 'r')
loaded_encoder_json = json_file.read()
json_file.close()
loaded_encoder = model_from_json(loaded_encoder_json)
# load weights into new model
loaded_encoder.load_weights('/home/tvincent/Dropbox/Research/Codes/python_examples/MachineLearning/ChessComposer/colab_cc_encoder/encoder_dense.h5')

json_file = open('/home/tvincent/Dropbox/Research/Codes/python_examples/MachineLearning/ChessComposer/colab_cc_encoder/decoder_dense.json', 'r')
loaded_decoder_json = json_file.read()
json_file.close()
loaded_decoder = model_from_json(loaded_decoder_json)
# load weights into new model
loaded_decoder.load_weights('/home/tvincent/Dropbox/Research/Codes/python_examples/MachineLearning/ChessComposer/colab_cc_encoder/decoder_dense.h5')

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

randomDim = 16
encoding_dim = 42

generator = Sequential()
generator.add(Dense(32, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(64))
generator.add(LeakyReLU(0.2))
generator.add(Dense(128))
generator.add(LeakyReLU(0.2))
generator.add(Dense(encoding_dim, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Dense(128, input_dim=encoding_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(64))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(32))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
# def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
#     noise = np.random.normal(0, 1, size=[examples, randomDim])
#     generatedImages = generator.predict(noise)
#     generatedImages = generatedImages.reshape(examples, 28, 28)

#     plt.figure(figsize=figsize)
#     for i in range(generatedImages.shape[0]):
#         plt.subplot(dim[0], dim[1], i+1)
#         plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
#         plt.axis('off')
#     plt.tight_layout()
    #plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    
    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for zzz in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            #imageBatch = X_train_tensor_encoded[np.random.randint(0, X_train.shape[0], size=batchSize)]
            imageBatch = X_train_tensor_encoded[np.random.randint(0, 1, size=batchSize)]
            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            #print(imageBatch.shape)
            #print(generatedImages.shape)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        #if e == 1 or e % 20 == 0:
            #plotGeneratedImages(e)
            #saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)


train(100, 64)

noise = np.random.normal(0, 1, size=[1, randomDim])
generatedImages = generator.predict(noise)
print(generatedImages)
decoded_pos = loaded_decoder.predict(generatedImages)
