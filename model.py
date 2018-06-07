import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from collections import Counter
from keras.layers import Embedding,Input, LSTM, Dense, Conv1D, Conv2D, MaxPool2D, MaxPooling1D, Dropout, Activation, Reshape, Concatenate, Flatten
from keras.models import Sequential, model_from_json
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils



# read the input
f = open("NLG_In.txt",'r')
Input = f.read()
Input = Input.split("\n")
Input_len = len(Input)
Input_data = {}
# seperating label and data

InputData = np.array([['                                            ' for _ in range(21)] for _ in range(len(Input)-1)])
XData = np.array([[0 for _ in range(21)] for _ in range(len(Input)-1)])

InputContent = np.array([['                                           ' for _ in range(21)] for _ in range(len(Input)-1)])
XContent = np.array([[0 for _ in range(21)] for _ in range(len(Input)-1)])

tokenizerlabel = Tokenizer(num_words=20000)
tokenizerlabel.fit_on_texts(Input)
sequenceslabel = tokenizerlabel.texts_to_sequences(Input)

tokenizerlabel.word_index[''] = len(tokenizerlabel.word_index) + 1

for j in range(0,len(Input)-1):
    b = Input[j].split(",")
    for k in range(0,len(b)):
       Input_data = b[k].split(":")
       InputData[j][k] = Input_data[0].lower()
       XData[j][k] = tokenizerlabel.word_index[InputData[j][k]]
       InputContent[j][k] = Input_data[1].lower()
       XContent[j][k] = tokenizerlabel.word_index[InputContent[j][k]]

f = open("NLG_Out.txt",'r')
Output = f.read()
Output = Output.split("\n")
x = 0
for i in range(len(InputData)):
    if(x < len(Output[i].split(" "))):
          x = len(Output[i].split(" "))


tokenizerlabel1 = Tokenizer(num_words=20000)
tokenizerlabel1.fit_on_texts(Output)
sequenceslabel1 = tokenizerlabel1.texts_to_sequences(Output)
tokenizerlabel1.word_index[''] = len(tokenizerlabel1.word_index) + 1

InputLabel = np.array([['                                           ' for _ in range(x)] for _ in range(len(Output))])
XLabel = np.array([[0 for _ in range(x)] for _ in range(len(Output))])
for j in range(0,len(Output)):
    b = Output[j].split(" ")
    for k in range(0,len(b)):
         InputLabel[j][k] = b[k].lower()
         XLabel[j][k] = tokenizerlabel1.word_index[InputLabel[j][k]]



X_train = np.concatenate([XData.T,XContent.T])
X_train = X_train.T
Y_train = XLabel




'''sequence_length = 30
vocabulary_size = 20000
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)'''
'''model.add(Embedding(20000, 128, input_length=20))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='softmax'))'''

# Built a CNN-LSTM model using Keras'''

vocabulary_size=20000
embedding_dim = 128
model = Sequential()
model.add(Embedding(vocabulary_size,embedding_dim, input_length=len(X_train[0])))
model.add(Dropout(0.2))
model.add(Conv1D(128, 10, activation='relu'))
model.add(MaxPooling1D(pool_size=6))
model.add(LSTM(64))
model.add(Dense(len(Y_train[0]), activation='softmax'))
'''print(model.summary())'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,Y_train,shuffle=True,batch_size=64,validation_split=0.40,epochs=50,verbose=True)

a = model.predict(X_train)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


