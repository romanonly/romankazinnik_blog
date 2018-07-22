import numpy as np 
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform

from matplotlib import pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer

import random
import itertools
import copy
import numpy

from validation import compute_f1
from preprocess import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding,\
    write_to_file,tag_dataset,romank_readfile,get_trainSentences,make_labels_words,\
    get_char2Idx_case2Idx_word2Idx_wordEmbeddings, reduce_dims_with_TFIDF



if __name__ == "__main__":

    testDataCase = 3

    if testDataCase > 0:

        trainSentences = romank_readfile("data/train2.txt")
        trainSentences = addCharInformatioin(trainSentences)

        if testDataCase == 1:
            devSentences = copy.deepcopy(trainSentences )#romank_readfile("data/atis.txt")
            testSentences = copy.deepcopy(trainSentences ) #romank_readfile("data/atis.txt")
        elif testDataCase == 2:
            devSentences = romank_readfile("data/dev2.txt")
            devSentences = addCharInformatioin(devSentences )
            testSentences = romank_readfile("data/test2.txt")
            testSentences = addCharInformatioin(testSentences )
        else:
            indices = numpy.random.permutation(len(trainSentences)) #.shape[0])
            indices_train_len = int( numpy.floor(len(indices) * 0.8) )
            training_idx, test_idx = indices[:indices_train_len], indices[indices_train_len:]
            training = [trainSentences[i] for i in training_idx]
            test     = [trainSentences[i] for i in test_idx]

            trainSentences = training
            devSentences    = test
            testSentences = romank_readfile("data/test_atis.txt")
            testSentences = addCharInformatioin(testSentences )

    else:
        trainSentences = get_trainSentences("data/train.txt")
        devSentences = get_trainSentences("data/valid.txt")
        testSentences = get_trainSentences("data/test.txt")


    #  Model setting:
    #
    #  Words embedding
    #  (1) bag of words (2) tfidf (3) google GloVe emeddings
    isUseTFIDFvsGOOGLE = False
    isUseTFIDFvsGOOGLE_LowDim = 99999 # large number if not interested in dim reduction with hashing

    # get labels, words used in sentences
    labelSet, words, label2Idx = make_labels_words([trainSentences, devSentences, testSentences])

    idx2Label = {v: k for k, v in label2Idx.items()}
    print (" #labels=", len(labelSet), " #words=", len(words), " words=",list( words.items() )[0])

    if (isUseTFIDFvsGOOGLE): # GOOGLE vs. Hot-Key Embedding
        if isUseTFIDFvsGOOGLE_LowDim < len(words): # hashing
            wordsTFIDF = reduce_dims_with_TFIDF(trainSentences)
            removedWords = set(words.keys()) - set(wordsTFIDF .keys)
            print(" removedWords=",removedWords)
            words = wordsTFIDF
            print(" isUseTFIDFvsGOOGLE #words=", len(words), " words=", list(words.items())[0])

    # Each word is a list of characters, plus cases, GloVe-word-embeddings, plus special-words
    char2Idx, case2Idx, word2Idx, wordEmbeddings, numRomankWords, notUsedEmbeddingWords\
        = get_char2Idx_case2Idx_word2Idx_wordEmbeddings(words)

    assert "DIGITDIGIT" in set(list(word2Idx.keys()))
    print(" #embedded words=", len(word2Idx), " #notUsedEmbeddingWords=", len(notUsedEmbeddingWords))
    wordsNotInGoogle = set(words) - set(word2Idx.keys())
    print ("len(words)=", len(words), " #embedded words=", len(word2Idx), " len wordsNotInGoogle =", len(wordsNotInGoogle))
    assert(len(wordsNotInGoogle) + len(word2Idx) - len(words) == numRomankWords)

    #========== one-hot v. TFIDF vs. hash
    if isUseTFIDFvsGOOGLE:  # hashing to low-dims
        if isUseTFIDFvsGOOGLE_LowDim < len(word2Idx):
            for k, v in word2Idx.items():
                word2Idx[k] = v % isUseTFIDFvsGOOGLE_LowDim
            wordEmbeddings = np.identity(isUseTFIDFvsGOOGLE_LowDim, dtype='float32')
        else:
            wordEmbeddings = np.identity(len(word2Idx), dtype='float32')  # wordEmbeddings_OneHot

    #======================================
    #
    # Set Bi-LSTM:
    # will each word include in addtion to word embedding also (1) case indices (2) char indices
    #
    epochs = 50
    case2Idx,char2Idx, maxlen = {}, {}, 0
    #case2Idx,maxlen = {}, 52

    #caseEmbeddings = None
    if len(case2Idx)>0:
        caseEmbeddings = np.identity(len(case2Idx), dtype='float32')


    # create input matrices: each word is a vector
    train_set = createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx)

    train_set = padding(train_set, maxlen = maxlen)
    dev_set = padding(createMatrices(devSentences,word2Idx, label2Idx, case2Idx,char2Idx), maxlen = maxlen)
    test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx), maxlen = maxlen)


    # batches of sentences of equal lengths
    train_batch,train_batch_len, train_batch_sentrences = createBatches(train_set, trainSentences)
    dev_batch,dev_batch_len, dev_batch_sentrences = createBatches(dev_set, devSentences)
    test_batch,test_batch_len, test_batch_sentences= createBatches(test_set, testSentences)

    # Keras: Bi-directional LSTM
    words_input = Input(shape=(None,),dtype='int32',name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)

    if len(case2Idx)>0:
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)

    if len(char2Idx) > 0:
        character_input=Input(shape=(None,52,),name='char_input')
        embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
        dropout= Dropout(0.5)(embed_char_out)
        conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
        char = TimeDistributed(Flatten())(maxpool_out)
        char = Dropout(0.5)(char)

    if len(case2Idx)>0 and len(char2Idx)>0:
        output = concatenate([words, casing,char])
    elif len(case2Idx) > 0 and len(char2Idx) == 0:
        output = concatenate([words, casing])
    elif len(case2Idx)== 0 and len(char2Idx)==0:
        output = words
    elif len(case2Idx) == 0 and len(char2Idx) > 0:
        output = concatenate([words, char])


    #
    # No overfit
    #
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)

    # 0.66 vs 0.75 overfit after 20 epochs
    #output = Bidirectional(LSTM(50, return_sequences=True, dropout=0.65, recurrent_dropout=0.50))(output)

    # 0.7 after 20 epochs
    #output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.3, recurrent_dropout=0.25))(output)#, stateful=False

    #output = Bidirectional(LSTM(10, return_sequences=True, dropout=0.3, recurrent_dropout=0.25))(output)  #accuracy f1 0.5 after 20 epochs , stateful=False

    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)

    # Three models: w/o characters CNN embedding, w/o casing input
    #model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
    if len(case2Idx)>0 and len(char2Idx)>0:
        model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
    elif len(case2Idx) > 0 and len(char2Idx) == 0:
        model = Model(inputs=[words_input, casing_input], outputs=[output])
    elif len(case2Idx)== 0 and len(char2Idx)==0:
        model = Model(inputs=words_input, outputs=[output])
    elif len(case2Idx) == 0 and len(char2Idx) > 0:
        model = Model(inputs=[words_input, character_input], outputs=[output])


    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()
    # plot_model(model, to_file='model.png')
    train_loss, test_loss=[], []

    plt.ion();  plt.figure(); #plt.xlim(0, 10); plt.ylim(0, 100)
    plt.hold(True)


    for epoch in range(epochs):
        print("Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(train_batch_len))

        train_batches = iterate_minibatches(train_batch, train_batch_len)
        dev_batches = iterate_minibatches(dev_batch, dev_batch_len)

        for i,batch_zip in enumerate(zip(train_batches, dev_batches)):
            labels, tokens, casing,char = batch_zip[0]
            labels0, tokens0, casing0, char0 = batch_zip[1]

            if casing.shape[1]>0 and char.shape[1]>0:
                #model.train_on_batch([tokens, casing,char], labels)
                x = [tokens, casing, char]
                x0= [tokens0, casing0, char0]
            elif casing.shape[1] > 0 and char.shape[1] == 0:
                #model.train_on_batch([tokens, casing], labels)
                x = [tokens, casing]
                x0= [tokens0, casing0]
            elif casing.shape[1] == 0 and char.shape[1] > 0:
                #model.train_on_batch([tokens, char], labels)
                x = [tokens, char]
                x0= [tokens0, char0]
            elif casing.shape[1]==0 and char.shape[1]==0:
                #model.train_on_batch(tokens, labels)
                x = tokens
                x0= tokens0


            temp1=model.train_on_batch(x, labels)
            train_loss.append(temp1) #, test_loss = [], [] #model.reset_states()
            temp2=model.test_on_batch(x0,labels0)
            test_loss.append(temp2) #model.reset_states()

            if epoch % 10 == 0 and i % 20 == 0:
                a.update(i)

        plt.plot(test_loss, color="black"); plt.plot(train_loss, color="red");
        if epoch % 5 == 0:
            # Performance on train dataset
            d_batch0 = random.choices(population=train_batch, k=10)
            predLabels0, correctLabels0 = tag_dataset(d_batch0, model,Progbar)
            pre_dev0, rec_dev0, f1_dev0 = compute_f1(predLabels0, correctLabels0, idx2Label)
            # Performance on dev dataset
            d_batch = random.choices(population=dev_batch, k=10)
            predLabels, correctLabels = tag_dataset(d_batch, model, Progbar)
            pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
            print("\n *** Dev   Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
            print("\n *** Train Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev0, rec_dev0, f1_dev0))

        print(' ========= epoch ',epoch)

    plt.hold(False)
    #plt.ioff();
    plt.figure();
    plt.plot(test_loss, color="black");
    plt.plot(train_loss, color="red");
    plt.title("loss train(red) test");
    plt.show()
    #   Performance on train dataset
    #predLabels, correctLabels = tag_dataset(train_batch, model,Progbar)
    #pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
    #print("\nTrain-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

    #   Performance on dev dataset
    #predLabels, correctLabels = tag_dataset(dev_batch, model,Progbar)
    #pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
    #print("\nDev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))

    #   Performance on test dataset
    predLabels, correctLabels = tag_dataset(test_batch, model,Progbar)
    pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
    print("\nTest-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

    write_to_file("data/result_test_attis_50epochs_cnn_nochars.txt", test_batch_sentences, correctLabels, predLabels, idx2Label)