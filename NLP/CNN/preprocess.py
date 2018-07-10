import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

def write_to_file(fn, testSentences, correctLabels, predLabels, idx2Label):
    thefile = open(fn, 'w')#''test.txt', 'w')
    for data, cLab, pLab in zip(testSentences, correctLabels, predLabels):
        #tokens, casing, char, labels = data
        #tokens, chars, labels = data
        tokens = [w[0] for w in data]
        assert len(tokens) == len(cLab)
        assert len(tokens) == len(pLab)
        for t, cL, pL in zip(tokens, cLab, pLab):
            thefile.write("%s  %s  %s \n" % (t, idx2Label[cL][:-1], idx2Label[pL][:-1]))
        thefile.write("\n\n")


def tag_dataset(dataset, model, Progbar):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])

        #pred = model.predict([tokens, casing,char], verbose=False)[0]

        if casing.shape[1]>0 and char.shape[1]>0:
            pred = model.predict([tokens, casing, char], verbose=False)[0]
        elif casing.shape[1] > 0 and char.shape[1] == 0:
            pred = model.predict([tokens, casing], verbose=False)[0]
        elif casing.shape[1] == 0 and char.shape[1] > 0:
            pred = model.predict([tokens, char], verbose=False)[0]
        elif casing.shape[1]==0 and char.shape[1]==0:
            pred = model.predict(tokens, verbose=False)[0]


        pred = pred.argmax(axis=-1) #Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

def romank_readfile(filename):
    #filename = "data/atis.txt"
    i = 0
    with open(filename, encoding="latin-1") as f:
        # i = 1;
        # for line in f: print(i); i += 1
        sentences = []
        sentence = []
        for line in f:
            i += 1;
            # if i > 10: break
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            splits = line.split(' ')
            if len(splits) == 4:
                sentence.append([splits[0], splits[-1]])
            else:
                print(len(splits), splits)
        if len(sentence) > 0:
            sentences.append(sentence)
            sentence = []
    return sentences

def get_trainSentences(fn):
    #fn = "data/train.txt"
    trainSentences = readfile(fn)
    trainSentences = addCharInformatioin(trainSentences)
    return trainSentences

def make_labels_words(trainSentences):#[trainSentences]: #, devSentences, testSentences]:
    labelSet = set()
    words = {}
    label2Idx = {}

    for dataset in trainSentences: #, devSentences, testSentences]:
        for sentence in dataset:
            for token,char,label in sentence:
                labelSet.add(label)
                #words[token.lower()] = True
                words[token.lower()] = words.get(token.lower(),0) + 1
    print(" #words=", len(words), " #labels=", len(labelSet))
    # :: Create a mapping for the labels ::
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    return labelSet, words, label2Idx

def get_char2Idx_case2Idx_word2Idx_wordEmbeddings(words):
    # :: Read in word embeddings ::
    # :: Hard coded case lookup ::
    case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                'contains_digit': 6, 'PADDING_TOKEN': 7}

    word2Idx = {}
    wordEmbeddings = []
    fEmbeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")

    notUsedEmbeddingWords = []
    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]

        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)

            #"DIGIT" #B-depart_time.time
            #"DIGITDIGIT" #B-arrive_time.time
            #"DIGITDIGITDIGIT" #B-depart_time.time
            #"DIGITDIGITDIGITDIGIT" #B - flight_number
            digit_word = "DIGIT"
            for d in range(4):
                word2Idx[digit_word] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                wordEmbeddings.append(vector)
                digit_word = digit_word + "DIGIT"
            numRomankWords = len(word2Idx)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)
        else:
            #print(" ... missing in Embedding:", split[0].lower())
            notUsedEmbeddingWords.append(split[0].lower())

    wordEmbeddings = np.array(wordEmbeddings)

    char2Idx = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|><":
        char2Idx[c] = len(char2Idx)

    return char2Idx, case2Idx, word2Idx, wordEmbeddings, numRomankWords, notUsedEmbeddingWords

def reduce_dims_with_TFIDF(trainSentences):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    #nltk.download('stopwords')
    #nltk.download('punkt')

    if False: # remove unfrequent?
        example_sent = list(words.keys())  # "This is a sample sentence, showing off the stop words filtration."
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(" ".join(example_sent))
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        print(" #len word2idx=", len(words), " len without stopwords=", filtered_sentence)

        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        print(vectorizer.fit_transform(corpus).todense())
        print(vectorizer.vocabulary_)


    from sklearn.feature_extraction.text import TfidfVectorizer

    text = [] # list of text documents
    for s in trainSentences: text.append(" ".join([w[0] for w in s]))

    vectorizer = TfidfVectorizer()# create the transform

    vectorizer.fit(text)# tokenize and build vocab

    print(" vectorizer.vocabulary_.__len__()=",vectorizer.vocabulary_.__len__())# summarize

    if False:  # encode document
        print(vectorizer.idf_)
        vector = vectorizer.transform(text)  # [text[0]])
        # summarize encoded vector
        print(vector.shape)
        print(vector.toarray())

    words = vectorizer.vocabulary_
    #stop_words = set(stopwords.words('english')) # shoud remove stop words?
    return words

#===========================
def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0],splits[-1]])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []
    return sentences

def getCasing(word, caseLookup):   
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
   
    return caseLookup[casing]
    
"""
def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len
"""
def createBatches(data, dataSentences = None):
    if None==dataSentences:
        dataSentences = data
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    batchSentences = []
    z = 0
    for i in l:
        for batch, s in zip(data, dataSentences):
            if len(batch[0]) == i:
                batches.append(batch)
                batchSentences.append(s)
                z += 1
        batch_len.append(z)
    return batches,batch_len,batchSentences

def createMatrices(sentences, word2Idx, label2Idx, case2Idx,char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
        
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        wordIndices = []    
        caseIndices = []
        charIndices = []
        labelIndices = []
        
        for word,char,label in sentence:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            wordIndices.append(wordIdx)

            if (0 < len(char2Idx)):
                charIdx = []
                for x in char:
                    assert x in char2Idx.keys()
                    if not x in char2Idx.keys():
                        print ("\n\n====ERRORERROR=x=",x)
                        char2Idx[x] = len(char2Idx)
                    charIdx.append(char2Idx[x])
                charIndices.append(charIdx)

            if (0 < len(case2Idx)):
                caseIndices.append(getCasing(word, case2Idx))

            # Get the label and map to int
            labelIndices.append(label2Idx[label])
           
        dataset.append([wordIndices, caseIndices, charIndices, labelIndices]) 
        
    return dataset#, char2Idx

def iterate_minibatches(dataset,batch_len): 
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,c,ch,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens),np.asarray(caseing),np.asarray(char)

def addCharInformatioin(Sentences):
    for i,sentence in enumerate(Sentences):
        for j,data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0],chars,data[1]]
    return Sentences

def padding(Sentences, maxlen=52, padding='post'):
    maxlenChar = 0
    for sentence in Sentences:
        char = sentence[2]
        for x in char:
            maxlenChar = max(maxlenChar ,len(x))
    maxlen = max(maxlenChar, maxlen)
    for i,sentence in enumerate(Sentences):
        #Sentences[i][2] = pad_sequences(Sentences[i][2],52,padding='post')
        assert Sentences[i][2] == sentence[2]
        Sentences[i][2] = pad_sequences(sentence[2], maxlen=maxlen, padding='post')
    return Sentences
