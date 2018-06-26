# romankazinnik_blog

code for romankazinnik.com blog

1. Convolutional Deep Learning: insights on multiscale representation

CNN: multiscale vs. redundant representation code for 
my blog: https://www.romankazinnik.com/feed/convolutional-deep-learning-insights-on-multiscale-representation


    1.0 I construct three multiscale CNNs to illustrate redundant network and how this problem is solved.

    1.1 download
The German Traffic Sign Recognition Benchmark(GTSRB) http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

GTSRB

├── GT-final_test.csv

├── Final_Test

    └── Images

└── Final_Training

    └── Images
    
        ├── 00000
        
        ├── ...

    1.2 python trafficsignscnn1vs2vs3.py
    
2. NLP comparative analysis of words embedding vs. bag of words, w/o characters  embedding trained with CNN, w/o capital cases embedding

    2.0 python nametagging_cnn_rnn.py
    
    2.1 two training examples: small input (<500 words) vs. large input (> 15K words)
    
    2.2 Conclusion 1: notice when you don't need [low dimensional] word embedding 
    
    2.5 Conclusion 2: notice when you don't need to train for letters embedding
    
  
