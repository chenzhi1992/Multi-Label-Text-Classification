# Multi-Label-Text-Classification
Tensorflow+bilstm+attention+multi label text classify (support Chinese text)

#Network:

  Word Embedding + bi-lstm + attention + Variable batch_size
  
#Requirements

  Python 3.5 (> 3.0)
  
  Tensorflow 1.2

#Introduction
 
   1. This is a multi-label text classification (sentence classification) problem. For example, the format of label is [0,1,0,1,1].
   2. This model was built with bi-lstm, attention and Word Embeddings(word2vec) on Tensorflow.
   3. It supports the variable batch size.(the batch size of test code(prediction) is 1)
   
    (在训练和测试时,每个epoch样本被分成很多batches,最后一个batch的size小于batch_size时也是可以去训练和测试的,不用舍弃这些样本)
   
   4. It supports Chinese text classification, but you need the pretrained word2vector model.
   
    (通过word2vector训练中文的词向量)
   
   5. input_helpers.py is my data processing code , and you can write it according to yourself dataset.
   
    (根据自己的数据集来写input_helpers.py, 将数据集写到trainset 和devset两个变量即可,trainset 和devset中包括所有样本的数据和对应的label)

#python file

Lstm_Attention_network.py: define the network

My_Attention.py: attention code

train.py: train the network

input_helpers.py : data helper code , data processing


注: 欢迎指正讨论.
