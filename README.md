# Week 8: Encoder-Decoder Architecture for Sequence-to-Sequence Tasks
Encoder-Decoder architecture is a type of model used in many sequence-to-sequence tasks in Natural Language Processing (NLP). These tasks, as the name suggests, involve converting an input sequence into an output sequence, and include machine translation, text summarization, and question answering among others.

How does the Encoder-Decoder Architecture Work?
The Encoder-Decoder architecture is made up of two main components: the encoder and the decoder, both of which are often implemented as Recurrent Neural Networks (RNNs) or more advanced variants like LSTMs (Long Short-Term Memory) or GRUs (Gated Recurrent Units).

Encoder: The encoder processes the input sequence and summarizes the information in its internal state, often referred to as the context or thought vector. This vector is expected to be a good representation of the entire input sequence.

Decoder: The decoder generates the output sequence step-by-step based on the context vector provided by the encoder and what it has generated so far. In other words, the decoder is an RNN that generates probabilities for the output sequence. For each step, the decoder is influenced by the context vector and the previously generated words.

While the simplest versions of the encoder and decoder share the same weights, more complex versions may use different parameters for the two components. The architecture is flexible and allows for many variations, such as the use of multiple layers of encoders and decoders.

Applications of the Encoder-Decoder Architecture
A classic example of the use of an Encoder-Decoder architecture is in machine translation, where the input sequence is a sentence in the source language, and the output sequence is the corresponding translation in the target language. The encoder transforms the source sentence into a context vector, and then the decoder generates the target sentence based on this vector.

Another example is text summarization, where the input is a long document, and the output is a shorter summary. The encoder reads the entire document and encodes it into a context vector. The decoder then uses this vector to generate a brief summary.

Challenges and Limitations
While Encoder-Decoder architectures have proven successful for many tasks, they are not without their limitations. One key issue is that the encoder must compress all the necessary information of the input sequence into a fixed-size context vector, which can lead to information loss, especially for longer sequences.

To overcome this limitation, attention mechanisms have been introduced, which allow the model to focus on different parts of the input sequence at each step of the output generation, instead of encoding the entire sequence into a single context vector. This has been a significant improvement and is now a standard component of many sequence-to-sequence models.

# Readings

[Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction](https://arxiv.org/pdf/1604.04677.pdf)


[Understanding How Encoder-Decoder Architectures Attend](https://proceedings.neurips.cc/paper_files/paper/2021/file/ba3c736667394d5082f86f28aef38107-Paper.pdf)


[Evaluating Sequence-to-Sequence Models for Handwritten Text Recognition](https://arxiv.org/pdf/1903.07377.pdf)


# Code example

Step 1: Data Preparation

In the beginning, we define a dummy English to French translation dataset (TranslationDataset). It contains 20 training pairs, where each pair consists of an English sentence and its corresponding French translation. The dataset is then converted into PyTorch tensors.

Step 2: Building the Encoder

We define the Encoder class, representing the encoder part of the Encoder-Decoder model. The encoder has an embedding layer to convert input words into dense vectors and a GRU layer (Gated Recurrent Unit) to process the input sequence. The encoder processes each word in the input sequence and produces a fixed-size hidden state representing the entire sequence's information.

Step 3: Building the Decoder

Next, we define the Decoder class, representing the decoder part of the Encoder-Decoder model. The decoder also has an embedding layer and a GRU layer, similar to the encoder. Additionally, it includes a linear layer to transform the decoder's hidden state into output probabilities for each word in the output vocabulary.

Step 4: Building the Encoder-Decoder Model

We create the EncoderDecoder class, which combines the encoder and decoder into a single model. It takes an input sequence (English sentence) and produces an output sequence (French translation). During training, the model uses teacher forcing, where the true output sequence is fed as input during training, but during inference (translation), the predicted output from the previous timestep is used as input.

Step 5: Training the Model

We define the loss function (CrossEntropyLoss) and the optimizer (Adam) for training. We then run a loop for a specified number of epochs to train the model. In each epoch, we pass the input sequence through the encoder and the output sequence through the decoder. We compute the loss by comparing the predicted output with the ground truth output (French translation). The model's parameters are updated using backpropagation and the Adam optimizer.

Step 6: Translating New Sentences

After training, we define a function (translate_sentence) to translate new English sentences into French. The function takes an English sentence, encodes it using the trained encoder, and then decodes it using the trained decoder. The decoder generates the French translation one word at a time until it predicts an end-of-sequence token.

Step 7: Test Translation

Finally, we test the trained model by providing a new English sentence ("We are happy") and obtain its French translation using the translate_sentence function. The translated French sentence is then displayed.