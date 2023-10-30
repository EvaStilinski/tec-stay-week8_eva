import torch
import torch.nn as nn
import torch.optim as optim
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.optim.lr_scheduler import StepLR

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # Remove leading and trailing punctuation
        word = word.strip(string.punctuation)

        # Remove trailing periods
        word = word.rstrip('.')

        # Remove punctuation inside the word and convert to lowercase
        word = ''.join(char.lower() for char in word if char.isalnum() or char.isspace())

        if word:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

def indexesFromSentence(lang, sentence):
    """
    Get word indexes from a sentence.

    Parameters:
    - lang (Lang): Language object.
    - sentence (str): Input sentence.

    Returns:
    - list: List of word indexes.
    """
    words = [word.rstrip('.') for word in sentence.split(' ')]
    return [lang.word2index[word.lower()] for word in words if word.lower() in lang.word2index]

def tensorFromSentence(lang, sentence):
    """
    Convert a sentence to a PyTorch tensor.

    Parameters:
    - lang (Lang): Language object.
    - sentence (str): Input sentence.

    Returns:
    - torch.Tensor: PyTorch tensor.
    """
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)  # Add EOS_token at the end of the sentence
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

# Define your SOS_token and EOS_token
SOS_token = 0
EOS_token = 1

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, dropout_p=0.7):
        """
        Initialize the Encoder.

        Parameters:
        - input_size (int): Input size.
        - hidden_size (int): Hidden size.
        - num_layers (int): Number of layers (default: 3).
        - dropout_p (float): Dropout probability (default: 0.7).
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p)

    def forward(self, input):
        """
        Forward pass of the Encoder.

        Parameters:
        - input: Input tensor.

        Returns:
        - tuple: Output tensor and hidden state.
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=3, dropout_p=0.7):
        """
        Initialize the Decoder.

        Parameters:
        - hidden_size (int): Hidden size.
        - output_size (int): Output size.
        - num_layers (int): Number of layers (default: 3).
        - dropout_p (float): Dropout probability (default: 0.7).
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        Forward pass of the Decoder.

        Parameters:
        - input: Input tensor.
        - hidden: Hidden state.

        Returns:
        - tuple: Output tensor and hidden state.
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded.view(1, 1, -1), hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

def evaluate(encoder, decoder, sentence, max_length=20):
    """
    Evaluate the seq2seq model.

    Parameters:
    - encoder: Encoder model.
    - decoder: Decoder model.
    - sentence (str): Input sentence.
    - max_length (int): Maximum length of the output (default: 20).

    Returns:
    - str: Translated output sentence.
    """
    with torch.no_grad():
        # Preprocess the input
        input_tensor = tensorFromSentence(input_lang, sentence)
        
        # Encoder forward pass
        encoder_output, encoder_hidden = encoder(input_tensor)

        # Decoder input
        decoder_input = torch.tensor([[SOS_token]])  # SOS_token marks the start of decoding
        decoder_hidden = encoder_hidden

        # Initialize the output sentence
        output_sentence = []

        # Decode until max length or EOS_token is reached
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # Get the index of the predicted word
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                break

            # Append the word to the output sentence
            output_sentence.append(output_lang.index2word[topi.item()])

            # Use the predicted word as the next input
            decoder_input = topi.squeeze().detach()

        return ', '.join(output_sentence)

# Replace this with your actual data
training_pairs = [
    ("I really liked the attention the employees gave me. As well as the nice establishment that the store has.",
     "Good customer service, Good infrastructure"),
    ("The product was of high quality and exceeded my expectations. Will definitely buy again!",
     "High-quality product, Exceeded expectations, Will buy again"),
    ("The customer support was terrible. They took forever to respond to my queries.",
     "Poor customer support, Slow response"),
    ("I enjoyed the ambiance of the restaurant, and the food was delicious.",
     "Great ambiance, Delicious food"),
    ("The shipping was fast, and the item arrived in perfect condition.",
     "Fast shipping, Item in perfect condition"),
]

# Create language objects
input_lang = Lang()
output_lang = Lang()

# Add sentences to language objects
for pair in training_pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])

input_size = input_lang.n_words
output_size = output_lang.n_words
hidden_size = 256

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
criterion = nn.NLLLoss()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

# Set up the learning rate scheduler
encoder_scheduler = StepLR(encoder_optimizer, step_size=10, gamma=0.9)
decoder_scheduler = StepLR(decoder_optimizer, step_size=10, gamma=0.9)

# Training loop (adapt this based on your data loading and training process)
num_epochs = 100  # You can adjust this to your desired number of epochs

for epoch in range(num_epochs):
    for input_sentence, target_sentence in training_pairs:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Convert input and target sentences to tensors
        input_tensor = tensorFromSentence(input_lang, input_sentence)
        target_tensor = tensorFromSentence(output_lang, target_sentence)

        # Encoder forward pass
        encoder_output, encoder_hidden = encoder(input_tensor)

        # Decoder input
        decoder_input = torch.tensor([[SOS_token]])  # SOS_token marks the start of decoding
        decoder_hidden = encoder_hidden

        # Training the decoder
        target_length = len(target_tensor)
        loss = 0
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Calculate loss
            loss += criterion(decoder_output, target_tensor[di])

            # Use teacher forcing for the next input
            decoder_input = target_tensor[di]

        # Backpropagation
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    # StepLR should be called outside the inner loop
    encoder_scheduler.step()
    decoder_scheduler.step()

# Test the model
test_sentence = "I really enjoyed the product. It was amazing!"

translated_sentence = evaluate(encoder, decoder, test_sentence)
print("Input:", test_sentence)
print("Translated:", translated_sentence)

# Reference (ground truth) sentences for BLEU calculation
reference_sentences = ["Great customer service, good infrastructure", "High-quality product, exceeded expectations, will buy again",
                       "Poor customer support, slow response", "Great ambiance, delicious food", "Fast shipping, item in perfect condition"]

# Tokenize reference and translated sentences
references = [ref.split() for ref in reference_sentences]
hypothesis = translated_sentence.split()

# Calculate BLEU score with smoothing
bleu_score = sentence_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method1)
print("BLEU Score:", bleu_score)
