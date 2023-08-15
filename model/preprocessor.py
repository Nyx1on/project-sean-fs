import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# load the dataset
data = pd.read_csv('IMDB Dataset.csv')

print('Loading dataset')

# select the text and label columns
texts = data['review']
labels = data['sentiment']

binary_labels = np.array([1 if label == 'positive' else 0 for label in labels])

# split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, binary_labels, test_size=0.3, random_state=42)

# tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

tokenizer.fit_on_texts(train_texts)

# convert the text into sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# After fitting the Tokenizer on the training text data using the 
# fit_on_texts() method, we can convert the text data into sequences of integers using the texts_to_sequences() method.
# For example, if we have the sentence "This is a good movie" in our training data and the Tokenizer has been fitted on 
# this data, then the texts_to_sequences() method will convert this sentence to the sequence[4, 3, 2, 6], where each word
#  is replaced by its corresponding integer index in the Tokenizer's word index dictionary.
# Similarly, we convert the test text data to sequences of integers using the same Tokenizer object.

# pad the sequences to have the same length
train_padded = pad_sequences(
    train_sequences, maxlen=1000, padding='post', truncating='post')

test_padded = pad_sequences(
    test_sequences, maxlen=1000, padding='post', truncating='post')

# The pad_sequences() function is used to ensure that all sequences in a list have the same length. In the case of natural language
#  processing,we often have sentences of varying lengths, and we need to pad them so that they all have the same length in order to pass them to a neural 
# network for training.In the code above, we use pad_sequences() to pad the sequences of integers obtained from the text data. We specify a maxlen 
# of 100, which means that all sequences will be padded or truncated to a length of 100. We also specify padding='post' to pad the sequences at the
#  end and truncating='post' to truncate the sequences at the end if they are longer than 100.
# The resulting train_padded and test_padded are numpy arrays of shape (num_samples, maxlen), where num_samples is the number of samples in the dataset
#  and maxlen is the maximum length of the padded sequences (in this case, 100). These arrays can be directly used as input to a neural network for training.

with open('preprocessed_data.pickle', 'wb') as f:
    pickle.dump((train_padded, test_padded, train_labels, test_labels), f)

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
