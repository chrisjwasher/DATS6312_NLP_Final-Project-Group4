import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load and preprocess the dataset
# Assuming 'X' is the preprocessed dataset and 'y' are the corresponding labels
# You need to replace 'X' and 'y' with your actual dataset
## for reading a csv file
Corpus = pd.read_csv(r"data_set.csv", encoding='latin-1').head(1000)

###drop missing values
Corpus['text'].dropna(inplace=True)

# Create a copy of the 'text' column for further processing
Corpus['text_original'] = Corpus['text'].copy()


# make lowercase
Corpus['text'] = [entry.lower() for entry in Corpus['text']]




# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(Corpus['text'], Corpus['label_num'], test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max([len(seq) for seq in X_train_seq])
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Generate class weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Define model architecture
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=300)(input_layer)

bi_lstm_layer = Bidirectional(LSTM(units=128, return_sequences=True))(embedding_layer)
bi_lstm_dropout = Dropout(0.2)(bi_lstm_layer)
bi_lstm_l2 = Dense(256, kernel_regularizer=l2(0.01))(bi_lstm_dropout)
bi_lstm_batchnorm = BatchNormalization()(bi_lstm_l2)

# Auxiliary outputs
auxiliary_outputs = []
for i in range(3):  # Assuming 3 classes
    aux_output = Dense(1, activation='sigmoid', name='aux_output_'+str(i+1))(bi_lstm_batchnorm)
    auxiliary_outputs.append(aux_output)

# Concatenate auxiliary outputs
concatenated_outputs = Concatenate()(auxiliary_outputs)

# Primary output
primary_output = Dense(3, activation='softmax', name='primary_output')(concatenated_outputs)

# Create model
model = Model(inputs=input_layer, outputs=[primary_output] + auxiliary_outputs)

# Compile model
model.compile(optimizer=Adam(), loss={'primary_output': 'categorical_crossentropy',
                                      'aux_output_1': 'binary_crossentropy',
                                      'aux_output_2': 'binary_crossentropy',
                                      'aux_output_3': 'binary_crossentropy'},
                            loss_weights={'primary_output': 1.0,
                                          'aux_output_1': 1.0,
                                          'aux_output_2': 1.0,
                                          'aux_output_3': 1.0},
                            metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train model
model.fit(X_train_padded, {'primary_output': y_train, 'aux_output_1': y_train, 'aux_output_2': y_train, 'aux_output_3': y_train},
          validation_data=(X_test_padded, {'primary_output': y_test, 'aux_output_1': y_test, 'aux_output_2': y_test, 'aux_output_3': y_test}),
          epochs=10, batch_size=64, class_weight={'primary_output': class_weights, 'aux_output_1': class_weights,
                                                  'aux_output_2': class_weights, 'aux_output_3': class_weights})