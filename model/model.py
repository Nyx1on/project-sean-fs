import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


with open('preprocessed_data.pickle', 'rb') as f:
    train_padded, test_padded, train_labels, test_labels = pickle.load(f)


model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=32, input_length=1000),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_padded, train_labels, epochs=10,
          validation_data=(test_padded, test_labels))

test_loss, test_acc = model.evaluate(test_padded, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

model.save('sentimental_analysis_model.h5')