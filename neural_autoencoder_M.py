import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
import random
# Define the number of images to create and the image dimensions
# Load your data as a NumPy array
train_data = np.load('/home/u956278/hackathon/M_data/extracted/train_trials.npy')
train_labels = np.load('/home/u956278/hackathon/M_data/extracted/train_labels.npy')
test_data = np.load('/home/u956278/hackathon/M_data/extracted/test_trials.npy')
test_labels = np.load('/home/u956278/hackathon/M_data/extracted/test_labels.npy')

print('-----------------------------------------')
print('initial shape of data')
print('-----------------------------------------')
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)


def preprocessing(train_data, test_data, train_labels, test_labels):
    # Shuffle the train data and labels
    # train_data = np.transpose(train_data, (0, 2, 1))
    train_data = train_data[:, :, :]
    train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
    print(train_data.shape)
    # Shuffle the test data and labels
    # test_data = np.transpose(test_data, (0, 2, 1))
    test_data = test_data[:, :, :]
    print(test_data.shape)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=0)
    train_labels = np.where(train_labels == -1, 0, train_labels)
    test_labels = np.where(test_labels == -1, 0, test_labels)
    # One-hot encode the train and test labels
    '''label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    train_labels_onehot = onehot_encoder.fit_transform(train_labels_encoded.reshape(-1, 1))
    test_labels_onehot = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))
    '''
    # Create an instance of MinMaxScaler with a range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Reshape the train and test data to 2D arrays with shape (num_samples, num_pixels)
    num_samples_train = train_data.shape[0]
    num_pixels_train = train_data.shape[1] * train_data.shape[2]
    train_data_2d = train_data.reshape(num_samples_train, num_pixels_train)

    num_samples_test = test_data.shape[0]
    num_pixels_test = test_data.shape[1] * test_data.shape[2]
    test_data_2d = test_data.reshape(num_samples_test, num_pixels_test)

    # Scale the train and test data using MinMaxScaler
    train_data_scaled_2d = scaler.fit_transform(train_data_2d)
    test_data_scaled_2d = scaler.transform(test_data_2d)

    # Reshape the scaled train and test data back to the original shape
    train_data = train_data_scaled_2d.reshape(train_data.shape)
    test_data = test_data_scaled_2d.reshape(test_data.shape)

    return train_data, test_data, train_labels, test_labels


# Define the autoencoder model

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    latent_dim = 100

    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2000, 16, 1)),
        tf.keras.layers.Conv2D(256, [20, 1], activation='relu', padding='same', strides=[4, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2D(128, [10, 1], activation='relu', padding='same', strides=[2, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2D(64, [8, 2], activation='relu', padding='same', strides=[2, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2D(32, [4, 2], activation='relu', padding='same', strides=[5, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2,
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(25 * 4 * 64, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Reshape((25, 4, 64)),
        tf.keras.layers.Conv2DTranspose(32, [4, 2], activation='relu', padding='same', strides=[5, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2DTranspose(64, [8, 2], activation='relu', padding='same', strides=[2, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2DTranspose(128, [10, 1], activation='relu', padding='same', strides=[2, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2DTranspose(256, [20, 1], activation='relu', padding='same', strides=[4, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2DTranspose(1, [20, 1], activation='sigmoid', padding='same')
    ])

    print(encoder.summary())
    print(decoder.summary())
    autoencoder = tf.keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# data preprocessing
train_data, test_data, train_labels, test_labels = preprocessing(train_data, test_data, train_labels, test_labels)

# Train the autoencoder
autoencoder.fit(x=train_data,
                y=train_data,
                epochs=500,
                batch_size=128,
                validation_data=(np.asarray(test_data), np.asarray(test_data)),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)],
                verbose=2)

# Generate the latent space
# latent_space = encoder.predict(np.asarray(train_data))



print('-----------------------------------------')
print('preprocessed shape of data')
print('-----------------------------------------')
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

n_images = 10
original_images = random.sample(list(test_data), n_images)
original_images=np.array(original_images)
print('diocane',original_images.shape)
reconstructed_images = autoencoder.predict(original_images)

# Create a figure object with adjusted wspace
fig = plt.figure(figsize=(n_images * 8, 16))
fig.subplots_adjust(wspace=2,hspace=0.5)

# Loop through the images
for i in range(n_images):
    # Original image
    ax = fig.add_subplot(2, n_images, i + 1)
    ax.pcolormesh(original_images[i], cmap='hot')
    ax.axis('on')
    ax.set_xticks(range(0, original_images[i].shape[1], 5))
    ax.set_xticklabels(range(0, original_images[i].shape[1], 5))

    # Reconstructed image
    ax = fig.add_subplot(2, n_images, n_images + i + 1)
    ax.pcolormesh(reconstructed_images[i][:,:,0], cmap='hot' )
    ax.axis('on')
    ax.set_xticks(range(0, reconstructed_images[i].shape[1], 5))
    ax.set_xticklabels(range(0, reconstructed_images[i].shape[1], 5))

# Set the title
fig.suptitle('Original vs Reconstructed Images')

# Save the plot as a PNG image file
fig.savefig('original_vs_reconstructed.png')

plt.close()


# save the recostructed dataset
latent_space_train = encoder.predict(train_data)
np.save('processed_data/LS_train_dataset_M.npy', latent_space_train)

latent_space_test = encoder.predict(test_data)
np.save('processed_data/LS_test_dataset_M.npy', latent_space_test)

reconstructed_train = autoencoder.predict(train_data)
np.save('processed_data/R_train_dataset_M.npy', reconstructed_train)

reconstructed_test = autoencoder.predict(test_data)
np.save('processed_data/R_test_dataset_M.npy', reconstructed_test)

np.save('processed_data/Shuffled_train_labels_M.npy', train_labels)
np.save('processed_data/Shuffled_test_labels_M.npy', test_labels)
np.save('processed_data/Shuffled_train_M.npy', train_data)
np.save('processed_data/Shuffled_test_M.npy', test_data)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
train_labels = onehot_encoder.fit_transform(train_labels_encoded.reshape(-1, 1))
test_labels = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))

# Define the input shape
input_shape = (2000, 16, 1)

# Define the number of classes
num_classes = 2

# Define the model on original images

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():

    model_original = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2000, 16, 1)),
        tf.keras.layers.Conv2D(256, [20, 1], activation='relu', padding='same', strides=[4, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2D(128, [10, 1], activation='relu', padding='same', strides=[2, 1]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2D(64, [8, 2], activation='relu', padding='same', strides=[2, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
        tf.keras.layers.Conv2D(32, [4, 2], activation='relu', padding='same', strides=[5, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5000, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3000, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model_original.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['accuracy'])

# Print the model summary
print('-----------------------------------------')
print('Model on original images')
print('-----------------------------------------')
model_original.summary()

# Train the model on original images
history_original= model_original.fit(train_data,
                       train_labels,
                       epochs=300,
                       validation_data=(test_data, test_labels),
                       batch_size=128,
                       verbose=2,)
#                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                                  patience=150,
#                                                                  restore_best_weights=True)])
model_original_history = history_original.history

# Reset the model states
model_original.reset_states()

# Print the model summary for reconstructed images
print('-----------------------------------------')
print('Model on reconstructed images')
print('-----------------------------------------')

# Train the model on reconstructed images
history_reconstructed=model_original.fit(reconstructed_train,
                       train_labels,
                       epochs=300,
                       validation_data=(reconstructed_test, test_labels),
                       batch_size=128,
                       verbose=2,)
#                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                                 patience=150,
#                                                                 restore_best_weights=True)])

model_reconstructed_history = history_reconstructed.history

# Define the model on model_latentspace
input_shape = latent_dim

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():

    model_latent_space = tf.keras.Sequential([
        tf.keras.layers.Dense(4000, activation=tf.keras.layers.LeakyReLU(), input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3000, activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2000, activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model_latent_space.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
                              
                              
print('-----------------------------------------')
print('model on model_latentspace')
print('-----------------------------------------')

# Print the model summary
model_latent_space.summary()

# Train the model
history_latent_space=model_latent_space.fit(latent_space_train,
                           train_labels, epochs=300,
                           validation_data=(latent_space_test, test_labels),
                           batch_size=128,
                           verbose=2,)
#                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150,
#                                                                    restore_best_weights=True)])
#

model_latent_space_history = history_latent_space.history

# Extracting validation accuracy of each model
model_original_acc = model_original_history['val_accuracy']
print(model_original_acc,'model_original_acc')
model_reconstructed_acc = model_reconstructed_history['val_accuracy']
print(model_reconstructed_acc,'model_reconstructed_acc')
model_latent_space_acc = model_latent_space_history['val_accuracy']

# create x-axis values
#epochs = np.arange(1, len(model_hybrid_acc) + 1)
epochs = np.arange(1, max(len(model_original_acc),len(model_reconstructed_acc), len(model_latent_space_acc)) + 1)

# plot the validation accuracy for both models
plt.plot(epochs, model_original_acc, 'r', label='model_original_acc')
plt.plot(epochs, model_reconstructed_acc, 'b', label='model_reconstructed_acc')
plt.plot(epochs, model_latent_space_acc, 'g', label='model_latent_space_acc')

# set the axis labels and title
plt.title('Validation accuracy comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.legend()

# save the figure to a file
plt.savefig('accuracy_comparison.png')