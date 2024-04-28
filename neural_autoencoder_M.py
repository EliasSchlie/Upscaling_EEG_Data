import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import random
from sklearn.model_selection import train_test_split

train_autoencoder = True

# Define the number of images to create and the image dimensions
# Load your data as a NumPy array
train_data = np.load("./extracted/train_trials.npy")
train_labels = np.load("./extracted/train_labels.npy")
test_data = np.load("./extracted/test_trials.npy")
test_labels = np.load("./extracted/test_labels.npy")
# train_random = np.load("./extracted/train_randomized.npy")


print("-----------------------------------------")
print("initial shape of data")
print("-----------------------------------------")
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
# print(train_random.shape)


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
    # Shuffle the train_random data
    # # train_random = np.transpose(train_random, (0, 2, 1))
    # train_random = train_random[:, :, :]
    # train_random = shuffle(train_random, random_state=0)

    train_labels = np.where(train_labels == -1, 0, train_labels)
    test_labels = np.where(test_labels == -1, 0, test_labels)
    # One-hot encode the train and test labels
    """label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    train_labels_onehot = onehot_encoder.fit_transform(train_labels_encoded.reshape(-1, 1))
    test_labels_onehot = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))
    """
    # Create a standardization scaler
    scaler = StandardScaler()

    # Reshape the train and test data to 2D arrays with shape (num_samples, num_pixels)
    num_samples_train = train_data.shape[0]
    num_pixels_train = train_data.shape[1] * train_data.shape[2]
    train_data_2d = train_data.reshape(num_samples_train, num_pixels_train)

    num_samples_test = test_data.shape[0]
    num_pixels_test = test_data.shape[1] * test_data.shape[2]
    test_data_2d = test_data.reshape(num_samples_test, num_pixels_test)

    # num_samples_train_random = train_random.shape[0]
    # num_pixels_train_random = train_random.shape[1] * train_random.shape[2]
    # train_random_2d = train_random.reshape(num_samples_train_random, num_pixels_train_random)

    # Scale the train and test data using MinMaxScaler
    train_data_scaled_2d = scaler.fit_transform(train_data_2d)
    test_data_scaled_2d = scaler.transform(test_data_2d)
    # train_random_scaled_2d = scaler.transform(train_random_2d)

    # Reshape the scaled train and test data back to the original shape
    train_data = train_data_scaled_2d.reshape(train_data.shape)
    test_data = test_data_scaled_2d.reshape(test_data.shape)
    # train_random = train_random_scaled_2d.reshape(train_random.shape)

    return train_data, test_data, train_labels, test_labels


# Define the autoencoder model

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    latent_dim = 100

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2048, 16, 1)),
            tf.keras.layers.Conv2D(
                # number of filters or output channels for the convolutional layer.
                # Each filter learns to detect different patterns or features in the input data.
                256,
                # specifies the size of the convolutional kernel or filter.
                # In this case, the kernel has a height of 20 and a width of 1.
                # The kernel slides over the input data to perform the convolution operation.
                [20, 1],
                # Rectified Linear Unit (ReLU) activation function is used,
                # which sets negative values to zero and keeps positive values unchanged.
                activation="relu",
                # This determines the padding strategy for the convolution operation.
                # "Same" padding pads the input data with zeros in such a way that the output has the same spatial dimensions as the input.
                padding="same",
                # This specifies the stride or step size of the convolutional kernel as it moves across the input data.
                # In this case, the kernel moves 4 steps horizontally and 1 step vertically at each convolution operation.
                strides=[4, 1],
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
            tf.keras.layers.Conv2D(
                128, [10, 1], activation="relu", padding="same", strides=[2, 1]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
            tf.keras.layers.Conv2D(
                64, [8, 2], activation="relu", padding="same", strides=[2, 2]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
            tf.keras.layers.Conv2D(
                32, [4, 2], activation="relu", padding="same", strides=[4, 2]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2,
        ]
    )

    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(32 * 4 * 32, activation="relu"),  # Matching the output of last encoder layer
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Reshape((32, 4, 32)),
        tf.keras.layers.Conv2DTranspose(
            32, [4, 2], activation="relu", padding="same", strides=[4, 2]
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2DTranspose(
            64, [8, 2], activation="relu", padding="same", strides=[2, 2]
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2DTranspose(
            128, [10, 1], activation="relu", padding="same", strides=[2, 1]
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2DTranspose(
            256, [20, 1], activation="relu", padding="same", strides=[4, 1]  # Changed to 'valid' to remove excess width
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2DTranspose(
            1, [20, 1], activation="sigmoid", padding="same"
        ),
    ])


    print(encoder.summary())
    print(decoder.summary())
    autoencoder = tf.keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse"
    )

# data preprocessing
train_data, test_data, train_labels, test_labels = preprocessing(
    train_data, test_data, train_labels, test_labels
)

# Sample the eeg channels number 1, 3, 4, 7, 10, 11, 15, 16
sub_sampled_train_data = train_data[:, :, [0, 2, 3, 6, 9, 10, 14, 15]]
sub_sampled_test_data = test_data[:, :, [0, 2, 3, 6, 9, 10, 14, 15]]

# split traindata and subsampled train data into train and validation sets
sub_sampled_train_data, sub_sampled_validation_data, train_data, validation_data = train_test_split(sub_sampled_train_data, train_data, test_size=0.1, random_state=42)

# save a sample of the subsampled train data and the corresponding full data in a plt image
n_images = 5
image_indeces = random.sample(range(len(sub_sampled_train_data)), n_images)
original_images = train_data[image_indeces]
sampled_images = sub_sampled_train_data[image_indeces]

# Create a figure object with adjusted wspace
fig = plt.figure(figsize=(n_images * 8, 16))
fig.subplots_adjust(wspace=2, hspace=0.5)

# Loop through the images
for i in range(n_images):
    # Original image
    ax = fig.add_subplot(2, n_images, i + 1)
    ax.pcolormesh(original_images[i], cmap="hot")
    ax.axis("on")
    ax.set_xticks(range(0, original_images[i].shape[1], 5))
    ax.set_xticklabels(range(0, original_images[i].shape[1], 5))

    # Sampled image
    ax = fig.add_subplot(2, n_images, n_images + i + 1)
    ax.pcolormesh(sampled_images[i], cmap="hot")
    ax.axis("on")
    ax.set_xticks(range(0, sampled_images[i].shape[1], 5))
    ax.set_xticklabels(range(0, sampled_images[i].shape[1], 5))

# Set the title
fig.suptitle("Original vs Sampled Images")

# Save the plot as a PNG image file
fig.savefig("original_vs_sampled.png")


print(sub_sampled_train_data.shape)
print(train_data.shape)
print(sub_sampled_validation_data.shape)
print(validation_data.shape)

# Constants
total_channels = 16  # Total channels (0 to 15)
num_channels_to_sample = 8  # Number of channels to sample
num_iterations = 15  # Number of times to sample and expand the dataset
split_index = int(len(train_data) * 0.9)

# Prepare lists to hold expanded datasets
expanded_train_data = []
expanded_train_full = []  # Corresponding full data for training
expanded_validation_data = []
expanded_validation_full = []  # Corresponding full data for validation

# Perform the sampling and data modification 10 times
for _ in range(num_iterations):
    # Generate a list of random channel indices
    random_indices = random.sample(range(total_channels), num_channels_to_sample)
    
    # Create zeroed copies of the dataset for this iteration
    iter_train_zeros = np.zeros_like(train_data[:split_index])
    iter_validation_zeros = np.zeros_like(train_data[split_index:])
    
    # Copy the data from the sampled channels into the zeroed arrays
    iter_train_zeros[:, :, random_indices] = train_data[:split_index, :, random_indices]
    iter_validation_zeros[:, :, random_indices] = train_data[split_index:, :, random_indices]
    
    # Append to the list of datasets
    expanded_train_data.append(iter_train_zeros)
    expanded_validation_data.append(iter_validation_zeros)
    # Append the corresponding full data batches
    expanded_train_full.append(train_data[:split_index])
    expanded_validation_full.append(train_data[split_index:])

# Concatenate all iterations to form the final datasets
final_train_data = np.concatenate(expanded_train_data, axis=0)
final_validation_data = np.concatenate(expanded_validation_data, axis=0)
final_train_full = np.concatenate(expanded_train_full, axis=0)
final_validation_full = np.concatenate(expanded_validation_full, axis=0)

# Print shape of the subsampled train data
print(final_train_data.shape)

if train_autoencoder:
    autoencoder.fit(
        x=final_train_data,
        y=final_train_full,
        epochs=1000,
        batch_size=128,
        validation_data=(final_validation_data, final_validation_full),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
        ],
        verbose=2,
    )

    # Save the autoencoder model
    autoencoder.save("autoencoder_M.h5")
else:
    # Load the autoencoder model
    autoencoder = tf.keras.models.load_model("autoencoder_M.h5")

# Generate the latent space
# latent_space = encoder.predict(np.asarray(train_data))

# Calculate loss on the test data
loss = autoencoder.evaluate(sub_sampled_test_data, test_data)
print("Test loss:", loss)


print("-----------------------------------------")
print("preprocessed shape of data")
print("-----------------------------------------")
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

n_images = 10
image_indeces = random.sample(range(len(test_data)), n_images)
original_images = test_data[image_indeces]
reconstructed_images = autoencoder.predict(sub_sampled_test_data[image_indeces])
print(original_images.shape)


# Create a figure object with adjusted wspace
fig = plt.figure(figsize=(n_images * 8, 16))
fig.subplots_adjust(wspace=2, hspace=0.5)

# Loop through the images
for i in range(n_images):
    # Original image
    ax = fig.add_subplot(2, n_images, i + 1)
    ax.pcolormesh(original_images[i], cmap="hot")
    ax.axis("on")
    ax.set_xticks(range(0, original_images[i].shape[1], 5))
    ax.set_xticklabels(range(0, original_images[i].shape[1], 5))

    # Reconstructed image
    ax = fig.add_subplot(2, n_images, n_images + i + 1)
    ax.pcolormesh(reconstructed_images[i][:, :, 0], cmap="hot")
    ax.axis("on")
    ax.set_xticks(range(0, reconstructed_images[i].shape[1], 5))
    ax.set_xticklabels(range(0, reconstructed_images[i].shape[1], 5))

# Set the title
fig.suptitle("Original vs Reconstructed Images")

# Save the plot as a PNG image file
fig.savefig("original_vs_reconstructed.png")

plt.close()

reconstructed_train = autoencoder.predict(train_data)
np.save("processed_data/R_train_dataset_M.npy", reconstructed_train)

reconstructed_test = autoencoder.predict(test_data)
np.save("processed_data/R_test_dataset_M.npy", reconstructed_test)

np.save("processed_data/Shuffled_train_labels_M.npy", train_labels)
np.save("processed_data/Shuffled_test_labels_M.npy", test_labels)
np.save("processed_data/Shuffled_train_M.npy", train_data)
np.save("processed_data/Shuffled_test_M.npy", test_data)

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# New in version 1.2: sparse was renamed to sparse_output
onehot_encoder = OneHotEncoder(sparse_output=False, categories="auto")
train_labels = onehot_encoder.fit_transform(train_labels_encoded.reshape(-1, 1))
test_labels = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))

# Define the input shape
input_shape = (2048, 16)

# Define the number of classes
num_classes = 2

# Define the model on original images

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    model_original = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2048, 16)),
            tf.keras.layers.Conv2D(
                256, [20, 1], activation="relu", padding="same", strides=[4, 1]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
            tf.keras.layers.Conv2D(
                128, [10, 1], activation="relu", padding="same", strides=[2, 1]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
            tf.keras.layers.Conv2D(
                64, [8, 2], activation="relu", padding="same", strides=[2, 2]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),  # added dropout layer with rate 0.2
            tf.keras.layers.Conv2D(
                32, [4, 2], activation="relu", padding="same", strides=[5, 2]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5000, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3000, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model
    model_original.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

# Print the model summary
print("-----------------------------------------")
print("Model on original images")
print("-----------------------------------------")
model_original.summary()

# Train the model on original images
history_original = model_original.fit(
    train_data,
    train_labels,
    epochs=300,
    validation_data=(test_data, test_labels),
    batch_size=128,
    verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)],
)

model_original_history = history_original.history

# Reset the model states
model_original.reset_states()

# Print the model summary for reconstructed images
print("-----------------------------------------")
print("Model on reconstructed images")
print("-----------------------------------------")

# Train the model on reconstructed images
history_reconstructed = model_original.fit(
    reconstructed_train,
    train_labels,
    epochs=300,
    validation_data=(reconstructed_test, test_labels),
    batch_size=128,
    verbose=2,
)
#                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                                 patience=150,
#                                                                 restore_best_weights=True)])

model_reconstructed_history = history_reconstructed.history

# Define the model on model_latentspace
input_shape = latent_dim

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    model_latent_space = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                4000, activation=tf.keras.layers.LeakyReLU(), input_shape=(input_shape,)
            ),
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
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model
    model_latent_space.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )


print("-----------------------------------------")
print("model on model_latentspace")
print("-----------------------------------------")

# Print the model summary
model_latent_space.summary()

# Train the model
history_latent_space = model_latent_space.fit(
    latent_space_train,
    train_labels,
    epochs=300,
    validation_data=(latent_space_test, test_labels),
    batch_size=128,
    verbose=2,
)
#                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150,
#                                                                    restore_best_weights=True)])
#

model_latent_space_history = history_latent_space.history

# Extracting validation accuracy of each model
model_original_acc = model_original_history["val_accuracy"]
print(model_original_acc, "model_original_acc")
model_reconstructed_acc = model_reconstructed_history["val_accuracy"]
print(model_reconstructed_acc, "model_reconstructed_acc")
model_latent_space_acc = model_latent_space_history["val_accuracy"]

# create x-axis values
# epochs = np.arange(1, len(model_hybrid_acc) + 1)
epochs = np.arange(
    1,
    max(
        len(model_original_acc),
        len(model_reconstructed_acc),
        len(model_latent_space_acc),
    )
    + 1,
)

# plot the validation accuracy for both models
plt.plot(epochs, model_original_acc, "r", label="model_original_acc")
plt.plot(epochs, model_reconstructed_acc, "b", label="model_reconstructed_acc")
plt.plot(epochs, model_latent_space_acc, "g", label="model_latent_space_acc")

# set the axis labels and title
plt.title("Validation accuracy comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")
plt.legend()

# save the figure to a file
plt.savefig("accuracy_comparison.png")
