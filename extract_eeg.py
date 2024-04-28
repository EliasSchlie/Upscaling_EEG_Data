import os
import pandas as pd
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

# Check if extracted folder exists or create it
if not os.path.exists('extracted'):
    os.makedirs('extracted')

# MARK: Extract training data

# get a list of all the EEG training data files in the eeg_data folder
eeg_files = [
    f for f in os.listdir("eeg_data") if f.startswith("P") and f.endswith("training.mat")
]

# Extract train data
# create empty lists to hold all the trials and labels
train_trials = []
train_labels = []

# Directory where the images will be saved
save_dir = "extracted_images"
os.makedirs(save_dir, exist_ok=True)

# loop through each EEG data file
for eeg_file in eeg_files:

    # load the EEG data for a specific subject and run
    eeg_data = scipy.io.loadmat(os.path.join("eeg_data", eeg_file))
    sub_dir = os.path.join(save_dir, eeg_file.split("_")[0])
    os.makedirs(sub_dir, exist_ok=True)

    targets = eeg_data["trig"]
    eeg_data = pd.DataFrame(eeg_data["y"])

    # Get the indices of the first data point per trial and the end of each trial
    trial_starts = []
    trial_ends = []
    switch = 0
    for i in range(len(targets)):
        if targets[i] != switch:
            if switch == 0 and targets[i] != 0:
                # Detect start of a new trial
                trial_starts.append(i)
            elif switch != 0 and targets[i] == 0:
                # Detect end of a trial
                trial_ends.append(i)
            # Update 'switch' to the current target
            switch = targets[i]

    # Handle the case where the last trial does not end with a zero
    if switch != 0:
        trial_ends.append(len(targets))

    # loop through each trial number and extract the corresponding data and label
    for trial in range(len(trial_starts)):
        # extract the data and label for the trial
        trial_data = eeg_data.iloc[trial_starts[trial] : trial_ends[trial], :].values
        label = targets[trial_starts[trial]]

        train_trials.append(trial_data)
        train_labels.append(label)

        # fig, ax = plt.subplots(figsize=(8, 16))  # Adjust figure size as needed
        # cax = ax.pcolormesh(trial_data, cmap="hot")  # Use a colormap that suits your data
        # ax.axis("on")  # Turn on axis if needed, or 'off' to turn them off
        # ax.set_xticks(range(0, trial_data.shape[1], 5))
        # ax.set_xticklabels(range(0, trial_data.shape[1], 5))

        # # Save each figure using the trial index to ensure unique filenames
        # fig.savefig(os.path.join(sub_dir, f"trial_{trial}.png"))
        # plt.close(fig)  # Close the figure after saving to free up memory



# convert the lists of trials and labels to numpy arrays
train_trials_np = np.array(train_trials)
train_labels_np = np.array(train_labels)

print(train_trials_np.shape)
print(train_labels_np.shape)

# save the trials and labels to numpy files
np.save("extracted/train_trials.npy", train_trials_np)
np.save("extracted/train_labels.npy", train_labels_np)

# MARK: Extract test data

# get a list of all the EEG data files in the eeg_data folder
eeg_files = [
    f for f in os.listdir("eeg_data") if f.startswith("P") and f.endswith("test.mat")
]

# Extract test data
test_trials = []
test_labels = []

# loop through each EEG data file
for eeg_file in eeg_files:
    # get the subject ID
    subject_id = eeg_file.split("_")[0]

    # load the EEG data for a specific subject and run
    eeg_data = scipy.io.loadmat(os.path.join("eeg_data", eeg_file))

    targets = eeg_data["trig"]
    eeg_data = pd.DataFrame(eeg_data["y"])

    # Get the indices of the first data point per trial and the end of each trial
    trial_starts = []
    trial_ends = []
    switch = 0
    for i in range(len(targets)):
        if targets[i] != switch:
            if switch == 0 and targets[i] != 0:
                # Detect start of a new trial
                trial_starts.append(i)
            elif switch != 0 and targets[i] == 0:
                # Detect end of a trial
                trial_ends.append(i)
            # Update 'switch' to the current target
            switch = targets[i]

    # Handle the case where the last trial does not end with a zero
    if switch != 0:
        trial_ends.append(len(targets))

    # loop through each trial number and extract the corresponding data and label
    for trial in range(len(trial_starts)):
        # extract the data and label for the trial
        trial_data = eeg_data.iloc[trial_starts[trial] : trial_ends[trial], :].values
        label = targets[trial_starts[trial]]

        test_trials.append(trial_data)
        test_labels.append(label)

# convert the lists of trials and labels to numpy arrays
test_trials_np = np.array(test_trials)
test_labels_np = np.array(test_labels)

print(test_trials_np.shape)
print(test_labels_np.shape)

# save the trials and labels to numpy files
np.save("extracted/test_trials.npy", test_trials_np)
np.save("extracted/test_labels.npy", test_labels_np)


# MARK: Extract unstructured 8 second chunks of data for unsupervised data augmentation

# get a list of all the EEG training data files in the eeg_data folder
eeg_files = [
    f for f in os.listdir("eeg_data") if f.startswith("P") and f.endswith("training.mat")
]

# Extract train data
# create empty lists to hold all the data
train_randomized = []

# loop through each EEG data file
for eeg_file in eeg_files:

    # load the EEG data for a specific subject and run
    eeg_data = scipy.io.loadmat(os.path.join("eeg_data", eeg_file))

    eeg_data = pd.DataFrame(eeg_data["y"])

    for i in range(0, len(eeg_data)- 2047, 2048):
        train_randomized.append(eeg_data.iloc[i : i + 2048, :].values) 

# convert the lists of trials and labels to numpy arrays
train_randomized_np = np.array(train_randomized)

print(train_randomized_np.shape)

# save the trials and labels to numpy files
np.save("extracted/train_randomized.npy", train_trials_np)

