import os
import pandas as pd
import numpy as np

# get a list of all the EEG data files in the eeg_data folder
eeg_files = [
    f for f in os.listdir("eeg_data") if f.startswith("S") and f.endswith(".csv")
]

# Extract train data
# create empty lists to hold all the trials and labels
train_trials = []
train_labels = []

# create empty lists to hold the train and test patient IDs
train_patients = []
test_patients = []

# loop through each EEG data file
for eeg_file in eeg_files:
    # get the subject ID
    subject_id = eeg_file.split("_")[0]

    # load the EEG data for a specific subject and run
    eeg_data = pd.read_csv(os.path.join("eeg_data", eeg_file))

    # get the trial numbers and corresponding labels
    trial_numbers = eeg_data["trial"].values
    labels = eeg_data["class"].values

    # loop through each trial number and extract the corresponding data and label
    for trial_number in np.unique(trial_numbers):
        # get the index of the first data point for the trial
        start_index = np.where(trial_numbers == trial_number)[0][0]

        # extract the data and label for the trial
        trial_data = eeg_data.iloc[start_index : start_index + 2000, 1:-2].values
        label = labels[start_index]

        # check if the subject ID is already in the train or test patient list
        if subject_id not in train_patients and subject_id not in test_patients:
            # add the subject ID to either the train or test patient list randomly
            if np.random.rand() < 0.8:
                train_patients.append(subject_id)
            else:
                test_patients.append(subject_id)

        # add the data and label to the train_trials and train_labels lists if the subject is in the train set
        if subject_id in train_patients:
            train_trials.append(trial_data)
            train_labels.append(label)

# convert the lists of trials and labels to numpy arrays
train_trials_np = np.array(train_trials)
train_labels_np = np.array(train_labels)

# save the trials and labels to numpy files
np.save("train_trials.npy", train_trials_np)
np.save("train_labels.npy", train_labels_np)

# Extract test data
test_trials = []
test_labels = []

# loop through each EEG data file again
for eeg_file in eeg_files:
    # get the subject ID
    subject_id = eeg_file.split("_")[0]

    # check if the subject ID is in the test patient list
    if subject_id in test_patients:
        # load the EEG data for a specific subject and run
        eeg_data = pd.read_csv(os.path.join("eeg_data", eeg_file))

        # get the trial numbers and corresponding labels
        trial_numbers = eeg_data["trial"].values
        labels = eeg_data["class"].values

        # loop through each trial number and extract the corresponding data and label
        for trial_number in np.unique(trial_numbers):
            # get the index of the first data point for the trial
            start_index = np.where(trial_numbers == trial_number)[0][0]

            # extract the data and label for the trial
            trial_data = eeg_data.iloc[start_index : start_index + 2000, 1:-2].values
            label = labels[start_index]

            # add the data and label to the all_trials and all_labels lists
            test_trials.append(trial_data)
            test_labels.append(label)


# convert the lists of trials and labels to numpy arrays
test_trials_np = np.array(test_trials)
test_labels_np = np.array(test_labels)

# save the trials and labels to numpy files
np.save("test_trials.npy", test_trials_np)
np.save("test_labels.npy", test_labels_np)