# Cognitive-Workload-Detection-using-EEG-Image
# DataSet Link: https://drive.google.com/file/d/1tuIO99xXS1Yb8LOvvT8ch4llieSgZdmc/view?usp=drive_link

# EEG-based Classification using Deep Learning
This project demonstrates a deep learning approach to classify EEG signals. It involves preprocessing raw EEG data, calculating functional connectivity, and using a Convolutional Neural Network (CNN) to perform the classification task.

# Project Overview
The core idea is to transform time-series EEG data into a spatial representation, specifically a connectivity matrix, which is then treated as an "image" for a CNN. This allows the model to learn spatial patterns and relationships between different brain regions.

The key steps in the pipeline are:

# 1.Data Loading & Preprocessing:
The raw EEG data is loaded from a CSV file. It is then preprocessed by applying a bandpass filter to isolate relevant frequency bands.

# 2.Functional Connectivity Analysis:
The preprocessed EEG signals are used to compute a Phase Locking Value (PLV) matrix. PLV is a measure of the synchronization of phases between two signals, providing a quantitative representation of functional connectivity. This matrix serves as the input "image" for the CNN.

# 3.Data Visualization:
The code includes an optional section to visualize the computed connectivity matrices using circular plots. This helps in understanding the brain networks and validating the connectivity measures.

# 4.CNN Model Training:
A custom CNN model is defined and trained on the connectivity matrices. The model learns to identify patterns in brain connectivity that correlate with the target variable (e.g., subject_understood).

# 5.Evaluation:
The trained model's performance is evaluated using metrics like accuracy, precision, and recall on a separate test set.

# Dependencies:
To run this code, you'll need the following libraries:

pandas

numpy

scipy

scikit-learn

tensorflow

matplotlib

mne-connectivity

# You can install them using pip:

pip install pandas numpy scipy scikit-learn tensorflow matplotlib mne_connectivity


# Usage:

# 1.Prepare your data:
Ensure your EEG data is in a CSV file named EEG_data.csv. The last column should be your target variable (e.g., subject_understood), and the preceding columns should contain your EEG channels and power band features.

# 2.Update file path:
Modify the file path in the pd.read_csv() function to point to your data file.

# 3.Run the script: Execute the Python script.
It will load the data, preprocess it, compute connectivity, train the CNN model, and print the evaluation metrics.

# Code Structure:
The script is organized into logical blocks:

# 1.Imports:
All necessary libraries are imported at the beginning.

# 2.Data Loading & Splitting:
The data is loaded and split into training and testing sets.

# 3.Preprocessing Functions:
bandpass_filter is defined to preprocess the raw EEG signals.

# 4.Connectivity Functions:
phase_locking_value and create_connectivity_images are defined to compute the PLV matrix and prepare the data for the CNN.

# 5.Visualization:
A loop is included to generate and save circular connectivity plots.

# 6.Model Definition & Training:
The CNN architecture is defined using tensorflow.keras.Sequential.

# 7.Evaluation:
The trained model is evaluated, and the results are printed to the console.

# Functional Connectivity: A Deeper Dive
Functional connectivity refers to the statistical interdependence of neural activity from different brain regions. It doesn't imply a direct physical connection but rather that the regions are working together to perform a task. Phase Locking Value (PLV) is a widely used measure for this. It quantifies the consistency of the phase difference between two signals over time. A PLV close to 1 indicates high phase synchronization, while a value close to 0 suggests a random phase relationship.

By converting time-series data into these static connectivity matrices, we can leverage the power of CNNs, which excel at identifying patterns in grid-like data (like images). The CNN learns to distinguish between different brain states based on the unique patterns of inter-regional synchronization.







