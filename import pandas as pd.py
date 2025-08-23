import pandas as pd
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Load the dataset
data = pd.read_csv(r'C:\Users\YOGESH YADAV\Desktop\Y\vscode\EEG_data.csv')
# Assuming the last column is the label (subject_understood)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Separate EEG channels and power bands
eeg_channels = X.filter(regex='^EEG').values
power_bands = X.filter(regex='^POW').values
# Preprocessing: Apply a bandpass filter to EEG channels
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

    # Assuming a sampling rate of 128 Hz (adjust if different)
fs = 128
filtered_eeg = np.apply_along_axis(bandpass_filter, 0, eeg_channels, 0.5, 50, fs)

# Combine filtered EEG and power bands
X_processed = np.hstack((filtered_eeg, power_bands))
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

from scipy.signal import hilbert
from scipy.spatial.distance import pdist, squareform

def phase_locking_value(data):
    analytic_signal = hilbert(data)
    phase = np.angle(analytic_signal)
    n_channels = phase.shape[1]
    plv = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = phase[:, i] - phase[:, j]
            plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv[j, i] = plv[i, j]
    return plv
# Convert EEG signals to functional connectivity images
def create_connectivity_images(data):
    n_samples = data.shape[0]
    n_channels = 14  # Number of EEG channels
    connectivity_images = np.zeros((n_samples, n_channels, n_channels))

    for i in range(n_samples):
        eeg_sample = data[i, :n_channels]
        connectivity_images[i] = phase_locking_value(eeg_sample.reshape(1, -1))

    return connectivity_images

X_train_conn = create_connectivity_images(X_train)
X_test_conn = create_connectivity_images(X_test)
# Reshape for CNN input
X_train_conn = X_train_conn.reshape(X_train_conn.shape[0], X_train_conn.shape[1], X_train_conn.shape[2], 1)
X_test_conn = X_test_conn.reshape(X_test_conn.shape[0], X_test_conn.shape[1], X_test_conn.shape[2], 1)



import matplotlib.pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle

# Define channel names (14 channels)
ch_names = [f'Ch{i}' for i in range(14)]

# Assuming 'con' is your connectivity data with shape (n_samples, n_channels, n_channels, 1)
con = X_train_conn
# Loop through each sample
for i in range(con.shape[0]):  # con.shape[0] is the number of samples
    # Get connectivity data for the current sample
    con_data = con[i, :, :, 0]

    # Plot circular connectivity for the current sample
    fig, ax = plot_connectivity_circle(con_data, ch_names, n_lines=20,
                                       title=f'EEG Connectivity - Sample {i}')

    # Save the circular plot for the current sample
    #fig.savefig(f'connectivity_circle_sample_{i}.png', bbox_inches='tight', pad_inches=0)

    # Display the plot (optional, you might want to comment this out for large datasets)
    # plt.show()
    plt.close(fig) # Close the figure to avoid memory issues

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(14, 14, 1)))  # Input shape matches connectivity images
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # Added padding='same'
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

