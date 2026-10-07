#Original code from https://dev.to/kartikmehta8/introduction-to-digital-signal-processing-with-python-bj5
#Added some features and made some things cleaner


import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Sample data
fs = 5000  # Sample rate, Hz
f = 500.0  # Frequency of the signal, Hz
T = 1/f
omega = 2 * np.pi * f
cutoff = 1000  # Desired cutoff frequency of the filter, Hz
order = 10
t = np.linspace(0, 5, 100)
data = np.cos(omega * t)# + (0.02 * np.random.randn(100) * t)

# Filter the data
filtered_data = butter_lowpass_filter(data, cutoff, fs, order)
#fourier transform of the original signal
S = np.fft.fftshift(np.fft.fft(data))
S_Mag= abs(S)
S_Phase = np.angle(S)


#Fourier Transform of the filtered signal
SFil = np.fft.fftshift(np.fft.fft(filtered_data))
SFil_Mag = abs(SFil)
SFil_Phase = np.angle(SFil)

f = np.linspace(-fs/2, fs/2, len(SFil_Mag))

# Plot the original and filtered signals time domain
plt.figure(0)
plt.plot(data, label='Original data')
plt.plot(filtered_data, label='Filtered signal')
plt.legend()
plt.show()

# Plot the original and filtered signals frequency domain
plt.figure(1)
plt.plot(f, S_Mag, label='FFT of Original data')
plt.plot(f, SFil_Mag, label='FFT of Filtered signal')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(f, S_Phase, label='Phase of Original data')
plt.plot(f, SFil_Phase, label='Phase of Filtered signal')
plt.legend()
plt.show()