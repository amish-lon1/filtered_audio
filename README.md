# 🎧 Modulated Audio Signal Processing

## 📌 Objective
This project involves processing an amplitude-modulated (AM) audio signal to recover the original clean audio. Key steps include estimating the carrier frequency, demodulating the signal, applying a bandpass filter, and saving the cleaned result.

---

## 🛠️ Tools Used
- **Python 3**
- **NumPy**
- **SciPy**
- **Matplotlib**
- **Jupyter Notebook (optional for visualization)**

---

## 📂 Files
- `modulated_signal.wav` – Input AM audio file.
- `filtered_output.wav` – Cleaned, filtered output audio.
- `process_signal.py` – Python script that performs the entire workflow.
- `README.md` – This file!

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib

   import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft, fftfreq

# 1. Load the modulated audio signal
rate, data = wavfile.read("modulated_signal.wav")
t = np.arange(len(data)) / rate

# Plot the raw signal
plt.plot(t[:1000], data[:1000])
plt.title("Raw Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 2. FFT Analysis to estimate the carrier frequency
N = len(data)
T = 1.0 / rate
yf = fft(data)
xf = fftfreq(N, T)

# Positive frequencies only
pos_mask = xf > 0
xf = xf[pos_mask]
yf = np.abs(yf[pos_mask])

# Plot FFT
plt.plot(xf, yf)
plt.title("FFT of Modulated Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Find peak frequency (carrier)
peak_index = np.argmax(yf)
fc = xf[peak_index]
print(f"Estimated Carrier Frequency Fc: {fc:.2f} Hz")

# 3. Demodulate using envelope detection
analytic_signal = hilbert(data)
envelope = np.abs(analytic_signal)

# Plot demodulated signal
plt.plot(t[:1000], envelope[:1000])
plt.title("Demodulated Signal (Envelope)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 4. Bandpass Filter
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

filtered_signal = bandpass_filter(envelope, 300, 3000, rate)

# Plot filtered signal
plt.plot(t[:1000], filtered_signal[:1000])
plt.title("Filtered Clean Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 5. Save filtered signal as audio file
normalized = np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767)
wavfile.write("filtered_output.wav", rate, normalized)

print("Filtered audio saved as 'filtered_output.wav'")

