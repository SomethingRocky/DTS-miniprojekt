import os
import numpy as np
import scipy as sp
from numpy import pi, sin
import librosa as lb
import matplotlib.pyplot as plt

def spectrogram(signal: np.ndarray, outputdir: str, sr, plotname:str) -> None: 
    #Computes and plots frequency vs time
    stft = lb.stft(signal)
    stft_db = lb.amplitude_to_db(np.abs(stft) ** 2, ref = np.max, top_db= 100)
    lb.display.specshow(stft_db, sr=sr, x_axis="time", y_axis="hz")
    plt.title(plotname)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label = "Power [dBFS]")
    outputpath = os.path.join(outputdir, plotname)
    plt.savefig(outputpath)
    plt.ylim(0,2000)
    plt.savefig(outputpath + "_zoomed")
    plt.clf()

def main(subdir: str, output_folder: str, filename: str, hfilter: np.ndarray) -> None:
    """
    Plots the time vs amplitude of an audio signal,
    and computes the spectrogram of the signal

    Args:
        subdir (str): the directory from which to fetch the audio signal
        filename (str): The name of the file

    Returns:
        None: The two plots, in a subfolder of an output folder
    """
    
    #Making the output folder
    outputdir = os.path.join(output_folder, filename)
    os.makedirs(outputdir, exist_ok=True)
    
    #Finds the signal and loads it
    path = os.path.join(subdir, filename)
    signal, sample_rate = lb.load(path, sr= None, duration = 1.0)
    sample_rate = 8000
    signal = sp.signal.decimate(signal, 2)
    
    # Extend the audio signal by appending it to itself
    signal = np.concatenate([signal, signal])
    signal = np.concatenate([signal, signal])
    
    spectrogram(signal, outputdir, sample_rate, "Original signal")
    
    #making noisy signal
    t = np.arange(len(signal)) / sample_rate
    
    """ num_noise = 5
    noise_frequencies = np.linspace(3000, 8000, num_noise)
    for freq in noise_frequencies:
        signal += (0.2/num_noise) * (sin(2 * pi * freq * t)) """
    noise = np.random.random(signal.size) * 20
    # High-pass filter at 3000 Hz to keep frequencies 3000-8000 Hz
    sos = sp.signal.butter(4, 2500, 'high', fs=sample_rate, output='sos')
    noise = sp.signal.sosfilt(sos, noise)
    signal += noise
    #to avoid clipping i normalize
    signal = signal / np.max(signal)
    spectrogram(signal, outputdir, sample_rate, "Noisy signal")
    sp.io.wavfile.write(os.path.join(outputdir, "Noisy.wav"), sample_rate, np.int16(signal * 32767))
  
    
    #filtering noisy signal
    #Bruger Fischers båndpas differensligning
    """ filtered_signal = np.zeros_like(signal)
    filtered_signal[0] = 0.281 * signal[0]
    filtered_signal[1] = 1.3981 * filtered_signal[0] + 0.281 * signal[1]
    for n in range(2, len(signal) - 2):
        filtered_signal[n] = 1.3981 * filtered_signal[n-1] - 0.438 * filtered_signal[n-2] + 0.281 * (signal[n] - signal[n-2]) """
    filtered_signal = np.convolve(signal, hfilter,"same")
    
    filtered_signal = filtered_signal / np.max(filtered_signal)
    
    spectrogram(filtered_signal, outputdir, sample_rate, "Filtered signal")
    sp.io.wavfile.write(os.path.join(outputdir, "Filtered.wav"), sample_rate, np.int16(filtered_signal * 32767))
  
    
        
hfreq = np.array([-0.00183913, -0.00872584, -0.02186005, -0.02384676,  0.02073786,  0.12803849, 0.25284077,  0.3093093,   0.25284077,  0.12803849,  0.02073786, -0.02384676, -0.02186005, -0.00872584, -0.00183913])
hwin = np.array([-0.00000000e+00, -1.44161046e-04, -8.65353477e-04, -1.45894863e-03,
        1.42718512e-18,  4.54690194e-03,  9.82489832e-03,  1.00649072e-02,
       -4.87271479e-18, -1.92135592e-02, -3.66768172e-02, -3.50126315e-02,
        8.31824446e-18,  6.87042489e-02,  1.53097469e-01,  2.22916663e-01,
        2.50000000e-01,  2.22916663e-01,  1.53097469e-01,  6.87042489e-02,
        8.31824446e-18, -3.50126315e-02, -3.66768172e-02, -1.92135592e-02,
       -4.87271479e-18,  1.00649072e-02,  9.82489832e-03,  4.54690194e-03,
        1.42718512e-18, -1.45894863e-03, -8.65353477e-04, -1.44161046e-04,
       -0.00000000e+00])

if __name__ == "__main__":
    for dir, folders, files in os.walk("data"):
        if len(files) != 0:
            for file in files:
                if dir == "data":
                    """
                    if there are files in the data folder,
                    they should not go straight in the output folder,
                    hence they go in a "data" folder in output instead
                    """
                    outputpath = os.path.join("output", "data")
                else: 
                    path = os.path.relpath(dir, "data") #Removing data prefix
                    outputpath = os.path.join("output", path)
                main(dir, os.path.join(outputpath, "frequency sampling method"), file, hfreq)
                main(dir, os.path.join(outputpath, "window method"), file, hwin)
    print("All spectrograms have been compiled")






