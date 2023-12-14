from scipy.io import wavfile
from scipy.signal import medfilt
import numpy as np
import os
from sys import argv
'''
Male:    85Hz - 155Hz
Female: 165Hz - 255Hz
'''


def find_freq(signal: np.ndarray, sample_freq: int, min_freq: int = 75, max_freq: int = 280) -> np.float64:
    signal_len = len(signal)
    # Hamming window
    windowed_signal = np.hamming(signal_len) * signal

    # Median filtr
    fil_signal = medfilt(windowed_signal)

    # Fast Fourier transform
    freq_vector = np.fft.rfftfreq(signal_len, 1 / sample_freq)
    fft_signal = np.fft.rfft(fil_signal)

    # Logarithm of fft signal
    log_signal = np.log(np.abs(fft_signal))

    # Cepstrum
    cepstrum = np.abs(np.fft.irfft(log_signal))

    # Quefrency
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.fftfreq(cepstrum.size, df)

    in_range = (quefrency_vector > 1 / max_freq) & (quefrency_vector <= 1 / min_freq)

    max_quefrency_index = np.argmax(np.abs(cepstrum)[in_range])
    max_frequency = 1 / quefrency_vector[in_range][max_quefrency_index]
    return max_frequency


def test_dir(dir_name: str, list_all: bool = False) -> None:
    passed_k = passed_m = total_k = total_m = 0
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        if os.path.isfile(file_path):
            fs, data = wavfile.read(file_path)
            # One channel .wav
            if len(np.shape(data)) == 1 and len(data) != 0:
                result = find_freq(data, fs)
                gender = ("M" if result < 165 else "K")
            # If there is more channels we take first one
            elif len(np.shape(data)) > 1:
                result = find_freq(data.T[0], fs)
                gender = ("M" if result < 165 else "K")
            else:
                continue

            passed = (True if filename.replace(".wav", "")[-1] == gender else False)
            if gender == "M":
                total_m += 1
                if passed:
                    passed_m += 1
            elif gender == "K":
                total_k += 1
                if passed:
                    passed_k += 1
            if list_all:
                print(f'{filename.replace(".wav", "")}: {passed} \t\t{gender} - {result}')

    print(f"Total_m: {total_m}\tPassed_m: {passed_m}\tAccuracy_m: {passed_m/total_m}")
    print(f"Total_k: {total_k}\tPassed_k: {passed_k}\tAccuracy_k: {passed_k/total_k}")
    print(f"Accuracy: {(passed_m + passed_k)/(total_m + total_k)}")


def test_file(file_path: str, one_letter: bool = True) -> None:
    if os.path.isfile(file_path):
        fs, data = wavfile.read(file_path)
        # One channel .wav
        if len(np.shape(data)) == 1 and len(data) != 0:
            result = find_freq(data, fs)
            gender = ("M" if result < 165 else "K")
        # If there is more channels we take first one
        elif len(np.shape(data)) > 1:
            result = find_freq(data.T[0], fs)
            gender = ("M" if result < 165 else "K")
        else:
            print("Invalid file")
            return
        passed = (True if os.path.basename(file_path).replace(".wav", "")[-1] == gender else False)
        if one_letter:
            print(gender)
            return
        print(f'{os.path.basename(file_path).replace(".wav", "")}: {passed} \t\t{gender} - {result}')
        return
    print(".wav file not found")


if __name__ == "__main__":
    if len(argv) > 1:
        test_file(argv[1], one_letter=True)
    else:
        print(".wav file not found")
    # test_file("trainall/001_K.wav", one_letter=True)
    # test_dir("trainall", list_all=False)
