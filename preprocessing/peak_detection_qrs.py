'''
Finds QRS peaks in ECG signals finder using Pan-Tompkins algorithm
Used to create labels for smartwatch HR estimation
Called from gather_raw_data.py
'''

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal as sg
import pandas as pd
import pickle
import subprocess
import wfdb
import csv


'''
### input format 2 - WESAD

session = 'S2'
file_name = '/Users/jamborghini/Documents/PYTHON/TRAINING DATA - WESAD/'+session+'/'+session+'.pkl'

with open(file_name, 'rb') as file:
    data = pickle.load(file, encoding='latin1')
ecg = data['signal']['chest']['ECG']
ecg = np.squeeze(ecg)

fs = 700
time = len(ecg) / fs
print(time)
print(ecg.shape)


### input format 3 - pulse transit time dataset
csv_file = '/Users/jamborghini/Documents/PYTHON/TRAINING_DATA_PULSE_TRANSIT_TIME_PPG_DATASET/csv/s6_walk.csv'
df = pd.read_csv(csv_file)
ecg = df['ecg']
print(ecg)
print(len(ecg))
fs = 500  # sampling rate (Hz)
time = len(ecg) / fs
'''


class Pan_Tompkins_QRS():
    '''
    Pan-Tompkins algorithm to produce mwin signal
    '''

    def __init__(self, fs):
        self.fs = fs

    def band_pass_filter(self, signal):
        '''
        Filter the ECG signal based on frequency of QRS complex
        '''

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y


        lowcut = 5  # Lower cutoff frequency (Hz)
        highcut = 15  # Upper cutoff frequency (Hz)
        result = butter_bandpass_filter(signal, lowcut, highcut, self.fs)

        return result

    def derivative(self, signal):
        '''
        Highlight parts of the signal where slope is steepest - ie QRS complex
        y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)
        '''

        result = signal.copy()

        for index in range(len(signal)):
            result[index] = 0  # starting point

            if (index >= 1):
                result[index] = - 2 * signal[index - 1]  # -2x(nT - T)

            if (index >= 2):
                result[index] = - signal[index - 2]  # -x(nT - 2T)

            if (index >= 2 and index <= len(signal) - 2):
                result[index] += 2 * signal[
                    index + 1]  # 2x(nT + T)

            if (index >= 2 and index <= len(signal) - 3):
                result[index] += signal[index + 2]  # x(nT + 2T)

            result[index] = (result[index] * self.fs) / 8

        return result

    def squaring(self, signal):
        '''
        Intensify highest parts of the derivative signal and attenuate T-wave
        '''

        result = signal.copy()
        for index in range(len(result)):
            result[index] = result[index] ** 2

        return result

    def moving_window_integration(self, signal):
        '''
        Soften the curves of the squared signal to separate further the R-waves from the noise
        y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N
        '''

        # Initialize result and window size for integration
        result = signal.copy()
        window_size = round(0.15 * self.fs)
        sum = 0
        # Calculate the sum for the first N terms
        for j in range(window_size):
            sum += signal[j] / window_size
            result[j] = sum

        # Apply the moving window integration using the equation given
        for index in range(window_size, len(signal)):
            sum += signal[index] / window_size
            sum -= signal[index - window_size] / window_size
            result[index] = sum

        return result

    def solve(self, signal):

        # ensure input signal is numpy array
        if isinstance(signal, np.ndarray):
            input_signal = signal
        else:
            input_signal = signal.to_numpy()

        # Bandpass Filter
        global bpass
        bpass = self.band_pass_filter(input_signal.copy())

        # Derivative Function
        global der
        der = self.derivative(bpass.copy())

        # Squaring Function
        global sqr
        sqr = self.squaring(der.copy())

        # Moving Window Integration Function
        global mwin
        mwin = self.moving_window_integration(sqr.copy())

        return mwin         # moving window integrated signal


# define instance of the object
class Heart_Rate():
    '''
    Finds locations of the peaks in mwin signal
    '''

    def __init__(self, signal, fs):
        '''
        :param signal: input signal
        :param fs: sampling frequency of input signal
        '''

        # RR1 = list of last 8 RR intervals
        # RR2 = list of last 8 RR intervals within an acceptable range
        # peaks = list of all local maxima
        # probable_peaks = list of peaks found from local maxima regions on bandpass filtered signal
        # r_locs = finalised locations of R peaks in bandpassed signal
        # result = actual location of R peaks on raw ECG signal
        self.RR1, self.RR2, self.peaks, self.probable_peaks, self.r_locs, self.result = ([] for _ in range(6))

        # SPKI = running estimate of R peak in mwin signal == highest peak that falls within RR window
        # NPKI = running estimate of noise peak in mwin signal == highest peak that falls within lower threshold
        # Threshold_I1 = higher threshold applied to mwin signal - above this threshold it's likely an R peak
        # Threshold_I2 = lower threshold applied to mwin signal - above this and it may be noise
        # SPKF = running estimate of R peak in bandpass signal
        # NPKF = running estimate of noise peak in bandpass signal
        # Threshold_F1 = first threshold applied to bandpass signal
        # Threshold_F2 = second threshold applied to bandpass signal
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (
        0 for _ in range(8))

        self.T_wave = False
        self.m_win = mwin
        self.b_pass = bpass
        self.fs = fs
        self.signal = signal
        self.win_150ms = round(0.15 * self.fs)

        self.RR_Low_Limit = 0
        self.RR_High_Limit = 0
        self.RR_Missed_Limit = 0
        self.RR_Average1 = 0

    def every_peak(self):
        '''
        Find all local maxima as a first approximate step
        '''

        # further smooth signal via convolution
        slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

        # find every peak location
        for i in range(round(0.5 * self.fs) + 1, len(slopes) - 1):
            if (slopes[i] > slopes[i - 1]) and (slopes[i + 1] < slopes[i]):
                self.peaks.append(i)

        # plt.plot(slopes)
        # plt.scatter(self.peaks, [slopes[i] for i in self.peaks], marker='x', color='red')
        # plt.show()

    def adjust_rr_interval(self, idx):
        '''
        Adjust acceptable localised HR (RR) interval range based on trailing average of intervals
        :param idx: current index in probable_peaks array
        '''

        # Get the 8 most recent RR intervals (RR1)
        self.RR1 = np.diff(self.peaks[max(0, idx - 8): idx + 1]) / self.fs

        # Calculate average interval
        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        # Finding the 8 most recent RR intervals lying between RR Low Limit and RR High Limit
        if (idx >= 8):
            for i in range(0, 8):
                # keep only those that lie in the range (RR2)
                if (self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit):
                    self.RR2.append(self.RR1[i])
                    # keep RR2 length dynamically set to 8
                    if (len(self.RR2) > 8):
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)

        # Reset limits based on average
        if (len(self.RR2) > 7 or idx < 8):
            self.RR_Low_Limit = 0.92 * RR_Average2
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2

    def searchback(self, peak_val, RRn, sb_win):
        '''
        Searchback to see if a local peak was missed
        :param peak_val: peak location in consideration
        :param RRn: the most recent RR interval
        :param sb_win: searchback window
        '''

        # Check if the most recent RR interval is greater than the RR Missed Limit (implying a missed peak)
        if (RRn > self.RR_Missed_Limit):
            # Create searchback window of size RRn in mwin
            win_rr = self.m_win[peak_val - sb_win + 1: peak_val + 1]

            # Find the x locations inside the window having y values greater than Threshold I1
            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

            # Find the x location of the max peak value in the search window within mwin signal
            if (len(coord) > 0):
                for pos in coord:
                    if (win_rr[pos] == max(win_rr[coord])):
                        x_max = pos
                        break
            else:
                x_max = None

            if (x_max is not None):
                # Update mwin thresholds based on this new max peak
                self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                # now search bandpass signal
                win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) - 1, x_max)]

                # Find the x locations inside the window having y values greater than Threshold F1
                coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

                # Find the x location of the max peak value in the search window
                if (len(coord) > 0):
                    for pos in coord:
                        if (win_rr[pos] == max(win_rr[coord])):
                            r_max = pos
                            break
                else:
                    r_max = None

                if (r_max is not None):
                    # Update the thresholds corresponding to bandpass filter
                    if self.b_pass[r_max] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1

                        # Append the location of probable missed peak
                        self.r_locs.append(r_max)

    def find_t_wave(self, peak_val, RRn, idx, prev_idx):
        '''
        Take probable missed peak found during searchback and ensure it isn't a T-wave
        :param peak_val: peak location in consideration
        :param RRn: the most recent RR interval
        :param idx: current index in peaks array
        :param prev_idx: previous index in peaks array
        '''

        if (self.m_win[peak_val] >= self.Threshold_I1):

            # set reasonable interval range for T-waves
            if (idx > 0 and 0.20 < RRn < 0.36):
                # Find the slope of current and last waveform detected
                curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms / 2): peak_val + 1]))
                last_slope = max(
                    np.diff(self.m_win[self.peaks[prev_idx] - round(self.win_150ms / 2): self.peaks[prev_idx] + 1]))

                # If current waveform slope is less than half of last waveform slope, it's likely a T-wave
                if (curr_slope < 0.5 * last_slope):
                    # Update noise threshold
                    self.T_wave = True
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI

            if (not self.T_wave):
                # T-Wave is not found - update signal thresholds
                if (self.probable_peaks[idx] > self.Threshold_F1):
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.b_pass[idx] + 0.875 * self.SPKF

                    # Having confirmed it's not a T-wave, append to final peak list
                    self.r_locs.append(self.probable_peaks[idx])

                else:
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[idx] + 0.875 * self.NPKF

        # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I1) or (self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[idx] + 0.875 * self.NPKF

    def adjust_thresholds(self, peak_val, idx):
        '''
        Global adjustment of signal and noise y-thresholds
        Adjust Noise and Signal Thresholds During Learning Phase
        :param peak_val: peak location in consideration
        :param idx: current index in peaks array
        '''

        # if (self.m_win[peak_val] >= self.Threshold_I1):
        #     # JB EDIT - if a mwin peak is above 10x the average, don't count in the threshold update
        #     # it's counting the T-Waves as well, that's why... fidx a way to move T-Wave up
        #     if (self.m_win[peak_val] >= 10 * np.mean(self.m_win[:peak_val])):
        #         pass
        #     else:
        #         self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI

        if (self.m_win[peak_val] >= self.Threshold_I1):
            # Update signal threshold
            self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI

            if (self.probable_peaks[idx] > self.Threshold_F1):

                self.SPKF = 0.125 * self.b_pass[idx] + 0.875 * self.SPKF

                # Append the probable R peak location
                self.r_locs.append(self.probable_peaks[idx])

            else:
                # Update noise threshold
                self.NPKF = 0.125 * self.b_pass[idx] + 0.875 * self.NPKF

                # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I2) or (self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[idx] + 0.875 * self.NPKF

    def update_thresholds(self):
        '''
        Update Noise and Signal Thresholds for next iteration
        '''

        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def ecg_searchback(self):
        '''
        Map found R peaks back to original ECG signal
        '''

        # Filter the unique R peak locations
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))

        # Initialize a backwards-looking window to searchback (to avoid detecting spurious T-waves)
        win_200ms = round(0.2 * self.fs)

        for r_val in self.r_locs:
            coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + 1), 1)

            # Find the x location of the max peak value
            if (len(coord) > 0):
                for pos in coord:
                    if (self.signal[pos] == max(self.signal[coord])):
                        x_max = pos
                        break
            else:
                x_max = None

            # Append the finalised ECG location
            if (x_max is not None):
                self.result.append(x_max)

    ### custom formula that makes sure no wrong peaks are detected, by checking the rr_interval is not unrealistically low
    # problem is that this wouldn't detect any heart problems....
    def final_check(self):
        min_rr_interval_ms = 250  # Set the minimum RR interval to 250 ms (240bpm)

        # Initialize a list to store the final R peaks
        final_r_peaks = []

        for i in range(len(self.result)):
            if i > 0:
                # Calculate the RR interval between the current peak and the previous peak
                rr_interval = (self.result[i] - self.result[i - 1]) / self.fs
                rr_interval_ms = rr_interval * 1000  # Convert RR interval to milliseconds

                # Check if the RR interval is greater than or equal to the minimum allowed (300 ms)
                if rr_interval_ms >= min_rr_interval_ms:
                    final_r_peaks.append(self.result[i])

        # Update the result with the final R peaks
        self.result = final_r_peaks

    def find_r_peaks(self):
        '''
        Runs all methods in HeartRate()
        '''

        # find every local maximum in smoothed mwin signal
        self.every_peak()

        # iterate through local maxima
        for idx in range(len(self.peaks)):

            # for each local maximum, set up a search window - giving approximate peak regions
            peak_val = self.peaks[idx]
            win_300ms = np.arange(max(0, self.peaks[idx] - self.win_150ms), min(self.peaks[idx] + self.win_150ms, len(self.b_pass) - 1), 1)
            max_val = max(self.b_pass[win_300ms], default=0)        # find maximum value in the search window

            # use the approximate peak regions to find the peak locations in the original bandpassed signal
            if (max_val != 0):
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0])

            # update rolling RR interval acceptability limits
            if (idx < len(self.probable_peaks) and idx != 0):
                self.adjust_rr_interval(idx)

                # Adjust y-axis thresholds
                if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit):
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2

                # Searchback to make sure we haven't missed a peak
                RRn = self.RR1[-1]          # set RRn to most recent RR interval
                self.searchback(peak_val, RRn, sb_win=round(RRn * self.fs))

                # Confirm the searchback isn't picking up T-waves instead of R peaks
                self.find_t_wave(peak_val, RRn, idx, idx - 1)

            else:
                # Adjust SPKI and NPKI
                self.adjust_thresholds(peak_val, idx)

            # update thresholds for the next iteration (local maximum to be analysed)
            self.update_thresholds()

        # Searchback in ECG signal
        self.ecg_searchback()

        # Custom formula that removes stupid-low RR-intervals
        # self.final_check()

        return np.array(self.result).astype(int), self.r_locs, np.array(self.probable_peaks).astype(int)



def plot_peaks(signal, result, r_locs, probable):

    # plt.plot(signal, color='blue')
    # plt.scatter(result, signal[result], color='red', s=50, marker='*')
    # # for i, (xi, yi) in enumerate(zip(result, signal[result])):
    # #     plt.text(xi, yi, str(i), fontsize=12, ha='center', va='bottom')

    # plt.plot(bpass, color = 'red')
    # plt.plot(der / 100, color = 'black')
    # plt.plot(sqr / 10000000, color = 'black')
    # plt.plot(mwin / 1000000, color = 'black')
    # for peak in probable_peaks:
    #     plt.axvline(x=peak, color='r', linestyle='--')

    plt.show()


def main(ecg,fs):

    print(ecg.shape)

    # produce mwin signal
    mwin = Pan_Tompkins_QRS(fs=fs).solve(ecg)
    # plt.plot(mwin)
    # plt.show()

    # find locations of peaks
    result, r_locs, probable = Heart_Rate(signal=mwin, fs=fs).find_r_peaks()

    # plot results
    print(result.shape)
    print(r_locs.shape)
    print(probable.shape)
    plot_peaks(ecg, result, r_locs, probable)


if __name__ == '__main__':
    main()