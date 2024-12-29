'''
Finds QRS peaks in ECG signals finder using Pan-Tompkins algorithm
Used to create labels for smartwatch HR estimation
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

# ### input format 1 - Wrist PPG dataset
#
# file_path = '/Users/jamborghini/Documents/PYTHON/TESTING DATA - Wrist PPG Dataset/'
# session = 's3_run'  # adjust this
#
# file_name = file_path + session
# record = wfdb.rdrecord(file_name)
# # from .hea file: 0 = ecg, 1 = ppg, 2-4 = gyro, 5-7 = 2g accelerometer, 8-10 = 16g accelerometer
# # ecg = pd.DataFrame({'ecg': record.adc()[:, 0]}) - old method that doesnt work because it doesn't denote the ecg column properly
# df = pd.DataFrame({'ecg': record.adc()[:, 0]})
# ecg = df['ecg']
# df = pd.DataFrame({'ppg': record.adc()[:, 1]})
# ppg = df['ppg']
# print(ecg)
# print(len(ecg))
# fs = 256  # from the paper
# time = len(ecg) / fs  # this is the total time in seconds

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


######## PART 1: PRE-PROCESSING ########################################################################

class Pan_Tompkins_QRS():
    # below is the P-T defined bandpassing. But doesn't work as well for ECG. bizarre...
    '''
    def band_pass_filter(self, signal):
        result = None

        # high and low-pass filter recursive equations from original Pan-Tompkins paper (1985)
        # low: y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)
        y = signal.copy()
        for index in range(len(signal)):
            y[index] = signal[index]   # this is the +x(nT) term
            # now add each term in turn with +=
            if (index >= 1):
                y[index] += 2 * y[index - 1]

            if (index >= 2):
                y[index] -= y[index - 2]

            if (index >= 6):
                y[index] -= 2 * signal[index - 6]

            if (index >= 12):
                y[index] += signal[index - 12]

        result = y.copy()

        # high: y(nT) = 32x(nT - 16T) - y(nT - T) - x(nT) + x(nT - 32T)
        for index in range(len(signal)):
            result[index] = -1 * y[index]   # -x(nT) term (x is now y for the second equation)
            if (index >= 1):
                result[index] -= result[index - 1]

            if (index >= 16):
                result[index] += 32 * y[index - 16]

            if (index >= 32):
                result[index] += y[index - 32]

        # normalise the result
        max_val = max(max(result),-min(result))
        result = result/max_val

        return result
    '''

    def band_pass_filter(self, signal):

        result = None

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

        # Remember this is the frequency of the QRS complex... can't equate it to heartrate
        lowcut = 5  # Lower cutoff frequency (Hz)
        highcut = 15  # Upper cutoff frequency (Hz)
        # Apply the bandpass filter
        result = butter_bandpass_filter(signal, lowcut, highcut, fs)

        return result

    ######## TAKE DERIVATIVE
    # purpose is to highlight the key parts of the signal where slope is steepest, hence taking out the noise between beats
    # equation from paper: y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)
    # ^finite difference equation

    def derivative(self, signal):
        result = signal.copy()

        for index in range(len(signal)):
            result[index] = 0  # this is the starting point, since there is no x(nT) term this time

            if (index >= 1):
                result[index] = - 2 * signal[index - 1]  # -2x(nT - T)

            if (index >= 2):
                result[index] = - signal[index - 2]  # -x(nT - 2T)

            if (index >= 2 and index <= len(signal) - 2):
                result[index] += 2 * signal[
                    index + 1]  # 2x(nT + T) - must go up to len(signal) - 2 so it doesn't exceed the bounds of the signal since it's +T (leaving one final point)

            if (index >= 2 and index <= len(signal) - 3):
                result[index] += signal[index + 2]  # x(nT + 2T)

            result[index] = (result[index] * fs) / 8

        return result

    ######## SQUARE THE SIGNAL
    # intensify important part of the derivative signal and attenuate 'T' parts of the signal

    def squaring(self, signal):
        result = signal.copy()
        for index in range(len(result)):
            result[index] = result[index] ** 2

        return result

    ######## MOVING WINDOW INTEGRATOR
    # 'soften' the curves of the squared signal, so the area of peaks (R waves) can easily be detected as humps
    # y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N

    def moving_window_integration(self, signal):

        # Initialize result and window size for integration
        result = signal.copy()
        window_size = round(0.15 * fs)
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
        ### combine all of the above to spit out preprocessed signal

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

        return mwin

        ## mwin = moving window integrated signal





######## PART 2: CALCULATE HEART RATE ########################################################################

# define instance of the object
class heart_rate():

    def __init__(self, signal, fs):
        '''
        Initialize Variables
        :param signal: input signal
        :param fs: sample frequency of input signal
        '''

        # Initialize variables
        self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (
        0 for i in range(8))
        # SPKI = running estimate of signal (R) peak. It's the HIGHEST PEAK that falls within an RR window
        # NPKI = running estimate of noise peak. It's the highest peak that falls BELOW the lower threshold (distinguishing it from an R peak)
        # I1, I2 thresholds used on moving average resultant signal; F1, F2 used on the bandpassed ecg signal. Done on both signals for increased accuracy.
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

    def approx_peak(self):
        '''
        Approximate peak locations by finding local maxima (could be R peaks and T waves)
        '''

        # FFT convolution (not transforming into the frequency domain, but for simply processing the signal)
        slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

        # Finding approximate peak locations
        for i in range(round(0.5 * self.fs) + 1, len(slopes) - 1):
            if (slopes[i] > slopes[i - 1]) and (slopes[i + 1] < slopes[i]):
                self.peaks.append(i)

    # THIS FUNCTION IS NOT ACTIVE - why did I choose to omit it?
    def adjust_rr_interval(self, ind):
        '''
        Adjust RR Interval and Limits
        :param ind: current index in peaks array
        '''

        # Finding the eight most recent RR intervals
        self.RR1 = np.diff(self.peaks[max(0, ind - 8): ind + 1]) / self.fs

        # Calculating RR Averages
        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        # Finding the eight most recent RR intervals lying between RR Low Limit and RR High Limit
        if (ind >= 8):
            for i in range(0, 8):
                if (self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit):
                    self.RR2.append(self.RR1[i])

                    if (len(self.RR2) > 8):
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)

        # Adjusting the RR Low Limit and RR High Limit
        if (len(self.RR2) > 7 or ind < 8):
            self.RR_Low_Limit = 0.92 * RR_Average2
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2

    def searchback(self, peak_val, RRn, sb_win):
        # point here is to see if we missed a beat on a local scale

        '''
        Searchback
        :param peak_val: peak location in consideration
        :param RRn: the most recent RR interval
        :param sb_win: searchback window
        '''

        # Round 1 - search in m_win
        # Check if the most recent RR interval is greater than the RR Missed Limit (ie implying we missed an R peak)
        if (RRn > self.RR_Missed_Limit):
            # Initialize a window to searchback
            win_rr = self.m_win[peak_val - sb_win + 1: peak_val + 1]

            # Find the x locations inside the window having y values greater than Threshold I1
            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

            # Find the x location of the max peak value in the search window
            if (len(coord) > 0):
                for pos in coord:
                    if (win_rr[pos] == max(win_rr[coord])):
                        x_max = pos
                        break
            else:
                x_max = None

            # If the max peak value is found
            if (x_max is not None):
                # Update the thresholds corresponding to moving window integration
                # Update SPKI and other thresholds
                self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                # Round 2 - search in bandpass
                # only searches if a mwin peak has been found for confirmation (computational savings)
                # Initialize a window to searchback
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

                # If the max peak value is found
                if (r_max is not None):
                    # Update the thresholds corresponding to bandpass filter
                    if self.b_pass[r_max] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1

                        # Append the probable R peak location
                        self.r_locs.append(r_max)

    def find_t_wave(self, peak_val, RRn, ind, prev_ind):

        # now check if the idenfied potential missed beat is a T-wave instead of an R-wave
        '''
        T Wave Identification
        :param peak_val: peak location in consideration
        :param RRn: the most recent RR interval
        :param ind: current index in peaks array
        :param prev_ind: previous index in peaks array
        '''
        if (self.m_win[peak_val] >= self.Threshold_I1):
            if (ind > 0 and 0.20 < RRn < 0.36):
                # Find the slope of current and last waveform detected
                curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms / 2): peak_val + 1]))
                last_slope = max(
                    np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms / 2): self.peaks[prev_ind] + 1]))

                # If current waveform slope is less than half of last waveform slope, it's a T wave
                if (curr_slope < 0.5 * last_slope):
                    # T Wave is found and update noise threshold
                    self.T_wave = True
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI

            if (not self.T_wave):
                # T Wave is not found and update signal thresholds
                if (self.probable_peaks[ind] > self.Threshold_F1):
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                    # Append the probable R peak location
                    self.r_locs.append(self.probable_peaks[ind])

                else:
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                    # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I1) or (
                self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def adjust_thresholds(self, peak_val, ind):

        # now we recalculate SPKI and NPKI on a GLOBAL scale

        '''
        Adjust Noise and Signal Thresholds During Learning Phase
        :param peak_val: peak location in consideration
        :param ind: current index in peaks array
        '''

        if (self.m_win[peak_val] >= self.Threshold_I1):
            # JB EDIT - if a m_win peak is above 10x the average, don't count in the threshold update
            # it's counting the T-Waves as well, that's why... find a way to move T-Wave up
            if (self.m_win[peak_val] >= 10 * np.mean(self.m_win[:peak_val])):
                pass
            else:
                self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI

            if (self.probable_peaks[ind] > self.Threshold_F1):

                self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                # Append the probable R peak location
                self.r_locs.append(self.probable_peaks[ind])

            else:
                # Update noise threshold
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I2) or (
                self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

        # print(np.mean(self.probable_peaks[ind]))

    def update_thresholds(self):

        # apply these new updated thresholds to the whole signal

        '''
        Update Noise and Signal Thresholds for next iteration
        '''

        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def ecg_searchback(self):

        ## this is where we map results back to the original ECG signal
        # we use the indices of peaks found in m_win and set up a window around it, placed over the original ECG signal to find where the R wave REALLY lies
        '''
        Searchback in ECG signal to increase efficiency
        '''

        # Filter the unique R peak locations
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))

        # Initialize a window to searchback
        win_200ms = round(0.2 * self.fs)

        # TINY CUSTOM CHANGE to code - only have the window look backwards instead of forwards
        # because the R peak will always lie before the m_avg peak (due to T-wave following) - original code has it either side
        # but if either side it picks up T-waves that are higher than the nearby R peak as the real peak - false positive

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

            # Append the peak location
            if (x_max is not None):
                self.result.append(x_max)

                ### x_max are the finalised R peaks

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

    # Finally run core function - combining all the previous 7 functions

    def find_r_peaks(self):
        '''
        R Peak Detection
        '''

        # Find approx peak location estimations using the mwin signal
        self.approx_peak()

        for ind in range(len(self.peaks)):

            # In the areas of the approximations, set up a search window
            peak_val = self.peaks[ind]
            win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms),
                                  min(self.peaks[ind] + self.win_150ms, len(self.b_pass) - 1), 1)
            max_val = max(self.b_pass[win_300ms], default=0)

            # Find the x location of the max peak value within this window to get 'true' peak
            # This is in the BPASS SIGNAL, not mwin
            if (max_val != 0):
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0])

            # Adjust RR interval thresholds in case of irregular beats
            if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit):
                self.Threshold_I1 /= 2
                self.Threshold_F1 /= 2

                RRn = self.RR1[-1]

                # Searchback to make sure we haven't missed a peak
                self.searchback(peak_val, RRn, round(RRn * self.fs))

                # Confirm the searchback isn't picking up T-waves instead of R peaks
                self.find_t_wave(peak_val, RRn, ind, ind - 1)

            else:
                # Adjust SPKI and NPKI
                self.adjust_thresholds(peak_val, ind)

            # Update Thresholds
            self.update_thresholds()

        # Searchback in ECG signal
        self.ecg_searchback()

        # Custom formula that removes stupid-low RR-intervals
        self.final_check()

        return self.result, self.probable_peaks, self.r_locs


######## PRESENTING DATA ########################################################################

# Convert ecg signal to numpy array

if isinstance(ecg, np.ndarray):
    signal = ecg
else:
    signal = ecg.to_numpy()

# Find the R peak locations
hr = heart_rate(signal, fs)
result, probable_peaks, r_locs = hr.find_r_peaks()
result = np.array(result)
probable_peaks = np.array(probable_peaks)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Calculate the overall heart rate
heartRate = (60 * fs) / np.average(np.diff(result[1:]))
print("Heart Rate", heartRate, "BPM")

### set up the sliding 8 second window for model prep
window_size = 8 * fs  # 8-second window size
step_size = 2 * fs  # 2-second step size
heart_rates = []
rmssd_values = []
rr_intervals_matrix = []

for i in range(0, len(signal) - window_size + 1, step_size):
    window_r_peaks = [peak for peak in result if i <= peak < i + window_size]
    # ^ 'list comprehension' - remember this format

    if len(window_r_peaks) >= 2:
        window_bpm = (60 * fs) / np.average(np.diff(window_r_peaks))
        heart_rates.append(window_bpm)

        rr_intervals = np.diff(window_r_peaks)  # convert to ms
        rr_intervals = rr_intervals * 1000 / fs
        rr_intervals_matrix.append(rr_intervals)

        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        rmssd_values.append(rmssd)

# SAVE HEARTRATES TO EXTERNAL FILE
# with open(session + '_heart_rate_wesad.pkl', 'wb') as file:
#     pickle.dump(heart_rates, file)


'''
output_csv_file = 'OUTPUT.csv'
try:
    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rr_intervals_matrix)
except Exception as e:
    print(f"An error occurred: {e}")
'''

rr_intervals = np.diff(result)
# plt.plot(rmssd_values)
# plt.plot(rmssd_values)
plt.plot(heart_rates, color='black')
plt.show()

plt.plot(signal, color='blue')
plt.scatter(result, signal[result], color='red', s=50, marker='*')
for i, (xi, yi) in enumerate(zip(result, signal[result])):
    plt.text(xi, yi, str(i), fontsize=12, ha='center', va='bottom')

## Check each stage (signal, bpass, der, sqr, mwin)
# plt.plot(bpass, color = 'red')
# plt.plot(der / 100, color = 'black')
# plt.plot(sqr / 10000000, color = 'black')
# plt.plot(mwin / 1000000, color = 'black')
# for peak in probable_peaks:
#     plt.axvline(x=peak, color='r', linestyle='--')

plt.show()

def main():

    

    # create instance & run peak detection
    QRS_detector = Pan_Tompkins_QRS()
    output_signal = QRS_detector.solve(ecg)

if __name__ == '__main__':
    main()