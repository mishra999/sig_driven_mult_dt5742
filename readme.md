# sdm.py
1-158, DT5742 parsers to read files
185-282, get timing and amp of original and diode signals
284-305, plot 2D hist of SiPM vs diode signal with rational function fitting
307- 355, divide the amplitude into two groups: fit smaller amplitudes with second order poly and larger amps with linear fit using symfit (didn't work!)
358-389, rational function fitting

438-470, 'DT5742_read' to read file for each channel for impulse response and save it as npy file
472, get impulse response
522, read npy files, get impulse response and save it as npy
547, read impulse responses from npy files
557-562, rotate the array
583-590, butterworth filter
617-697, get energy and timing
1034-1064, get mean and std automatically using 'find_single_pulse_mean_var'

# 8to1_new is the same as sdm.py