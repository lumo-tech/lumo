#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.

# Updated: May 2017 for Speaker Recognition collaboration.

from .audio_tools import *
import numpy as np


#### Display tools
def plot_this(s, title=''):
    """
     
    """
    import pylab
    s = s.squeeze()
    if s.ndim == 1:
        pylab.plot(s)
    else:
        pylab.imshow(s, aspect='auto')
        pylab.title(title)
    pylab.show()


def plot_these(s1, s2):
    import pylab
    try:
        # If values are numpy arrays
        pylab.plot(s1 / max(abs(s1)), color='red')
        pylab.plot(s2 / max(abs(s2)), color='blue')
    except:
        # Values are lists
        pylab.plot(s1, color='red')
        pylab.plot(s2, color='blue')
    pylab.legend()
    pylab.show()


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes, axis=1)
    xframes = xframes - np.tile(m, (xframes.shape[1], 1)).T
    return xframes


def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes, xframes.T)) / float(n_frames)


def compute_log_nrg(xframes):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes + 1e-5)) / float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs)) / (np.sqrt(np.var(raw_nrgs)))


def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes, axis=1)
    X = np.abs(X[:, :X.shape[1] / 2]) ** 2
    return np.sqrt(X)


def nrg_vad(xframes, percent_thr, nrg_thr=0., context=5):
    """
        Picks frames with high energy as determined by a 
        user defined threshold.
        
        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding. 
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.
        
        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        """
    xframes = zero_mean(xframes)
    n_frames = xframes.shape[0]

    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames, 1))
    for i in range(n_frames):
        start = max(i - context, 0)
        end = min(i + context, n_frames - 1)
        n_above_thr = np.sum(xnrgs[start:end] > nrg_thr)
        n_total = end - start + 1
        xvad[i] = 1. * ((float(n_above_thr) / n_total) > percent_thr)
    return xvad


def vad_x(file):
    fs, ns, s = read_wav(test_file)
    win_len = int(fs * 0.025)
    hop_len = int(fs * 0.010)
    sframes = enframe(ns, win_len, hop_len)  # rows: frame index, cols: each frame
    percent_high_nrg = 0.5
    vad = nrg_vad(sframes, percent_high_nrg)
    x_samples = deframe(vad, win_len, hop_len)
    return ns[np.where(x_samples.squeeze().astype(np.int) == 1)]



if __name__ == '__main__':
    # test_file='data/sa1.wav'
    test_file = 'data/L001-train-000.wav'
    fs, ns, s = read_wav(test_file)
    win_len = int(fs * 0.025)
    hop_len = int(fs * 0.010)
    sframes = enframe(ns, win_len, hop_len)  # rows: frame index, cols: each frame
    plot_this(compute_log_nrg(sframes))

    # percent_high_nrg is the VAD context ratio. It helps smooth the
    # output VAD decisions. Higher values are more strict.
    percent_high_nrg = 0.5

    vad = nrg_vad(sframes, percent_high_nrg)
    x_samples = deframe(vad, win_len, hop_len)
    plot_these(x_samples, ns)
    from matplotlib import pyplot as plt

    plt.plot(ns[np.where(x_samples.squeeze().astype(np.int) == 1)])
    plt.show()
