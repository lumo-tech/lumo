import numpy as np
from scipy.io import wavfile


def add_wgn(s, var=1e-4):
    """
        Add white Gaussian noise to signal
        If no variance is given, simply add jitter. 
        Jitter helps eliminate all-zero values.
        """
    np.random.seed(0)
    noise = np.random.normal(0, var, len(s))
    return s + noise


def read_wav(filename):
    """
        read wav file.
        Normalizes signal to values between -1 and 1.
        Also add some jitter to remove all-zero segments."""
    fs, s = wavfile.read(filename)  # scipy reads int
    s = np.array(s)
    ns = add_wgn(s)  # Add jitter for numerical stability
    return fs, s, ns


def enframe(x, win_len, hop_len):
    """
        receives a 1D numpy array and divides it into frames.
        outputs a numpy matrix with the frames on the rows.
    """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
    x_framed = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        x_framed[i] = x[i * hop_len: i * hop_len + win_len]
    return x_framed


def deframe(x_framed, win_len, hop_len):
    '''
        interpolates 1D data with framed alignments into persample values.
        This function helps as a visual aid and can also be used to change 
        frame-rate for features, e.g. energy, zero-crossing, etc.
        '''
    n_frames = len(x_framed)
    n_samples = n_frames * hop_len + win_len
    x_samples = np.zeros((n_samples, 1))
    for i in range(n_frames):
        x_samples[i * hop_len: i * hop_len + win_len] = x_framed[i]
    return x_samples


if __name__ == '__main__':
    pass
