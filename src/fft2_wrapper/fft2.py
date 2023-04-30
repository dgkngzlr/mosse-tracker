import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def fft2_forward(image):
    return fft2(image)

def fft2_backward(spectrum):
    """
        If image double use np.astype(np.float32) to get orginal image
        If image uint8  use np.astype(np.uint8) to get orginal image
    """
    return ifft2(spectrum) # Default scaled with 1/N

def fft2shift(spectrum):
    return fftshift(spectrum)

def ifft2shift(spectrum):
    return ifftshift(spectrum)

def real(spectrum):
    return np.real(spectrum)

def imag(spectrum):
    return np.imag(spectrum)

def conj(spectrum):
    return np.conjugate(spectrum)

def magnitude(spectrum):
    return np.abs(spectrum)

def phase(spectrum):
    return np.arctan2(imag(spectrum), real(spectrum))

def mul_spectrum(spectrum1, spectrum2):
    return spectrum1 * spectrum2

def div_spectrum(spectrum1, spectrum2):
    return spectrum1 / spectrum2

def visualize_spectrum(spectrum):

    # Shift the FFT result
    shifted_fft = fft2shift(spectrum)

    # Calculate the magnitude spectrum
    magnitude_spectrum = magnitude(shifted_fft)

    # Logarithm of the magnitude spectrum
    log_magnitude_spectrum = np.log(1 + magnitude_spectrum)

    # Plot the logarithm of the magnitude spectrum
    plt.subplot(1, 1, 1)
    plt.title('Log Magnitude Spectrum')
    plt.imshow(log_magnitude_spectrum, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()