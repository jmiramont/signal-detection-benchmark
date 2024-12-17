""" 
This file contains utilities for time-frequency analysis. 
"""

import numpy as np
from scipy.fft import fft, ifft
from numpy import pi as pi


def get_gauss_window(Nfft,L,prec=1e-6):
    l=np.floor(np.sqrt(-Nfft*np.log(prec)/pi))+1
    N = 2*l+1
    t0 = l+1
    tmt0=np.arange(0,N)-t0
    g = np.exp(-(tmt0/L)**2 * pi)    
    g=g/np.linalg.norm(g)
    return g

def get_round_window(Nfft, prec = 1e-6):
    """ Generates a round Gaussian window, i.e. same essential support in time and 
    frequency: g(n) = exp(-pi*(n/T)^2) for computing the Short-Time Fourier Transform.
    
    Args:
        Nfft: Number of samples of the desired fft.

    Returns:
        g (ndarray): A round Gaussian window.
        T (float): The scale of the Gaussian window (T = sqrt(Nfft))
    """
    # analysis window
    L=np.sqrt(Nfft)
    l=np.floor(np.sqrt(-Nfft*np.log(prec)/pi))+1
 
    N = 2*l+1
    t0 = l+1
    tmt0=np.arange(0,N)-t0
    g = np.exp(-(tmt0/L)**2 * pi)    
    g=g/np.linalg.norm(g)
    return g, L

def get_stft(x,window=None,t=None,Nfft=None):
    xrow = len(x)

    if t is None:
        t = np.arange(0,xrow)

    if Nfft is None:
        Nfft = 2*xrow

    if window is None:
        window = get_round_window(Nfft)

    tcol = len(t)
    hlength=np.floor(Nfft/4)
    hlength=int(hlength+1-np.remainder(hlength,2))

    hrow=len(window)
    
    assert np.remainder(hrow,2) == 1

    Lh=(hrow-1)//2
    tfr= np.zeros((Nfft,tcol))   
    for icol in range(0,tcol):
        ti= t[icol]; 
        tau=np.arange(-np.min([np.round(Nfft/2),Lh,ti]),np.min([np.round(Nfft/2),Lh,xrow-ti])).astype(int)
        indices= np.remainder(Nfft+tau,Nfft).astype(int); 
        tfr[indices,icol]=x[ti+tau]*np.conj(window[Lh+tau])/np.linalg.norm(window[Lh+tau])
    
    tfr=fft(tfr, axis=0) 
    return tfr

def get_istft(tfr,window=None,t=None):
    
    N,NbPoints = tfr.shape
    tcol = len(t)
    hrow = len(window) 
    Lh=(hrow-1)//2
    window=window/np.linalg.norm(window)
    tfr=ifft(tfr,axis=0)

    x=np.zeros((tcol,),dtype=complex)
    for icol in range(0,tcol):
        valuestj=np.arange(np.max([1,icol-N/2,icol-Lh]),np.min([tcol,icol+N/2,icol+Lh])).astype(int)
        for tj in valuestj:
            tau=icol-tj 
            indices= np.remainder(N+tau,N).astype(int)
            x[icol]=x[icol]+tfr[indices,tj]*window[Lh+tau]
        
        x[icol]=x[icol]/np.sum(np.abs(window[Lh+icol-valuestj])**2)
    return x

def get_spectrogram(signal,window=None,Nfft=None,t=None,onesided=True):
    """
    Get the round spectrogram of the signal computed with a given window. 
    
    Args:
        signal(ndarray): A vector with the signal to analyse.

    Returns:
        S(ndarray): Spectrogram of the signal.
        stft: Short-time Fourier transform of the signal.
        stft_padded: Short-time Fourier transform of the padded signal.
        Npad: Number of zeros added in the zero-padding process.
    """

    N = np.max(signal.shape)
    if Nfft is None:
        Nfft = 2*N

    if window is None:
        window, _ = get_round_window(Nfft)

    if t is None:
        t = np.arange(0,N)
        
    stft=get_stft(signal,window=window,t=t,Nfft=Nfft)

    if onesided:
        S = np.abs(stft[0:Nfft//2+1,:])**2
    else:
        S = np.abs(stft)**2                
    return S, stft


def find_zeros_of_spectrogram(S):
    aux_S = np.zeros((S.shape[0]+2,S.shape[1]+2))+np.Inf
    aux_S[1:-1,1:-1] = S
    S = aux_S
    aux_ceros = ((S <= np.roll(S,  1, 0)) &
            (S <= np.roll(S, -1, 0)) &
            (S <= np.roll(S,  1, 1)) &
            (S <= np.roll(S, -1, 1)) &
            (S <= np.roll(S, [-1, -1], [0,1])) &
            (S <= np.roll(S, [1, 1], [0,1])) &
            (S <= np.roll(S, [-1, 1], [0,1])) &
            (S <= np.roll(S, [1, -1], [0,1])) 
            )
    [y, x] = np.where(aux_ceros==True)
    pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
    pos[:, 0] = y-1
    pos[:, 1] = x-1
    return pos
