import librosa
import numpy as np
import torch
# import cupy as cp


def butterworth_lowpass_coefficients(cutoff, fs, order=2):
    cutoff = torch.tensor([cutoff], dtype=torch.float64)
    fs = torch.tensor([fs], dtype=torch.float64)
    omega = torch.tan(torch.pi * cutoff / fs)
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))  # 정수 2를 텐서로 변환
    Q = 1 / sqrt2    # Butterworth filter's quality factor

    # Transfer function coefficients using the bilinear transform
    b0 = omega**2 / (1 + sqrt2 * omega + omega**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2 * (omega**2 - 1) / (1 + sqrt2 * omega + omega**2)
    a2 = (1 - sqrt2 * omega + omega**2) / (1 + sqrt2 * omega + omega**2)
    
    b = torch.tensor([b0.item(), b1.item(), b2.item()], dtype=torch.float64)
    a = torch.tensor([1.0, a1.item(), a2.item()], dtype=torch.float64)
    return b, a

def butterworth_lowpass_filter(signal, cutoff, fs, order=2):
    b, a = butterworth_lowpass_coefficients(cutoff, fs, order)
    filtered_signal = torch.zeros_like(signal)
    
    # Applying the filter to the signal
    for n in range(len(signal)):
        if n < 2:
            filtered_signal[n] = b[0] * signal[n]
        else:
            filtered_signal[n] = b[0]*signal[n] + b[1]*signal[n-1] + b[2]*signal[n-2] \
                                - a[1]*filtered_signal[n-1] - a[2]*filtered_signal[n-2]

    return filtered_signal

def hilbert(signal):
    """힐버트 변환을 수동으로 구현"""
    N = signal.shape[2]  # 신호 길이
    FFT_signal = torch.fft.fft(signal, axis=2)
    h = torch.zeros_like(signal)  # 신호와 동일한 형상의 배열 생성

    if N % 2 == 0:
        h[:, 0, 0] = 1
        h[:, 0, N//2] = 1
        h[:, 0, 1:N//2] = 2
    else:
        h[:, 0, 0] = 1
        h[:, 0, 1:(N+1)//2] = 2

    return torch.fft.ifft(FFT_signal * h, axis=2)


class Envelope:
    def __init__(self, x, max_freq, order=2):    
        self.signal = x
        self.sr = 24000
        self.max_freq = max_freq
        if self.max_freq == 0:
            self.cutoff = 0  # 또는 적절한 기본값 설정
        else:
            self.cutoff = self.max_freq
        self.order = order
        self.envelope = torch.tensor(self.apply_lowpass_and_compute_envelope()).float()
        if torch.cuda.is_available():
            self.envelope = self.envelope.cuda()  # GPU가 사용 가능하면, GPU로 이동

    def apply_lowpass_and_compute_envelope(self):
        self.signal = self.signal
        if self.max_freq == -1:
            envelope_lower = -torch.abs(hilbert(-self.signal))
            return envelope_lower
        elif self.max_freq == 0:
            return self.signal
        elif self.max_freq == 1:
            envelope_upper = torch.abs(hilbert(self.signal))
            return envelope_upper
        else:
            filtered_signal  = butterworth_lowpass_filter(self.signal, self.cutoff, fs=self.sr)
            return torch.abs(hilbert(filtered_signal))
        
        
# def butterworth_lowpass_coefficients(cutoff, fs, order=2):
#     omega = cp.tan(cp.pi * cutoff / fs)
#     Q = 1 / cp.sqrt(2)  # Butterworth filter's quality factor

#     # Transfer function coefficients using the bilinear transform
#     b0 = omega**2 / (1 + cp.sqrt(2)*omega + omega**2)
#     b1 = 2 * b0
#     b2 = b0
#     a1 = 2 * (omega**2 - 1) / (1 + cp.sqrt(2)*omega + omega**2)
#     a2 = (1 - cp.sqrt(2)*omega + omega**2) / (1 + cp.sqrt(2)*omega + omega**2)
#     b = [b0, b1, b2]
#     a = [1, a1, a2]

#     return b, a

# def butterworth_lowpass_filter(signal, cutoff, fs, order=2):
#     b, a = butterworth_lowpass_coefficients(cutoff, fs, order)
#     filtered_signal = cp.zeros_like(signal)
    
#     # Applying the filter to the signal
#     for n in range(len(signal)):
#         if n < 2:
#             filtered_signal[n] = b[0] * signal[n]
#         else:
#             filtered_signal[n] = b[0]*signal[n] + b[1]*signal[n-1] + b[2]*signal[n-2] \
#                                 - a[1]*filtered_signal[n-1] - a[2]*filtered_signal[n-2]

#     return filtered_signal

# def hilbert(signal):
#     """힐버트 변환을 수동으로 구현"""
#     N = signal.shape[2]  # 신호 길이
#     FFT_signal = cp.fft.fft(signal, axis=2)
#     h = cp.zeros_like(signal)  # 신호와 동일한 형상의 배열 생성

#     if N % 2 == 0:
#         h[:, 0, 0] = 1
#         h[:, 0, N//2] = 1
#         h[:, 0, 1:N//2] = 2
#     else:
#         h[:, 0, 0] = 1
#         h[:, 0, 1:(N+1)//2] = 2

#     return cp.fft.ifft(FFT_signal * h, axis=2)


# class Envelope:
#     def __init__(self, x, max_freq, order=2):    
#         self.signal = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
#         self.sr = 22050
#         self.max_freq = max_freq
#         if self.max_freq == 0:
#             self.cutoff = 0  # 또는 적절한 기본값 설정
#         else:
#             self.cutoff = self.max_freq
#         self.order = order
#         self.envelope = torch.tensor(self.apply_lowpass_and_compute_envelope()).float()
#         if torch.cuda.is_available():
#             self.envelope = self.envelope.cuda()  # GPU가 사용 가능하면, GPU로 이동

#     def apply_lowpass_and_compute_envelope(self):
#         self.signal = self.signal
#         if self.max_freq == -1:
#             envelope_lower = -cp.abs(hilbert(-self.signal))
#             return envelope_lower
#         elif self.max_freq == 0:
#             return self.signal
#         elif self.max_freq == 1:
#             envelope_upper = cp.abs(hilbert(self.signal))
#             return envelope_upper
#         else:
#             filtered_signal  = butterworth_lowpass_filter(self.signal, self.cutoff, fs=self.sr)
#             return cp.abs(hilbert(filtered_signal))
        