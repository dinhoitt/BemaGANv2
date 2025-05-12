import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

# wav file을 불러옴
def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

# 데이터를 clip_val ~ a_max(최대값 제한 없음)로 제한
# 이후, 압축계수 C를 곱함
# 최종적으로 데이터에 자연로그
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

# 압축 풀기~
def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

# torch로 압축
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

# 아 스펙트럼 노말라이징
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# 디노말
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

# dictionary로 설정된 애들 
mel_basis = {}
hann_window = {}

# mel spectrogram 정의
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # 입력 신호의 최대 최솟값이 |1|을 넘지 않도록, 경고 표시 
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    # mel_basis, hann_window를 전역 변수로 받아옴
    global mel_basis, hann_window
    # mel_basis에 fmax가 정의되어 있지 않으면, librosa_mel_fn을 통해 mel 필터뱅크 생성
    # mel filter bank : 주파수 스펙트럼을 mel scale로 변환하는데 사용되는 필터
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # 계산된 mel 필터뱅크를 텐서로 변환하고, 장치에 이동
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        # hann_window도 이동 
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # 입력 신호에 대한 패딩을 적용, STFT 수행 전, 양 끝에 반사 패딩을 추가 
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # STFT을 수행해서 스펙트럼 계산
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    # 스펙트럼에 제곱근
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    # mel 스페그로그램을 필터 뱅크를 적용하여 계싼 
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec) # 압축 적용 

    return spec

# 훈련 파일 리스트를 읽어서 training_files 리스트에 저장
def get_dataset_filelist(a):
    # 파일경로 : input_wavs_dir + 파일 이름을 조합
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    # 검사 파일 저장 
    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

# 데이터 셋을 받아서 Mel 스펙트로그램으로 변환
class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files # 오디오 파일 리스트 
        random.seed(1234) # 랜덤 시드 저장 
        # 파일 셔플 여부
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft 
        self.num_mels = num_mels # mel의 갯수 
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss # loss에 사용될 최대 주파수
        self.cached_wav = None # 캐시된 오디오 데이터 
        self.n_cache_reuse = n_cache_reuse # 캐시 재사용 횟수 
        self._cache_ref_count = 0 # 캐시 참조 카운터
        self.device = device # 연산에 사용할 디바이스
        self.fine_tuning = fine_tuning # 미세 조정 모드 활성화 여부 
        self.base_mels_path = base_mels_path # 기존 mel 스펙트로그램 파일 경로

    # 데이터셋에서 특정 인덱스의 아이템을 가져옴 
    def __getitem__(self, index):
        filename = self.audio_files[index] # 인덱스에 해당하는 파일 이름
        # 캐시된 데이터가 없으면 오디오 파일 로드 
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename) # 오디오 파일 로드
            audio = audio / MAX_WAV_VALUE # 정규화
            # 미세 조정 모드가 아니면, 추가 정규하 시행 
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio # 오디오 캐싱 생성
            # 샘플링 레이트 일치 확인 
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse # 캐시 참조 카운터 초기화
        # 캐시된 인덱스가 있으면 사용 
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1 # 참조 카운터 감소

        # 텐서로 변환 및 차원 추가 
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            # 오디오 분할 처리, 오디오 파일을 고정된 크기의 세그먼트로 나눔 
            if self.split:
                # 오디오 길이가 세그먼트 크기보다 크면, 
                if audio.size(1) >= self.segment_size:
                    # 오디오 시작점을 랜덤하게 선택 후, 선택된 시점부터 세그먼트 크기만큼 오디오를 잘라냄
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            # 멜 스펙트로그램 생성
            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        # 미세 조정 모드인 경우
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            # 스펙트로그램의 차원이 3보다 작으면, 차원 삽입
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            # 스플릿 옵션이 활성화된경우, 멜 스펙트로그램도 해당 세그먼트에 맞게 조정
            if self.split:
                # 세그먼트 크기에 해당하는 mel 프레임 수 계산
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                # 오디오 길이가 세그먼트 이상인 경우, 일부를 선택한 후
                if audio.size(1) >= self.segment_size:
                    # 랜덤 시작점 선택 
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    # 시작점으로부터 세그먼트만큼 멜 스펙트로그램을 잘라냄
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    # 마찬가지로 오디오의 세그먼트 선택
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    # 멜 스펙트로그램과 오디오의 길이가 세그먼트 크기보다 작은 경우, 뒤쪽을 0으로 패딩
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # mel loss 생성
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        # return : 멜 스펙트로그램, 오디오 세그먼트, 파일 이름, mel loss
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    # 오디오 파일의 length 출력
    def __len__(self):
        return len(self.audio_files)
