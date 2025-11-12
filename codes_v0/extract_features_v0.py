import os, librosa, argparse, torch, sys 
import pandas as pd
import numpy as np
import scipy.io.wavfile
import scipy.fftpack
from scipy.fft import fft, fftfreq
from pynwb import NWBHDF5IO
# import MelFilterBank as mel
from scipy.io import wavfile
from scipy.signal import lfilter
#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

sys.path.append(os.path.dirname('/home/ahmed348/SingleWordProductionDutch/meldataset.py'))
from meldataset import mel_spectrogram ## function from HIFIGAN

def get_HGA_LFC(data, sr, lfc_cutoff=30, lfc_only=False, hga_only=False, sr_new = 100):
    #Linear detrend
    orig_data = scipy.signal.detrend(data, axis=0)
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,orig_data,axis=0)
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    # Attenuate second harmonic of line noise 
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    data = np.abs(hilbert3(data)) # this is the envelop amplitude
    # lowpass filter original signal to get the low-frequency signals (supposedly they are important for synthesis)
    sos = scipy.signal.iirfilter(6, (lfc_cutoff-1)/(sr/2), btype='lowpass', output='sos')
    data_extra = scipy.signal.sosfiltfilt(sos, orig_data, axis = 0)
    if lfc_only:
        concatenated_data = data_extra
    elif hga_only:
        concatenated_data = data 
    else:
        concatenated_data = np.concatenate([data, data_extra], axis = -1)
    sr_new = sr_new # to make it the same as the mel spectrogram 
    total_channels = concatenated_data.shape[1]
    new_data_length = int(np.floor(concatenated_data.shape[0]*sr_new/sr))
    eeg_feat = np.zeros((new_data_length, concatenated_data.shape[1]))
    for i in range(total_channels): # no. of channels
        tmp = scipy.signal.resample(concatenated_data[:, i], new_data_length) 
        eeg_feat[:, i] = tmp
    return eeg_feat


def make_eeg_spec_pairs(audio, eeg, T, audio_sr, eeg_sr=100, n_mels=80, stride=0.05, threshold=None): ## HGA-LFC has been computed already!
    n_channels = eeg.shape[1]
    eeg_window = int(T * eeg_sr)
    eeg_stride = int(stride * eeg_sr)
    audio_window = int(T * audio_sr)
    audio_stride = int(stride * audio_sr)
    
    # STFT parameters (50ms window, 10ms hop)
    hop_seconds = 0.01
    hop_length = int(audio_sr * hop_seconds)
    win_length = int(audio_sr * 0.05)
    # n_fft = 2 ** int(np.ceil(np.log2(win_length)))
    
    eeg_segments = []; va_labels = []; mel_segments = []
    total_eeg_samples = len(eeg)
    total_audio_samples = len(audio)
    max_segments = min(
        (total_eeg_samples - eeg_window) // eeg_stride + 1,
        (total_audio_samples - audio_window) // audio_stride + 1
    )
    for i in range(max_segments):
        eeg_start = i * eeg_stride
        audio_start = i * audio_stride
        eeg_seg = eeg[eeg_start : eeg_start + eeg_window, :]
        aud_seg = audio[audio_start : audio_start + audio_window]

        # === Compute RMS energy per frame ===
        rms = librosa.feature.rms(y=aud_seg, frame_length=win_length, hop_length=hop_length)[0]

        # === Create speech/silence mask ===
        threshold = np.mean(rms) * 0.1
        va_label = (rms > threshold).astype(int)  # 1 = speech, 0 = silence

        ### previous method to calculate log-mel spectrogram
        # scaled = np.int16(((aud_seg-np.min(aud_seg))/(np.max(aud_seg)-np.min(aud_seg))) * 32767)  
        #Extract spectrogram 

        MAX_WAV_VALUE = np.max(np.abs(aud_seg))
        scaled = aud_seg/MAX_WAV_VALUE # normalize for hifiGAN
        scaled = torch.from_numpy(scaled).unsqueeze(0)

        n_mel_channels = n_mels  # Default for HiFi-GAN
        sampling_rate = audio_sr  # Default 22000 for HiFi-GAN
        # hop_length = 220  # Default for HiFi-GAN to make a 0.01 sec hop
        # win_length = 1100  # Default for HiFi-GAN to make a 0.05 sec long window
        n_fft = win_length  # Default for HiFi-GAN

        # COMPUTE MEL SPECTROGRAM
        S = mel_spectrogram(
            scaled,
            n_fft=n_fft,
            num_mels=n_mel_channels,
            sampling_rate=sampling_rate,
            hop_size=hop_length,
            win_size=win_length,
            fmin=0,
            fmax=8000,
            center = False
        )
        S = S.numpy()
        # print(S.shape) 

        # # Compute mel spectrogram
        # S = librosa.feature.melspectrogram(
        #     y=aud_seg,
        #     sr=audio_sr,
        #     n_fft=n_fft,
        #     win_length=win_length,
        #     hop_length=hop_length,
        #     n_mels=n_mels,
        #     power=2.0,
        # )

        expected_frames = int(T * 100) ## this produces 300 exactly!
        # pad/trim to match EEG length exactly
        if S.shape[1] < expected_frames:
            pad = expected_frames - S.shape[1]
            S = np.pad(S, ((0,0),(0,pad)), mode='constant')
        elif S.shape[1] > expected_frames:
            S = S[:, :expected_frames]

        if len(va_label) < expected_frames:
            pad = expected_frames - S.shape[1]
            va_label = np.pad(va_label, (0, pad), mode='constant')
        elif len(va_label) > expected_frames:
            va_label = va_label[:expected_frames]

        eeg_segments.append(eeg_seg)
        mel_segments.append(S.T)
        va_labels.append(va_label)
        
    eeg_segments = np.stack(eeg_segments)
    mel_segments = np.stack(mel_segments)
    va_labels = np.stack(va_labels)
    print(f'eeg, mel, va_label shapes : {eeg_segments.shape}, {mel_segments.shape}, {va_labels.shape}')
    
    return eeg_segments, mel_segments, va_labels


if __name__=="__main__":
    path_bids = '/scratch/gilbreth/ahmed348/public_datasets/Dutch_Dataset'
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=3, help="duration of eeg/spec segments")
    parser.add_argument("--stride", type=float, default=0.05, help="stride between consecutive segments")
    parser.add_argument("--lfc_cutoff", type=int, default=30, help="cutoff frequency for LFC")
    parser.add_argument("--lfc_only", type=lambda x: x.lower() == "true", default=False, help="Set to true or false")
    parser.add_argument("--hga_only", type=lambda x: x.lower() == "true", default=False, help="Set to true or false")
    parser.add_argument("--root_keyword", type=str, default="HGA_LFC", help="specify experiment")
    parser.add_argument("--audio_sr", type=int, default=22000, help="sampling rate required for vocoder")
    parser.add_argument("--n_mels", type=int, default=80, help="no. of mel bins")
    args = parser.parse_args()

    ROOT_KEYWORD = args.root_keyword # LFC_only/HGA_only/HGA_LFC, etc.
    if args.lfc_cutoff == 30 and args.stride == 0.05:
        KEYWORD = f'{args.root_keyword}_T_{args.T}' ## for pooling the appropriate data, DEFAULT ONE!
    elif args.lfc_cutoff != 30:
        KEYWORD =  f'{ROOT_KEYWORD}_T_{args.T}_fc_{args.lfc_cutoff}' ## for experiments with lfc_cutoff
    elif args.stride != 0.05:
        KEYWORD = f'{args.root_keyword}_T_{args.T}_stride_{args.stride}'

    path_output = f'/scratch/gilbreth/ahmed348/Dutch_dataset_features/TCNN_folder/FEATURES/features_{KEYWORD}'

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    file_list = [f"/scratch/gilbreth/ahmed348/public_datasets/Dutch_Dataset/sub-{i:02d}/ieeg/sub-{i:02d}_task-wordProduction_events.tsv" 
                for i in range(1, 11)]
    
    if args.root_keyword.lower() == "vocalmind":
        eeg_sr_new = 400
    else: 
        eeg_sr_new = 100 # default
    participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')
    for p_id, participant in enumerate(participants['participant_id']): 
        io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        eeg_hga_lfc = get_HGA_LFC(eeg, eeg_sr, lfc_cutoff=args.lfc_cutoff, lfc_only=args.lfc_only, hga_only=args.hga_only, sr_new=eeg_sr_new) ## this gets the HGA_LFC features
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000; hifiGAN_sr = args.audio_sr
        audio = scipy.signal.resample(audio, int(len(audio)*hifiGAN_sr/audio_sr))
        eeg_segments, mel_segments, va_labels = make_eeg_spec_pairs(audio, eeg_hga_lfc, T=args.T, n_mels=args.n_mels, 
                                                                    stride=args.stride, audio_sr=hifiGAN_sr, eeg_sr = eeg_sr_new)
        print(eeg_segments.shape, mel_segments.shape)

        np.save(os.path.join(path_output,f'{participant}_feat.npy'), eeg_segments)
        np.save(os.path.join(path_output,f'{participant}_spec.npy'), mel_segments)
        np.save(os.path.join(path_output,f'{participant}_va.npy'), va_labels)


    
