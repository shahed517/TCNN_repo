import os, random, sys, pickle 
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import torch, os, json, scipy 
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from machine_learning import sEEG_Dataset, train_model, evaluate_model, save_loss_plot
from machine_learning import TemporalCNN_deep
from machine_learning import VocalMind_model, compute_mcd
from tcnn_utils import get_fold_i, sEEG_EvalDataset
from pystoi import stoi
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import librosa, pysptk, pyworld
from sklearn.decomposition import PCA

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


sys.path.append(os.path.dirname('/home/ahmed348/SingleWordProductionDutch/hifigan/models.py'))
sys.path.append(os.path.dirname('/home/ahmed348/SingleWordProductionDutch/hifigan/env.py'))

from models import Generator
from env import AttrDict

def generate_audio_hifiGAN(mel_spec, output_dir, sampling_rate, filename, device):
    config_file = '/home/ahmed348/SingleWordProductionDutch/hifigan/pretrained/UNIVERSAL_V1/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint('/home/ahmed348/SingleWordProductionDutch/hifigan/pretrained/UNIVERSAL_V1/g_02500000', device)
    generator.load_state_dict(state_dict_g['generator'])
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    generator.remove_weight_norm()
    # Generate audio
    # seconds_to_generate = 35
    # mel_spec = mel_spec[:int(seconds_to_generate/0.01)] ## limit Hifi-GAN to producing smaller segments of speech
    mel_spec = torch.from_numpy(mel_spec).unsqueeze(0).float()
    print(f'input shape : {mel_spec.shape}')
    with torch.no_grad():
        mel_spec = mel_spec.permute(0, 2, 1).to(device)
        generated_audio = generator(mel_spec)
    # # Output shape from the generator: (1, 1, T' * hop_length)
    print(f"Generated audio shape: {generated_audio.shape}")
    generated_audio *= 32767
    generated_audio = generated_audio.squeeze().cpu().numpy().astype('int16')
    output_path = os.path.join(output_dir, filename)
    scipy.io.wavfile.write(output_path, sampling_rate, generated_audio)

def mcd_calc(C, C_hat):
    """ Computes MCD between ground truth and target MFCCs. First computes DTW aligned MFCCs

    Consistent with Anumanchipalli et al. 2019 Nature, we use MC 0 < d < 25 with k = 10 / log10
    """
    # ignore first MFCC
    K = 10 / np.log(10)
    C = C[:, 1:25]
    C_hat = C_hat[:, 1:25]
    # compute alignment
    distance, path = fastdtw(C, C_hat, dist=euclidean)
    distance/= (len(C) + len(C_hat))
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    C, C_hat = C[pathx], C_hat[pathy]
    frames = C_hat.shape[0]
    # compute MCD
    z = C_hat - C
    s = np.sqrt((z * z).sum(-1)).sum()
    MCD_value = K * float(s) / float(frames)
    return MCD_value

def wav2mcep_numpy(wav, sr, alpha=0.65, fft_size=512, mcep_size=25):
    """ Given a waveform, extract the MCEP features """
    _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=sr,frame_period=5.0, fft_size=fft_size)
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return mgc

def normalize_volume(audio):
    """ Normalize an audio waveform to be between 0 and 1 """
    rms = librosa.feature.rms(y=audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    return audio

def compute_mcd(sample, y, sr_desired=16000):
    """ Computes MCD between target waveform and predicted waveform """
    # equalize lengths
    if len(sample) < len(y):
        y = y[:len(sample)]
    else:
        sample = sample[:len(y)]
    # normalize volume
    y = normalize_volume(y)
    sample = normalize_volume(sample)
    # compute MCD
    mfcc_y_ = wav2mcep_numpy(sample, sr_desired)
    mfcc_y = wav2mcep_numpy(y, sr_desired)
    mcd = mcd_calc(mfcc_y, mfcc_y_)
    return mcd

def calc_mcd_stoi(y1, y2, sr):
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    mcd_val = compute_mcd(y1, y2, sr)
    stoi_val = stoi(y1, y2, sr, extended=False)
    return mcd_val, stoi_val

def calculate_speech_perception_metrics(path1, path2):
    y1, sr1 = librosa.load(path1, sr=None)
    y2, sr2 = librosa.load(path2, sr=None)
    mcd, stoi_val = calc_mcd_stoi(y1, y2, sr1)
    return mcd, stoi_val 

import matplotlib.pyplot as plt
import tikzplotlib, argparse

# run this for evaluation (note the kernel size)
# python /home/ahmed348/TCNN_repo/main.py --T 6 --root_keyword HGA_LFC --model tcnn --kernel_size 15,15,15 --just_evaluate True --speech_only_eval True

## WINNER -> 15,15,15 === 1, 3, 5 --> R.Field size = 1.27 sec
if __name__=="__main__":
    np.random.seed(42)
    random.seed(4)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=3, help="duration of eeg/spec segments")
    parser.add_argument("--testT", type=int, default=3, help="duration of test eeg/spec segments")
    parser.add_argument("--just_evaluate", type=lambda x: x.lower() == "true", default=False, help="Set to true or false")
    parser.add_argument("--lfc_cutoff", type=int, default=30, help="cutoff frequncy of LFC")
    parser.add_argument("--causality", type=int, default=0, help="causality of convolutions")
    
    parser.add_argument("--sub", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated subject indices")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated fold indices")
    parser.add_argument("--dilations", type=str, default="5,7,9", help="Comma-separated dilations params")
    parser.add_argument("--first_layer_ch", type=int, default=128, help="first conv. channel count")
    parser.add_argument("--kernel_size", type=str, default="7,7,7", help="kernel size of Conv. layers")
    parser.add_argument("--root_keyword", type=str, default="HGA_LFC", help="specify experiment")
    parser.add_argument("--speech_only_eval", type=lambda x: x.lower() == "true", default=False, help="Set to true or false")
    parser.add_argument("--silence_only_eval", type=lambda x: x.lower() == "true", default=False, help="Set to true or false")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate") 
    parser.add_argument("--total_iter", type=int, default=7500, help="total epochs")
    parser.add_argument("--iter_per_epoch", type=int, default=100, help="total epochs")
    parser.add_argument("--nfolds", type=int, default=10, help="Number of folds for KFold")
    parser.add_argument("--model", type=str, default="TCNN", help="specify model")
    parser.add_argument("--aug", type=lambda x: x.lower() == "true", default=True, help="Set to true or false")
    parser.add_argument("--use_pca", type=lambda x: x.lower() == "true", default=False, help="Set to true or false")

    args = parser.parse_args()

    sub_list = [int(x) for x in args.sub.split(",")]
    fold_list = [int(x) for x in args.folds.split(",")]
    DILATIONS =  [int(x) for x in args.dilations.split(",")]
    dil_str = "".join(map(str, DILATIONS)) + "_dil"
    KERNELS =  [int(x) for x in args.kernel_size.split(",")]
    kernel_str = "".join(map(str, KERNELS)) + "_ker"
    SEGMENTS_PER_EPOCH = args.batch_size * args.iter_per_epoch ## arbitrarily set

    ROOT_KEYWORD = args.root_keyword # LFC_only/HGA_only/HGA_LFC, etc.
    if args.lfc_cutoff == 30:
        KEYWORD = f'{args.root_keyword}' ## for pooling the appropriate data, DEFAULT ONE!
    else:
        KEYWORD =  f'{ROOT_KEYWORD}_fc_{args.lfc_cutoff}' ## for experiments with lfc_cutoff

    WEIGHTS_KEYWORD = f'{KEYWORD}_{dil_str}_{kernel_str}' ## for the dilation/kernel_size experiments.

    if args.first_layer_ch != 128:
        WEIGHTS_KEYWORD = f'{WEIGHTS_KEYWORD}_{args.first_layer_ch}_ch' ## for the conv channel experiments.

    if args.model.lower() == 'vocalmind':
        WEIGHTS_KEYWORD = f'{KEYWORD}_{args.model}'

    WEIGHTS_KEYWORD = f'{WEIGHTS_KEYWORD}_{args.total_iter}_T_{args.T}'

    feat_path = f'/scratch/gilbreth/ahmed348/Dutch_dataset_features/TCNN_folder/FEATURES/features_{KEYWORD}'

    if args.causality>0:
        WEIGHTS_KEYWORD = f'{WEIGHTS_KEYWORD}_causal'
    elif args.causality<0:
        WEIGHTS_KEYWORD = f'{WEIGHTS_KEYWORD}_anti_causal'

    if args.use_pca:
        WEIGHTS_KEYWORD = f'{WEIGHTS_KEYWORD}_pca'
    
    if args.speech_only_eval:
        result_path = f'/scratch/gilbreth/ahmed348/Dutch_dataset_features/TCNN_folder/RESULTS/results_{WEIGHTS_KEYWORD}_speech_only_eval'
    else:
        result_path = f'/scratch/gilbreth/ahmed348/Dutch_dataset_features/TCNN_folder/RESULTS/results_{WEIGHTS_KEYWORD}' ## this is where the final TCNN results will be saved

    # feat_path = '/scratch/gilbreth/ahmed348/Dutch_dataset_features/features_vocalmind'
    # result_path = '/scratch/gilbreth/ahmed348/Dutch_dataset_features/vocalmind_results'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = ['sub-%02d'%i for i in range(1,11)]

    JUST_EVALUATE = args.just_evaluate # False by default   

    winLength = 0.05
    frameshift = 0.01
    audiosr = 22000 ## these three are fixed due to hifi-GAN

    nfolds = args.nfolds
    # kf = KFold(nfolds,shuffle=False)

    FINAL_SCORES = []; FINAL_MEAN_SCORES = []
    FINAL_MSE = []
    n_mels = 80
    BATCH_SIZE = args.batch_size

    SPECTROGRAMS = []
    INPUT_CHANNEL_LIST = []
    EEG = [] 
    COORDINATES = []
    WORD_LABELS = []
    VA_LABELS = [] 
    MCD_SCORES = []
    PCC_all = []; MCD_all = []; STOI_all = []
    PCC_SEQ_all = []; VAs_all = []

    for pNr, pt in enumerate(pts):
        if pNr not in sub_list: # just on the first one/two patients for now 
            continue 
        print(f'pnr : {pNr}')
        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        print(f'spectrogram shape: {spectrogram.shape}')
        eeg = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
        va_labels = np.load(os.path.join(feat_path,f'{pt}_va.npy'))
        
        #Initialize an empty spectrogram to save the reconstruction to
        # Save the correlation coefficients for each fold
        fold_wise_pcc = []
        fold_wise_mse = []
        MCDs = []; STOIs = []; PCC_SEQs = []; VAs = []

        for k in range(args.nfolds):
            if k not in fold_list: # just on the first one/two patients for now 
                continue
            print(f'patient number : {pNr+1}, fold number : {k}') 
            X_train, y_train, va_labels_train, X_test, y_test, \
                va_labels_test = get_fold_i(eeg, spectrogram, va_labels, args.nfolds, k)

            channel_means_train = np.mean(X_train, axis=0)  # Shape: (127,)
            channel_stds_train = np.std(X_train, axis=0)    # Shape: (127,)
            channel_means_test = np.mean(X_test, axis=0)  # Shape: (127,)
            channel_stds_test = np.std(X_test, axis=0)    # Shape: (127,)

            X_train = (X_train - channel_means_train)/channel_stds_train
            X_test = (X_test - channel_means_test)/channel_stds_test

            if args.use_pca:
                # Fit PCA without fixing k first
                pca_full = PCA()
                pca_full.fit(X_train)   # shape: (N_samples, C)
                # Cumulative explained variance
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                # Smallest k that reaches 90%
                pca_k = np.searchsorted(cumvar, 0.90) + 1
                print(f"Chosen k = {pca_k}, explained variance = {cumvar[pca_k-1]:.4f}")

                pca = PCA(n_components=pca_k, svd_solver="full")
                X_train_pca = pca.fit_transform(X_train)   # (N, pca_k)
                X_test_pca  = pca.transform(X_test)        # (N, pca_k)
                X_train = X_train_pca
                X_test = X_test_pca 
                print("Explained variance:", pca.explained_variance_ratio_.sum())

            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train)
            X_test = torch.from_numpy(X_test)
            y_test = torch.from_numpy(y_test)

            print(f'test data eeg shape : {X_test.shape}')
            print(f'test data mel shape : {y_test.shape}')

            train_dataset = sEEG_Dataset(X_train, y_train, va_labels_train, args.T, n_segments_per_epoch=SEGMENTS_PER_EPOCH, augment=args.aug) ## true for TemporalCNN
            test_dataset = sEEG_EvalDataset(X_test, y_test, va_labels_test, args.T) ## won't augment this ever!
            train_eval_dataset = sEEG_EvalDataset(X_train, y_train, va_labels_train, args.T) ## won't augment this ever!
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            train_eval_loader = DataLoader(train_eval_dataset, batch_size=1, shuffle=False) 
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            input_channels = X_train.shape[-1]   
            feature_length = y_train.shape[-2]   
            output_dim = n_mels

            if args.model.lower() == "tcnn": ## default
                model = TemporalCNN_deep(input_channels, output_dim, KERNELS, DILATIONS, args.first_layer_ch, args.causality).to(DEVICE) ## for HGA-only and HGA-LFC with fc<51
                print('chosing the tcnn model!')
            elif args.model.lower() == "vocalmind":
                model = VocalMind_model(input_channels, output_dim, args.T).to(DEVICE)
                print('chosing the vocalmind model!')
            else:
                print('specify model properly!')
                exit()
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")

            criterion = nn.MSELoss() 
            optimizer = optim.Adam(model.parameters(), lr=args.lr)    

            # Train the model 
            num_epochs = args.total_iter//args.iter_per_epoch
            # MODEL_SAVE_PATH_ROOT = f'/home/ahmed348/TCNN_repo/TCNN_weights/weights_{WEIGHTS_KEYWORD}'
            MODEL_SAVE_PATH_ROOT = f'/scratch/gilbreth/ahmed348/Dutch_dataset_features/TCNN_folder/SAVED_WEIGHTS/weights_{WEIGHTS_KEYWORD}'
            if not os.path.exists(MODEL_SAVE_PATH_ROOT):
                os.makedirs(MODEL_SAVE_PATH_ROOT) 
            L_CURVE_SAVE_PATH_ROOT = f'/home/ahmed348/TCNN_repo/TCNN_L_curves/L_curves_{WEIGHTS_KEYWORD}'
            if not os.path.exists(L_CURVE_SAVE_PATH_ROOT):
                os.makedirs(L_CURVE_SAVE_PATH_ROOT)
            MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH_ROOT, f'weights_{pt}_fold_{k}.pth') 
            L_CURVE_SAVE_PATH = os.path.join(L_CURVE_SAVE_PATH_ROOT, f'{pt}_fold_{k}.jpg') 

            if not JUST_EVALUATE:
                train_loss_plot, val_loss_plot = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs) 
                save_loss_plot(train_loss_plot, val_loss_plot, L_CURVE_SAVE_PATH)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                _, pcc_train, _ = evaluate_model(model, train_loader, MODEL_SAVE_PATH) 
                print('%s, fold %s has train correlation of %f' % (pt, k, np.mean(pcc_train)))
            
            # Predict the reconstructed spectrogram for the test data 
            mse_test, pcc_test, [gt_spec, pred_spec] = evaluate_model(model, test_loader, MODEL_SAVE_PATH, speech_only_testing=args.speech_only_eval, silence_only_testing=args.silence_only_eval)

            if JUST_EVALUATE:
                generate_audio_hifiGAN(pred_spec, result_path, audiosr, f'{pt}_fold_{k}_pred.wav', DEVICE)
                generate_audio_hifiGAN(gt_spec, result_path, audiosr, f'{pt}_fold_{k}_gt.wav', DEVICE)
                pred_wav_path = os.path.join(result_path, f'{pt}_fold_{k}_pred.wav')
                gt_wav_path = os.path.join(result_path, f'{pt}_fold_{k}_gt.wav')
                mcd, stoi_score = calculate_speech_perception_metrics(gt_wav_path, pred_wav_path) 
                MCDs.append(mcd)
                STOIs.append(stoi_score)

            # rec_spec[test, :] = pred_spec
            print(f'predicted spectrogram shape: {pred_spec.shape}')
            fold_wise_pcc.append(pcc_test)
            fold_wise_mse.append(np.mean(mse_test))
             
            # Show evaluation result
            print(f'fold : {k}:\n')
            print('%s, fold %s has test correlation of %f' % (pt, k, pcc_test))
            print('%s, fold %s has test mse of %f' % (pt, k, mse_test))

        print('%s has test correlation of %f' % (pt, np.mean(np.array(fold_wise_pcc))))
        FINAL_MEAN_SCORES.append(np.mean(np.array(fold_wise_pcc)))
        FINAL_SCORES.append(np.array(fold_wise_pcc))
        FINAL_MSE.append(np.mean(np.array(fold_wise_mse)))

        if JUST_EVALUATE:
            PCC_all.append(fold_wise_pcc)
            MCD_all.append(MCDs)
            STOI_all.append(STOIs)

    print(f'final PCC scores for all folds: {FINAL_SCORES}')
    print(f'Avg PCC scores for each subject: {FINAL_MEAN_SCORES}')
    print(f'final averaged MSE score of all {nfolds} folds : {FINAL_MSE}')

    if JUST_EVALUATE:
        results = {
        "MCD": MCD_all,
        "STOI": STOI_all,
        "PCC": PCC_all,
        }
        # Save to file
        if args.speech_only_eval:
            pkl_file = f"/home/ahmed348/TCNN_repo/pkl_files/TCNN_{WEIGHTS_KEYWORD}_speech_only.pkl"
        elif args.silence_only_eval:
            pkl_file = f"/home/ahmed348/TCNN_repo/pkl_files/TCNN_{WEIGHTS_KEYWORD}_sil_only.pkl"
        else:
            pkl_file = f"/home/ahmed348/TCNN_repo/pkl_files/TCNN_{WEIGHTS_KEYWORD}.pkl"

        with open(pkl_file, "wb") as f:
            pickle.dump(results, f)