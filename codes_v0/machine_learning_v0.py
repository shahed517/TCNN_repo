import torch, random , librosa
import matplotlib.pyplot as plt 
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from scipy.stats import pearsonr
from fastdtw import fastdtw
import pysptk
import pyworld
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn.functional as F
from scipy.spatial.distance import euclidean

def save_loss_plot(train_loss_plot, val_loss_plot, filename="loss_plot.png", limit_y = True):
    epochs = range(1, len(train_loss_plot) + 1)
    min_val_loss = min(val_loss_plot)
    min_val_epoch = val_loss_plot.index(min_val_loss) + 1  # +1 since epochs start at 1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_plot, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss_plot, 'r-', label='Validation Loss')
    # Mark the lowest validation loss point
    plt.scatter(min_val_epoch, min_val_loss, color='red', s=100, zorder=5, 
               label=f'Lowest Val Loss: {min_val_loss:.2f} (Epoch {min_val_epoch})')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    y_min, y_max = min(min(train_loss_plot), min(val_loss_plot)), max(max(train_loss_plot), max(val_loss_plot))
    if limit_y:
        plt.ylim([0, 6])
        plt.yticks(np.arange(0, 6, 0.33))  # Adjust step size (0.5) as needed
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curve(s) saved as {filename}")

# Define the custom dataset class
class sEEG_Dataset(Dataset): 
    def __init__(self, X, y, sub_id, va_labels, augment = True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.va_labels = torch.tensor(va_labels, dtype=torch.float32) 
        self.augment = augment 
        self.sub_id = torch.tensor(sub_id).int()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        sEEG = self.X[idx] 
        mel = self.y[idx] 
        va_labels = self.va_labels.long() 
        if self.augment:
            if random.random() < 0.5:
                noise = (0.316)*torch.randn(size = sEEG.shape, dtype=torch.float32) # variance of 0.1
                sEEG = sEEG + noise 
        return sEEG, mel, va_labels 
    
class VocalMind_model(nn.Module):
    def __init__(self, in_channels, output_size):
        super(VocalMind_model, self).__init__()
        self.cnn_dim = 64
        self.rnn_dim = 256
        self.conv_block = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, 
                               out_channels= self.cnn_dim, 
                               kernel_size=4, 
                               stride=4, ## 4 in the original model to account for shape mismatch
                               padding=2), ## 2 in the original model
                nn.BatchNorm1d(self.cnn_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.fc = nn.Linear(self.rnn_dim*2, output_size)
     
        self.rnn = nn.GRU(input_size=self.cnn_dim, 
                          hidden_size=self.rnn_dim, 
                          num_layers=3, 
                          bidirectional= True,
                          batch_first=True,
                          dropout=0.7
                          )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, channels, timesteps)
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = x[:, :300, :] ## have to do this to address size mismatch error
        x, _ = self.rnn(x) 
        x = self.fc(x) # take only the final layer 
        return x
    
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
    

class TemporalCNN_deep(nn.Module):
    def __init__(self, input_channels=127, output_size=80, kernel_size = [7,7,7], dilations=[5,7,9], first_layer_ch=64):
        super(TemporalCNN_deep, self).__init__()
        # Default convolutional parameters if none are provided
        conv_params = [
            (first_layer_ch, kernel_size[0], dilations[0], 0.5),  # (out_channels, kernel_size, dilation, dropout_rate)
            (first_layer_ch*2, kernel_size[1], dilations[1], 0.5),
            (first_layer_ch*4, kernel_size[2], dilations[2], 0.5)
            ## (512, 15, 7, 0.5)
        ]
        # Create convolutional blocks dynamically 
        # self.total_channels = sum(out_channels for out_channels, _, _, _ in conv_params)
        self.total_channels = conv_params[-1][0]
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        for out_channels, kernel_size, dilation, dropout_rate in conv_params:
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding='same'  # Ensures output length = input length
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                # squeeze_and_excite(out_channels),
                nn.Dropout(dropout_rate)
            )
            self.conv_blocks.append(conv_block)
            in_channels = out_channels  # Update input channels for the next block
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.total_channels, self.total_channels//2),  # in_channels is the output of the last conv block
            # nn.BatchNorm1d(105), # number of time steps
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.total_channels//2, output_size)
        )
        self.first_dropout = nn.Dropout(0.1)
    def forward(self, x):
        # Input shape: (batch_size, timesteps, channels)
        x = x.permute(0, 2, 1)  # (batch_size, channels, timesteps)
        x = self.first_dropout(x)
        # Pass through all convolutional blocks
        outputs = []
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            # outputs.append(x)
        # x = torch.cat(outputs, dim = 1)
        x = x.permute(0, 2, 1)
        return self.fc(x)
    

def calculate_receptive_field(conv_params):
    receptive_field = 1  # Starting with a single element
    for layer in conv_params:
        _, kernel_size, dilation, _ = layer
        receptive_field += (kernel_size - 1) * dilation
    return receptive_field

def pearson_cc(pred, y):
    bs, mel_bins = pred.shape
    cost = torch.Tensor([0]).to(DEVICE)
    for i in range(mel_bins):
        x = pred[:, i] # take the i-th mel-bin
        vx = x - torch.mean(x)
        vy = y[:, i] - torch.mean(y[:, i])
        eps = torch.Tensor([1e-7]).to(DEVICE)
        cost = cost + torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return cost/mel_bins 

def bin_weighted_MSE(pred, labels):
    bs, time_dim, bins = pred.shape
    mel_bin_cutoff = 30 ## focus more the first 30 bins!
    weight_factor = 5
    squared_diff = (pred - labels) ** 2
    weights = torch.ones_like(squared_diff)
    weights[:, :, :mel_bin_cutoff] *= weight_factor
    weighted_mse = weights * squared_diff
    loss = torch.mean(weighted_mse)
    return loss 


# Define training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    model.train()
    train_loss_plot = []; val_loss_plot = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss1, running_loss2 = 0, 0
        batch_idx = 0
        for X_batch, y_batch, va_label in train_loader: 
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss1 = criterion(outputs, y_batch) 
            # loss2 = fft_loss(outputs, y_batch) 
            loss = loss1 #+ loss2 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_idx += 1

        model.eval()
        running_val_loss = 0; batch_idx_val = 0
        with torch.no_grad():
            for X_batch, y_batch, va_label in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)  ## pick specific model parameters!!
                running_val_loss += criterion(outputs, y_batch)
                batch_idx_val += 1

        a = running_loss/batch_idx
        b = running_val_loss/batch_idx_val

        if a>6 or b>6:
            train_loss_plot.append(6)
            val_loss_plot.append(6)
        else:
            train_loss_plot.append(running_loss/batch_idx)
            val_loss_plot.append(running_val_loss.to('cpu')/batch_idx_val)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/batch_idx:.4f}, Validation Loss : {running_val_loss/batch_idx_val:.4f} ")
        model.train()
    return train_loss_plot, val_loss_plot

# Define inference function
def evaluate_model(model, test_loader, PATH, speech_only_testing=False, drop_edges = False):
    model.load_state_dict(torch.load(PATH, weights_only=True)) # not really necessary!
    model.eval()
    rec_spec = []; gt_spec = []
    running_mse = 0
    with torch.no_grad():
        for X_batch, y_batch, va_labels in test_loader: 
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE) 
            # print(f'model input shape : {X_batch.shape}') 
            outputs = model(X_batch) ## take the first element in batch
            outputs = outputs[0]; y_batch = y_batch[0] ## consider only the first element in the batch (T sec) 
            if drop_edges: ## somehow doing this is bad?!
                outputs = outputs[64:, :]; y_batch = y_batch[64:, :]; va_label = va_label[64:, :] # do it for the va_labels also!
                outputs = outputs[:-64, :]; y_batch = y_batch[:-64, :]; va_label = va_label[:-64, :] ## drop the elements for which the RF is missing
            indices = np.where(va_labels[0] == 1)[0] ## a bit unsure about this :/
            if speech_only_testing:
                rec_spec.append(outputs[indices, :].cpu().numpy())  ## remove the batch dimension
                gt_spec.append(y_batch[indices, :].cpu().numpy())
            else:
                rec_spec.append(outputs.cpu().numpy())  ## remove the batch dimension
                gt_spec.append(y_batch.cpu().numpy())
            running_mse += F.mse_loss(outputs, y_batch)
    stacked_predictions = np.vstack(rec_spec) ## for 2D arrays, stacks along axis=0 (row)
    stacked_gt = np.vstack(gt_spec)
    mse_score = running_mse.item()/len(test_loader)
    R = []
    for specBin in range(stacked_predictions.shape[1]):
        r, p = pearsonr(stacked_gt[:, specBin], stacked_predictions[:, specBin]) # default axis = 0, across the column
        R.append(r) # make of list of all spectral coefficients
    pcc_score = np.array(R)
    return mse_score, pcc_score, [stacked_gt, stacked_predictions]

def evaluate_model_twiceT(model, test_loader, PATH, speech_only_testing=False):
    model.load_state_dict(torch.load(PATH, weights_only=True)) # not really necessary!
    model.eval()
    rec_spec = []; va_labels = []; gt_spec = []
    running_mse = 0
    with torch.no_grad():
        for X_batch, y_batch, va_labels in test_loader: 
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE) 
            batch_size = X_batch.shape[0]
            X_large = torch.cat([X_batch[0:1, :, :], X_batch[batch_size//2:batch_size//2 + 1, :, :]], dim = 1) ## outputs will have shape : (600, 80)
            outputs = model(X_large)[0] # drop batch dimension
            # print(f'X-large shape : {X_large.shape}')
            # print(f'y-pred shape : {outputs.shape}')
            gt = torch.cat([y_batch[0], y_batch[batch_size//2]], dim = 0) 
            selected_va_labels = torch.cat((va_labels[0], va_labels[batch_size//2]), dim=0)
            indices = np.where(selected_va_labels == 1)
            if speech_only_testing:
                rec_spec.append(outputs[indices, :].cpu().numpy())  ## remove the batch dimension
                gt_spec.append(gt[indices, :].cpu().numpy())
            else:
                rec_spec.append(outputs.cpu().numpy())  ## remove the batch dimension
                gt_spec.append(gt.cpu().numpy())
            running_mse += F.mse_loss(outputs, gt)
    stacked_predictions = np.vstack(rec_spec) ## for 2D arrays, stacks along axis=0 (row)
    stacked_gt = np.vstack(gt_spec)
    mse_score = running_mse.item()/len(test_loader)
    R = []
    for specBin in range(stacked_predictions.shape[1]):
        r, p = pearsonr(stacked_gt[:, specBin], stacked_predictions[:, specBin]) # default axis = 0, across the column
        R.append(r) # make of list of all spectral coefficients
    pcc_score = np.array(R)
    return mse_score, pcc_score, [stacked_gt, stacked_predictions]


if __name__ == "__main__":
    np.random.seed(42)
    # Initialize model
    model = TemporalCNN_deep(127, 80)
    # model = VocalMind_model(127, 80)
    model = model.to(DEVICE) 

    model.eval()
    random_X = torch.rand(size = (8, 420, 127)).to(DEVICE)
    random_Y = model(random_X)
    print(f'Input shape: {random_X.shape}')
    print(f'output shape: {random_Y.shape}')
    # print(f'output-2 shape : {random_Z[0].shape}')

