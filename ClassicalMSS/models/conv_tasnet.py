import torch
import torch.nn as nn

class Conv_Tasnet(nn.Module):
    def __init__(self, N=8, small_size=False, dropout_rate=0.1):
        super(Conv_Tasnet, self).__init__()
        self.N = N
        if small_size:
            self.name = f"Conv_Tasnet_{N}_dropout{dropout_rate}_small_size"
        else:
            self.name = f"Conv_Tasnet_{N}_dropout{dropout_rate}"

        # Add dropout_rate as a parameter
        self.dropout_rate = dropout_rate

        # Encoder (single layer)
        self.encoder = nn.Conv1d(2, 256, kernel_size=20, stride=10, padding=5)
        # Now the result is 256 channels, each channel 4800Hz

        # Separation Network
        self.separation = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        for i in range(4 * N):
            dilation = 2 ** (i % N)
            self.separation.append(nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=1),# Stride is 1 by default
                nn.PReLU(),
                nn.GroupNorm(1, 256), # The scale and shift are learned parameters by default, if don't want these, add parameter affine=False
                nn.Conv1d(256, 256, kernel_size=3, padding=dilation, dilation=dilation, groups=256),
                nn.Conv1d(256, 256, kernel_size=1),
                nn.PReLU(),
                nn.GroupNorm(1, 256),
                nn.Dropout(self.dropout_rate)  # Add dropout layer
            ))
            # The 1*1 conv used after every block, before participates to the final mask estimation
            self.skip_connections.append(nn.Conv1d(256, 256, kernel_size=1))

        # Mask estimation
        # That's because I want two stems (piano and violin), each 256 channels
        # So before multiplying the result of the encoder by the mask (256 channels)
        # I need two set of masks, one for the violin and one for the piano, 256*2 = 512
        self.mask_conv = nn.Conv1d(256, 512, kernel_size=1)

        # Add a dropout layer after mask estimation
        self.mask_dropout = nn.Dropout(self.dropout_rate)

        # Decoder (single layer, now outputting 2 channels)
        self.decoder = nn.ConvTranspose1d(256, 2, kernel_size=20, stride=10, padding=5)
        # First pad 0's, 4800->48010, then perform normal conv1d with kernel size 20 and stride 1,
        # If we don't pad, then the result will be 47990, padding=5 can make it 48000

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Separation
        skip_sum = 0
        for sep_layer, skip_layer in zip(self.separation, self.skip_connections):
            residual = x
            x = sep_layer(x)
            x += residual
            skip_sum += skip_layer(x) # The mask is obtained summing the output of all blocks (and then applying ReLU)

        # Mask estimation
        masks = self.mask_conv(nn.ReLU()(skip_sum)) # Now 512 channels
        masks = nn.ReLU()(masks) # All value in the mask are non-negative
        # May also try sigmoid, etc.

        # Split masks for two sources
        mask1, mask2 = torch.chunk(masks, 2, dim=1)  # Split a tensor into 2 tensors along dimension 1 (channels)
        # Each mask is now 256 channels

        # Apply masks
        masked_features1 = x * mask1
        masked_features2 = x * mask2

        # Decoding
        violin = self.decoder(masked_features1)
        piano = self.decoder(masked_features2)
        
        # violin = torch.tanh(violin)  # Normalize to (-1, 1)
        # piano = torch.tanh(piano)  # Normalize to (-1, 1)

        # Combine violin and piano outputs into a single tensor
        combined_output = torch.cat([violin, piano], dim=1)

        return combined_output  # Return a single tensor with shape [batch, 4, time]



class Conv_Tasnet_Tanh(nn.Module):
    def __init__(self, N=8, small_size=False):
        super(Conv_Tasnet_Tanh, self).__init__()
        self.N = N
        if small_size:
            self.name = f"Conv_Tasnet_{N}_small_size"
        else:
            self.name = f"Conv_Tasnet_{N}"

        # Encoder (single layer)
        self.encoder = nn.Conv1d(2, 256, kernel_size=20, stride=10, padding=5)
        # Now the result is 256 channels, each channel 4800Hz

        # Separation Network
        self.separation = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        for i in range(4 * N):
            dilation = 2 ** (i % N)
            self.separation.append(nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=1),# Stride is 1 by default
                nn.PReLU(),
                nn.GroupNorm(1, 256), # The scale and shift are learned parameters by default, if don't want these, add parameter affine=False
                nn.Conv1d(256, 256, kernel_size=3, padding=dilation, dilation=dilation, groups=256),
                nn.Conv1d(256, 256, kernel_size=1),
                nn.PReLU(),
                nn.GroupNorm(1, 256)
            ))
            # The 1*1 conv used after every block, before participates to the final mask estimation
            self.skip_connections.append(nn.Conv1d(256, 256, kernel_size=1))

        # Mask estimation
        # That's because I want two stems (piano and violin), each 256 channels
        # So before multiplying the result of the encoder by the mask (256 channels)
        # I need two set of masks, one for the violin and one for the piano, 256*2 = 512
        self.mask_conv = nn.Conv1d(256, 512, kernel_size=1)

        # Decoder (single layer, now outputting 2 channels)
        self.decoder = nn.ConvTranspose1d(256, 2, kernel_size=20, stride=10, padding=5)
        # First pad 0's, 4800->48010, then perform normal conv1d with kernel size 20 and stride 1,
        # If we don't pad, then the result will be 47990, padding=5 can make it 48000

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Separation
        skip_sum = 0
        for sep_layer, skip_layer in zip(self.separation, self.skip_connections):
            residual = x
            x = sep_layer(x)
            x += residual
            skip_sum += skip_layer(x) # The mask is obtained summing the output of all blocks (and then applying ReLU)

        # Mask estimation
        masks = self.mask_conv(nn.ReLU()(skip_sum)) # Now 512 channels
        masks = nn.ReLU()(masks) # All value in the mask are non-negative
        # May also try sigmoid, etc.

        # Split masks for two sources
        mask1, mask2 = torch.chunk(masks, 2, dim=1)  # Split a tensor into 2 tensors along dimension 1 (channels)
        # Each mask is now 256 channels

        # Apply masks
        masked_features1 = x * mask1
        masked_features2 = x * mask2

        # Decoding
        violin = self.decoder(masked_features1)
        piano = self.decoder(masked_features2)
        
        violin = torch.tanh(violin)  # Normalize to (-1, 1)
        piano = torch.tanh(piano)  # Normalize to (-1, 1)

        # Combine violin and piano outputs into a single tensor
        combined_output = torch.cat([violin, piano], dim=1)

        return combined_output  # Return a single tensor with shape [batch, 4, time]
