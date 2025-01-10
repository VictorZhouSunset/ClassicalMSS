import torch
import torch.nn as nn
# Define the Wave-U-Net model

class WaveUNet(nn.Module):
    def __init__(self, num_layers=12, num_initial_filters=24, dropout_rate=0.1):
        super(WaveUNet, self).__init__()
        self.num_layers = num_layers
        self.name = f"WaveUNet_{num_layers}_{num_initial_filters}_dropout{dropout_rate}"
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Downsampling path
        self.downsample_conv = nn.ModuleList()
        for i in range(num_layers + 1):
            in_channels = 2 if i == 0 else num_initial_filters * (2**(i-1))  # Changed from 1 to 2
            out_channels = num_initial_filters * (2**i)
            self.downsample_conv.append(nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=1, padding=7))
        
        # Upsampling path
        self.upsample_conv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_initial_filters * (2**(num_layers-i))
            out_channels = num_initial_filters * (2**(num_layers-i-1))
            self.upsample.append(nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2))
            self.upsample_conv.append(nn.Conv1d(in_channels + num_initial_filters * (2**(num_layers-i-1)), out_channels, kernel_size=5, stride=1, padding=2))
        
        # Final convolution
        self.final_conv = nn.Conv1d(num_initial_filters + 2, 4, kernel_size=1, stride=1)  # Changed from 2 to 4
    
    def forward(self, x):
        # Downsampling
        features = []
        features.append(x) # The first feature is the input itself
        for i in range(self.num_layers):
            x = self.downsample_conv[i](x)
            features.append(x)
            x = nn.functional.leaky_relu(x, 0.2)
            x = self.dropout(x)  # Add dropout after activation
            x = x[:, :, ::2] 
        x = self.downsample_conv[self.num_layers](x)
        # features.append(x)
        # Upsampling
        for i in range(self.num_layers):
            x = self.upsample[i](x)
            # Ensure dimensions match by padding if necessary
            if x.size(2) != features[self.num_layers-i].size(2):
                diff = features[self.num_layers-i].size(2) - x.size(2)
                x = nn.functional.pad(x, (0, diff))
            # Skip connection
            x = torch.cat([x, features[self.num_layers-i]], dim=1)
            x = self.upsample_conv[i](x)
            x = nn.functional.leaky_relu(x, 0.2)
            x = self.dropout(x)  # Add dropout after activation

        x = torch.cat([x, features[0]], dim=1)
        # Final convolution
        x = self.final_conv(x)
        # Apply tanh activation
        x = torch.tanh(x)

        return x


class WaveUNet_Mono(nn.Module):
    def __init__(self, num_layers=12, num_initial_filters=24):
        super(WaveUNet_Mono, self).__init__()
        self.num_layers = num_layers
        self.name = "WaveUNet_Mono_" + str(num_layers) + "_" + str(num_initial_filters)
        
        # Downsampling path
        self.downsample_conv = nn.ModuleList()
        for i in range(num_layers + 1):
            in_channels = 1 if i == 0 else num_initial_filters * (2**(i-1))  # Changed from 2 to 1
            out_channels = num_initial_filters * (2**i)
            self.downsample_conv.append(nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=1, padding=7))
        
        # Upsampling path
        self.upsample_conv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_initial_filters * (2**(num_layers-i))
            out_channels = num_initial_filters * (2**(num_layers-i-1))
            self.upsample.append(nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2))
            self.upsample_conv.append(nn.Conv1d(in_channels + num_initial_filters * (2**(num_layers-i-1)), out_channels, kernel_size=5, stride=1, padding=2))
        
        # Final convolution
        self.final_conv = nn.Conv1d(num_initial_filters + 1, 2, kernel_size=1, stride=1)  # Changed from 4 to 1 output channels, and from 2 to 1 input channels
    
    def forward(self, x):
        # Downsampling
        features = []
        features.append(x) # The first feature is the input itself
        for i in range(self.num_layers):
            x = self.downsample_conv[i](x)
            features.append(x)
            x = nn.functional.leaky_relu(x, 0.2)
            x = x[:, :, ::2] 
        x = self.downsample_conv[self.num_layers](x)
        # features.append(x)
        # Upsampling
        for i in range(self.num_layers):
            x = self.upsample[i](x)
            # Ensure dimensions match by padding if necessary
            if x.size(2) != features[self.num_layers-i].size(2):
                diff = features[self.num_layers-i].size(2) - x.size(2)
                x = nn.functional.pad(x, (0, diff))
            x = torch.cat([x, features[self.num_layers-i]], dim=1)
            x = self.upsample_conv[i](x)
            x = nn.functional.leaky_relu(x, 0.2)

        x = torch.cat([x, features[0]], dim=1)
        # Final convolution
        x = self.final_conv(x)
        # Apply tanh activation
        x = torch.tanh(x)

        return x

class WaveUNet_Simple(nn.Module):
    def __init__(self, num_layers=3, num_initial_filters=8, pool_size=4, tanh=True, leaky_alpha=0.2, dropout_rate=0.1):
        super(WaveUNet_Simple, self).__init__()
        self.num_layers = num_layers
        self.tanh = tanh
        self.leaky_alpha = leaky_alpha
        if leaky_alpha == 0.2:
            if tanh:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_tanh_dropout{dropout_rate}"
            else:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_dropout{dropout_rate}"
        else:
            if tanh:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_leaky{leaky_alpha}_tanh_dropout{dropout_rate}"
            else:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_leaky{leaky_alpha}_dropout{dropout_rate}"
        self.pool_size = pool_size

        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Downsampling path
        self.downsample_conv = nn.ModuleList()
        for i in range(num_layers + 1):
            # i=0: in_channels = 2, out_channels = num_initial_filters
            # i=1: in_channels = num_initial_filters, out_channels = num_initial_filters * 2
            # i=num_layers: out_channels = num_initial_filters * (2**num_layers)
            in_channels = 2 if i == 0 else num_initial_filters * (pool_size**(i-1))  # Changed from 1 to 2
            out_channels = num_initial_filters * (pool_size**i)
            self.downsample_conv.append(nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=1, padding=7))
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # Upsampling path
        self.upsample_conv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(num_layers):
            # i=0: in_channels = num_initial_filters * (2**num_layers), out_channels = num_initial_filters * (2**(num_layers-1))
            in_channels = num_initial_filters * (pool_size**(num_layers-i))
            out_channels = num_initial_filters * (pool_size**(num_layers-i-1))
            self.upsample.append(nn.ConvTranspose1d(in_channels, in_channels, kernel_size=pool_size, stride=pool_size))
            # i=0时，concatenate必须length一样，由于已经进行ConvTranspose1d, 所以只能和
            # upsampling的倒数第二层 concatenate
            self.upsample_conv.append(nn.Conv1d(in_channels + num_initial_filters * (pool_size**(num_layers-i-1)), out_channels, kernel_size=5, stride=1, padding=2))
        
        # Final convolution
        self.final_conv = nn.Conv1d(num_initial_filters + 2, 4, kernel_size=1, stride=1)  # Changed from 2 to 4
    
    def forward(self, x):
        # Downsampling
        features = []
        features.append(x) # The first feature is the input itself
        for i in range(self.num_layers):
            # The final layer of downsampling is not concatenated
            x = self.downsample_conv[i](x)
            features.append(x)
            x = nn.functional.leaky_relu(x, self.leaky_alpha)
            x = self.maxpool(x)
            x = self.dropout(x)  # Add dropout after activation and maxpool to add efficiency
        
        x = self.downsample_conv[self.num_layers](x)
        # features.append(x)
        # ----------------------------
        # Upsampling
        for i in range(self.num_layers):
            x = self.upsample[i](x)
            # Ensure dimensions match by padding if necessary
            # i=0时，upsample的结果和倒数第二层concatenate，但因为feature的第一个元素是初始mix audio
            if x.size(2) != features[self.num_layers-i].size(2):
                diff = features[self.num_layers-i].size(2) - x.size(2)
                x = nn.functional.pad(x, (0, diff))
            # Skip connection
            x = torch.cat([x, features[self.num_layers-i]], dim=1)
            x = self.upsample_conv[i](x)
            x = nn.functional.leaky_relu(x, self.leaky_alpha)
            x = self.dropout(x)  # Add dropout after activation

        x = torch.cat([x, features[0]], dim=1)
        # Final convolution
        x = self.final_conv(x)
        if self.tanh:
            # Apply tanh activation
            x = torch.tanh(x)

        return x


class WaveUNet_Simple_Aux(nn.Module):
    def __init__(self, num_layers=3, num_initial_filters=16, pool_size=4, tanh=True, leaky_alpha=0.2, dropout_rate=0.1):
        super(WaveUNet_Simple_Aux, self).__init__()
        self.num_layers = num_layers
        self.tanh = tanh
        self.leaky_alpha = leaky_alpha
        if leaky_alpha == 0.2:
            if tanh:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_tanh_dropout{dropout_rate}"
            else:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_dropout{dropout_rate}"
        else:
            if tanh:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_leaky{leaky_alpha}_tanh_dropout{dropout_rate}"
            else:
                self.name = f"WaveUNet_Simple_{num_layers}_{num_initial_filters}_pool{pool_size}_leaky{leaky_alpha}_dropout{dropout_rate}"
        self.pool_size = pool_size

        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Downsampling path
        self.downsample_conv = nn.ModuleList()
        for i in range(num_layers + 1):
            # i=0: in_channels = 2, out_channels = num_initial_filters
            # i=1: in_channels = num_initial_filters, out_channels = num_initial_filters * 2
            # i=num_layers: out_channels = num_initial_filters * (2**num_layers)
            in_channels = 6 if i == 0 else num_initial_filters * (pool_size**(i-1))  # Changed from 1 to 2
            out_channels = num_initial_filters * (pool_size**i)
            self.downsample_conv.append(nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=1, padding=7))
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # Upsampling path
        self.upsample_conv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(num_layers):
            # i=0: in_channels = num_initial_filters * (2**num_layers), out_channels = num_initial_filters * (2**(num_layers-1))
            in_channels = num_initial_filters * (pool_size**(num_layers-i))
            out_channels = num_initial_filters * (pool_size**(num_layers-i-1))
            self.upsample.append(nn.ConvTranspose1d(in_channels, in_channels, kernel_size=pool_size, stride=pool_size))
            # i=0时，concatenate必须length一样，由于已经进行ConvTranspose1d, 所以只能和
            # upsampling的倒数第二层 concatenate
            self.upsample_conv.append(nn.Conv1d(in_channels + num_initial_filters * (pool_size**(num_layers-i-1)), out_channels, kernel_size=5, stride=1, padding=2))
        
        # Final convolution
        self.final_conv = nn.Conv1d(num_initial_filters + 2, 4, kernel_size=1, stride=1)  # Changed from 2 to 4
    
    def forward(self, x):
        # Downsampling
        features = []
        features.append(x) # The first feature is the input itself
        for i in range(self.num_layers):
            # The final layer of downsampling is not concatenated
            x = self.downsample_conv[i](x)
            features.append(x)
            x = nn.functional.leaky_relu(x, self.leaky_alpha)
            x = self.maxpool(x)
            x = self.dropout(x)  # Add dropout after activation and maxpool to add efficiency
        
        x = self.downsample_conv[self.num_layers](x)
        # features.append(x)
        # ----------------------------
        # Upsampling
        for i in range(self.num_layers):
            x = self.upsample[i](x)
            # Ensure dimensions match by padding if necessary
            # i=0时，upsample的结果和倒数第二层concatenate，但因为feature的第一个元素是初始mix audio
            if x.size(2) != features[self.num_layers-i].size(2):
                diff = features[self.num_layers-i].size(2) - x.size(2)
                x = nn.functional.pad(x, (0, diff))
            # Skip connection
            x = torch.cat([x, features[self.num_layers-i]], dim=1)
            x = self.upsample_conv[i](x)
            x = nn.functional.leaky_relu(x, self.leaky_alpha)
            x = self.dropout(x)  # Add dropout after activation

        x = torch.cat([x, features[0]], dim=1)
        # Final convolution
        x = self.final_conv(x)
        if self.tanh:
            # Apply tanh activation
            x = torch.tanh(x)

        return x
