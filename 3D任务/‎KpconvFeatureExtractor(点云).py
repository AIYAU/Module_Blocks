import torch
import torch.nn as nn
import numpy as np

""" V2PNet：体素到点特征传播和融合，可改善点云配准的特征表示
基于点和基于体素的方法可以学习点云的局部特征。然而，尽管基于点的方法在几何上是精确的，但点云的离散性质会对特征学习性能产生负面影响。此外，尽管基于体素的方法可以利用卷积神经网络的学习能力，但它们的分辨率和细节提取可能不足。
因此，在本研究中，基于点和基于体素的方法相结合，以提高定位精度和匹配独特性。核心程序体现在 V2PNet 中，这是我们设计的一种创新的融合神经网络，用于执行体素到像素的传播和融合，它无缝集成了两个编码器-解码器分支。
实验是在具有不同平台和传感器的室内和室外基准数据集上进行的，即 3DMatch 和kitti数据集，注册召回率为 89.4%，成功率为 99.86%。
定性和定量评估表明，V2PNet 在语义感知、几何结构辨别和其他性能指标方面有所提升。
"""

# Define a dummy config class
class Config:
    def __init__(self):
        self.first_subsampling_dl = 0.1
        self.conv_radius = 2.0
        self.kpconv_in_dim = 32
        self.first_features_dim = 16
        self.num_kernel_points = 15
        self.kpconv_architecture = ['simple', 'simple', 'pool', 'simple', 'upsample']
        self.voxel_size = 0.05

# Define the block_decoder function
def block_decoder(block, r, in_dim, out_dim, layer, config):
    if 'simple' in block:
        return nn.Conv1d(in_dim, out_dim, kernel_size=1)
    elif 'pool' in block or 'strided' in block:
        return nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)
    elif 'upsample' in block:
        return nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2)
    else:
        raise ValueError(f"Unknown block type: {block}")


class KpconvEncoder(nn.Module):
    """
       Class defining KpconvEncoder
    """
    CHANNELS = [None, 16, 32, 64, 128, 256, 512, 1024, 2048]

    def __init__(self, config):
        super(KpconvEncoder, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.kpconv_in_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.kpconv_architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decoder(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
        return

    def forward(self, batch):
        # Get input features
        x = batch['features'].clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x)

        return x, skip_x, batch

class KpconvDecoder(nn.Module):
    """
       Class defining KpconvDecoder
    """

    def __init__(self, config):
        super(KpconvDecoder, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.first_features_dim * (2 ** (len(config.kpconv_architecture) - 2))  # Correct initial input dimension for decoder
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []
        self.decoder_skip_dims = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.kpconv_architecture):
            if 'upsample' in block:
                start_i = block_i
                break

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.decoder_skip_dims.append(in_dim)

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                layer += 1
                r *= 2
                out_dim *= 2

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.kpconv_architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.kpconv_architecture[start_i + block_i - 1]:
                in_dim += self.decoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decoder(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim
        return

    def forward(self, x, skip_x):
        # Get input features
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x)

        return x

class KpconvFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(KpconvFeatureExtractor, self).__init__()

        self.voxel_size = config.voxel_size

        # Initialize the backbone network
        self.encoder = KpconvEncoder(config)
        self.decoder = KpconvDecoder(config)

    def forward(self, batch):
        enc_feat, skip_x, batch = self.encoder(batch)
        dec_feat = self.decoder(enc_feat, skip_x)
        kpconv_feat = dec_feat
        return kpconv_feat

if __name__ == '__main__':
    config = Config()

    block = KpconvFeatureExtractor(config)

    # Mock batch input
    batch = {
        'features': torch.rand(8, 32, 64)  # Example: Batch size 8, 3 features, length 64
    }

    output = block(batch)
    print(output.size())

