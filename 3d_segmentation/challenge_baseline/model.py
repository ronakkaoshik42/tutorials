import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, MaxAvgPool, ResidualSELayer, SimpleASPP, UpSample
from monai.networks.layers.factories import Act, Norm
from monai.utils import ensure_tuple_rep

class ConvBNActBlock(nn.Module):
    """Two convolution layers with batch norm, leaky relu, dropout and SE block"""

    def __init__(self, in_channels, out_channels, dropout_p, spatial_dims: int = 2):
        super().__init__()
        self.conv_conv_se = nn.Sequential(
            Convolution(spatial_dims, in_channels, out_channels, kernel_size=3, norm=Norm.BATCH, act=Act.LEAKYRELU),
            nn.Dropout(dropout_p),
            Convolution(spatial_dims, out_channels, out_channels, kernel_size=3, norm=Norm.BATCH, act=Act.LEAKYRELU),
            ResidualSELayer(spatial_dims=spatial_dims, in_channels=out_channels, r=2),
        )

    def forward(self, x):
        return self.conv_conv_se(x)


class DownBlock(nn.Module):
    """
    Downsampling with a concatenation of max-pool and avg-pool, followed by ConvBNActBlock
    """

    def __init__(self, in_channels, out_channels, dropout_p, spatial_dims: int = 2):
        super().__init__()
        self.max_avg_pool = MaxAvgPool(spatial_dims=spatial_dims, kernel_size=2)
        self.conv = ConvBNActBlock(2 * in_channels, out_channels, dropout_p, spatial_dims=spatial_dims)

    def forward(self, x):
        x_pool = self.max_avg_pool(x)
        return self.conv(x_pool) + x_pool


class UpBlock(nn.Module):
    """Upssampling followed by ConvBNActBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True, dropout_p=0.5, spatial_dims: int = 2):
        super().__init__()
        self.up = UpSample(spatial_dims, in_channels1, in_channels2, scale_factor=2,
        #  with_conv=not bilinear
         )
        self.conv = ConvBNActBlock(in_channels2 * 2, out_channels, dropout_p, spatial_dims=spatial_dims)

    def forward(self, x1, x2):
        x_cat = torch.cat([x2, self.up(x1)], dim=1)
        return self.conv(x_cat) + x_cat


class Custom_UNet(nn.Module):
    def __init__(
        self,
        # dimensions: int = 3,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        features=(32, 64, 128, 256, 512),
        # act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        # norm: Union[str, tuple] = ("instance", {"affine": True}),
        # dropout: Union[float, tuple] = 0.0,
        dropout=(0.0, 0.0, 0.3, 0.4, 0.5),
        bilinear: bool = True,
        # upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(dimensions=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        ft_chns = ensure_tuple_rep(features, 5)
        # print(f"BasicUNet features: {fea}.")  

        f0_half = int(ft_chns[0] / 2)
        f1_half = int(ft_chns[1] / 2)
        f2_half = int(ft_chns[2] / 2)
        f3_half = int(ft_chns[3] / 2)


        self.in_conv = ConvBNActBlock(in_channels, ft_chns[0], dropout[0], spatial_dims)
        self.down1 = DownBlock(ft_chns[0], ft_chns[1], dropout[1], spatial_dims)
        self.down2 = DownBlock(ft_chns[1], ft_chns[2], dropout[2], spatial_dims)
        self.down3 = DownBlock(ft_chns[2], ft_chns[3], dropout[3], spatial_dims)
        self.down4 = DownBlock(ft_chns[3], ft_chns[4], dropout[4], spatial_dims)

        self.bridge0 = Convolution(spatial_dims, ft_chns[0], f0_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)
        self.bridge1 = Convolution(spatial_dims, ft_chns[1], f1_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)
        self.bridge2 = Convolution(spatial_dims, ft_chns[2], f2_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)
        self.bridge3 = Convolution(spatial_dims, ft_chns[3], f3_half, kernel_size=1, norm=Norm.BATCH, act=Act.LEAKYRELU)

        self.up1 = UpBlock(ft_chns[4], f3_half, ft_chns[3], bilinear, dropout[3], spatial_dims)
        self.up2 = UpBlock(ft_chns[3], f2_half, ft_chns[2], bilinear, dropout[2], spatial_dims)
        self.up3 = UpBlock(ft_chns[2], f1_half, ft_chns[1], bilinear, dropout[1], spatial_dims)
        self.up4 = UpBlock(ft_chns[1], f0_half, ft_chns[0], bilinear, dropout[0], spatial_dims)

        self.aspp = SimpleASPP(
            spatial_dims, ft_chns[4], int(ft_chns[4] / 4), kernel_sizes=[1, 3, 3, 3], dilations=[1, 2, 4, 6]
        )

        self.out_conv = Convolution(spatial_dims, ft_chns[0], out_channels, conv_only=True)


        # self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        # self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, dropout)
        # self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, dropout)
        # self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, dropout)
        # self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, dropout)

        # self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        # self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        # self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        # self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        # self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        # x0 = self.conv_0(x)

        # x1 = self.down_1(x0)
        # x2 = self.down_2(x1)
        # x3 = self.down_3(x2)
        # x4 = self.down_4(x3)

        # u4 = self.upcat_4(x4, x3)
        # u3 = self.upcat_3(u4, x2)
        # u2 = self.upcat_2(u3, x1)
        # u1 = self.upcat_1(u2, x0)

        # logits = self.final_conv(u1)
        x_shape = list(x.shape)
        if len(x_shape) == 5:
            [batch, chns, dim1, dim2, dim3] = x_shape
            new_shape = [batch * dim1, chns, dim2, dim3]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        elif len(x_shape) == 3:
            raise NotImplementedError("spatial dimension = 1 not supported.")

        x0 = self.in_conv(x)
        x0b = self.bridge0(x0)
        x1 = self.down1(x0)
        x1b = self.bridge1(x1)
        x2 = self.down2(x1)
        x2b = self.bridge2(x2)
        x3 = self.down3(x2)
        x3b = self.bridge3(x3)
        x4 = self.down4(x3)

        x4 = self.aspp(x4)

        x = self.up1(x4, x3b)
        x = self.up2(x, x2b)
        x = self.up3(x, x1b)
        x = self.up4(x, x0b)
        output = self.out_conv(x)

        if len(x_shape) == 5:
            new_shape = [batch, dim1] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output

        return logits