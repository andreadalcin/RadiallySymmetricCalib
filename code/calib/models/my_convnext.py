from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from torchvision.models.convnext import _log_api_usage_once, CNBlock, LayerNorm2d, Conv2dNormActivation, StochasticDepth, Permute

class My_CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
        dilation: List[int] = 1,
        stride: List[int] = 2,
        padding: List[int] = 0,
        polar: bool = False,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.polar = polar

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ", stride={stride}"
        s += ", dilation={dilation}"
        s += ", padding={padding}"
        s += ", polar={polar}"
        s += ")"
        return s.format(**self.__dict__)
    
    
def polar_padding(x:torch.Tensor, padding):
    """Circular padding on the width, zero on height"""
    if isinstance(padding, int):
        padding = [padding,padding]
    # pad: (padding_left, padding_right, padding_top, padding_bottom)
    # padding: (H,W)
    # Circular left, right
    x = F.pad(x, (padding[1], padding[1], 0, 0), mode='circular')

    # Pad bottom
    x = F.pad(x, (0, 0, 0, padding[0]), mode='constant', value=0)

    # Pad TOP
    x = F.pad(x, (0, 0, padding[0], 0), mode='reflect')
    # AVOID ARTIFACTS IN THE MIDDLE REGION

    return x


class PolarConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode, device, dtype)
        self.polar_padding = padding
        
    def forward(self, input: Tensor) -> Tensor:
        input = polar_padding(input, padding=self.polar_padding)
        return super().forward(input)
    

class My_CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            
        if conv_layer is None:
            conv_layer = nn.Conv2d

        self.block = nn.Sequential(
            conv_layer(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result
    

class ConvNeXt_FE(nn.Module):
    def __init__(
        self,
        block_setting: List[My_CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stem = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, My_CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = My_CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=stem,
                stride=stem,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            
            conv_layer = PolarConv2D if cnf.polar else nn.Conv2d 
            
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, conv_layer=conv_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        conv_layer(cnf.input_channels, cnf.out_channels, kernel_size=2, padding=cnf.padding, stride=cnf.stride, dilation=cnf.dilation),
                    )
                )

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
