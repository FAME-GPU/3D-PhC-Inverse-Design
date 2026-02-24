from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from monai.networks.blocks.encoder import BaseEncoder
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer, get_pool_layer
from utils.SymmetryAwareAtt import SymmetryAwareAtt
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option, optional_import

hf_hub_download, _ = optional_import("huggingface_hub", name="hf_hub_download")
EntryNotFoundError, _ = optional_import("huggingface_hub.utils._errors", name="EntryNotFoundError")

MEDICALNET_HUGGINGFACE_REPO_BASENAME = "TencentMedicalNet/MedicalNet-Resnet"
MEDICALNET_HUGGINGFACE_FILES_BASENAME = "resnet_"

__all__ = [
    "ResNet",
    "ResNetBlock",
    "ResNetBottleneck",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]

resnet_params = {
    # model_name: (block, layers, shortcut_type, bias_downsample, datasets23)
    "resnet10": ("basic", [1, 1, 1, 1], "B", False, True),
    "resnet18": ("basic", [2, 2, 2, 2], "A", True, True),
    "resnet34": ("basic", [3, 4, 6, 3], "A", True, True),
    "resnet50": ("bottleneck", [3, 4, 6, 3], "B", False, True),
    "resnet101": ("bottleneck", [3, 4, 23, 3], "B", False, False),
    "resnet152": ("bottleneck", [3, 8, 36, 3], "B", False, False),
    "resnet200": ("bottleneck", [3, 24, 36, 3], "B", False, False),
}

logger = logging.getLogger(__name__)


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]

def mul(x, y):
    out = torch.matmul(x, y)
    return out

class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.act = get_act_layer(name=act)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out

class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_layer = partial(get_norm_layer, name=norm, spatial_dims=spatial_dims)

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(channels=planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(channels=planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(channels=planes * self.expansion)
        self.act = get_act_layer(name=act)
        self.downsample = downsample
        self.stride = stride
        self.SAA = SymmetryAwareAtt(planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        y = self.SAA(out)
        out = out + y
        out = out + residual
        out = self.act(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=tuple(k // 2 for k in conv1_kernel_size),
            bias=False,
        )

        norm_layer = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=self.in_planes)
        self.bn1 = norm_layer
        self.act = get_act_layer(name=act)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, type(norm_layer)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            '''elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)'''

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: type[ResNetBlock | ResNetBottleneck],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
        norm: str | tuple = "batch",
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        downsample: nn.Module | partial | None = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                spatial_dims=spatial_dims,
                stride=stride,
                downsample=downsample,
                norm=norm,
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims, norm=norm))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x


class ResNetFeatures(ResNet):

    def __init__(self, model_name: str, pretrained: bool = True, spatial_dims: int = 3, in_channels: int = 1) -> None:
        if model_name not in resnet_params:
            model_name_string = ", ".join(resnet_params.keys())
            raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

        block, layers, shortcut_type, bias_downsample, datasets23 = resnet_params[model_name]

        super().__init__(
            block=block,
            layers=layers,
            block_inplanes=get_inplanes(),
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            conv1_t_stride=2,
            shortcut_type=shortcut_type,
            feed_forward=False,
            bias_downsample=bias_downsample,
        )
        if pretrained:
            if spatial_dims == 3 and in_channels == 1:
                _load_state_dict(self, model_name, datasets23=datasets23)
            else:
                raise ValueError("Pretrained resnet models are only available for in_channels=1 and spatial_dims=3.")

    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        features = []
        features.append(x)

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features


class ResNetEncoder(ResNetFeatures, BaseEncoder):

    backbone_names = ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]

    @classmethod
    def get_encoder_parameters(cls) -> list[dict]:
        parameter_list = []
        for backbone_name in cls.backbone_names:
            parameter_list.append(
                {"model_name": backbone_name, "pretrained": True, "spatial_dims": 3, "in_channels": 1}
            )
        return parameter_list

    @classmethod
    def num_channels_per_output(cls) -> list[tuple[int, ...]]:
        return [
            (64, 64, 128, 256, 512),
            (64, 64, 128, 256, 512),
            (64, 64, 128, 256, 512),
            (64, 256, 512, 1024, 2048),
            (64, 256, 512, 1024, 2048),
            (64, 256, 512, 1024, 2048),
            (64, 256, 512, 1024, 2048),
        ]

    @classmethod
    def num_outputs(cls) -> list[int]:
        return [5] * 7

    @classmethod
    def get_encoder_names(cls) -> list[str]:
        return cls.backbone_names


def _resnet(
    arch: str,
    block: type[ResNetBlock | ResNetBottleneck],
    layers: list[int],
    block_inplanes: list[int],
    pretrained: bool | str,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, **kwargs)
    if pretrained:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(pretrained, str):
            if Path(pretrained).exists():
                logger.info(f"Loading weights from {pretrained}...")
                model_state_dict = torch.load(pretrained, map_location=device)
            else:
                # Throw error
                raise FileNotFoundError("The pretrained checkpoint file is not found")
        else:
            # Also check bias downsample and shortcut.
            if kwargs.get("spatial_dims", 3) == 3:
                if kwargs.get("n_input_channels", 3) == 1 and kwargs.get("feed_forward", True) is False:
                    search_res = re.search(r"resnet(\d+)", arch)
                    if search_res:
                        resnet_depth = int(search_res.group(1))
                    else:
                        raise ValueError("arch argument should be as 'resnet_{resnet_depth}")

                    # Check model bias_downsample and shortcut_type
                    bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(resnet_depth)
                    if shortcut_type == kwargs.get("shortcut_type", "B") and (
                        bias_downsample == kwargs.get("bias_downsample", True)
                    ):
                        # Download the MedicalNet pretrained model
                        model_state_dict = get_pretrained_resnet_medicalnet(
                            resnet_depth, device=device, datasets23=True
                        )
                    else:
                        raise NotImplementedError(
                            f"Please set shortcut_type to {shortcut_type} and bias_downsample to {bias_downsample} "
                            f"when using pretrained MedicalNet resnet{resnet_depth}"
                        )
                else:
                    raise NotImplementedError(
                        "Please set n_input_channels to 1"
                        "and feed_forward to False in order to use MedicalNet pretrained weights"
                    )
            else:
                raise NotImplementedError("MedicalNet pretrained weights are only avalaible for 3D models")
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        model.load_state_dict(model_state_dict, strict=True)
    return model


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet101", ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet152", ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet200(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet200", ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


def get_pretrained_resnet_medicalnet(resnet_depth: int, device: str = "cpu", datasets23: bool = True):

    medicalnet_huggingface_repo_basename = "TencentMedicalNet/MedicalNet-Resnet"
    medicalnet_huggingface_files_basename = "resnet_"
    supported_depth = [10, 18, 34, 50, 101, 152, 200]

    logger.info(
        f"Loading MedicalNet pretrained model from https://huggingface.co/{medicalnet_huggingface_repo_basename}{resnet_depth}"
    )

    if resnet_depth in supported_depth:
        filename = (
            f"{medicalnet_huggingface_files_basename}{resnet_depth}.pth"
            if not datasets23
            else f"{medicalnet_huggingface_files_basename}{resnet_depth}_23dataset.pth"
        )
        try:
            pretrained_path = hf_hub_download(
                repo_id=f"{medicalnet_huggingface_repo_basename}{resnet_depth}", filename=filename
            )
        except Exception:
            if datasets23:
                logger.info(f"{filename} not available for resnet{resnet_depth}")
                filename = f"{medicalnet_huggingface_files_basename}{resnet_depth}.pth"
                logger.info(f"Trying with {filename}")
                pretrained_path = hf_hub_download(
                    repo_id=f"{medicalnet_huggingface_repo_basename}{resnet_depth}", filename=filename
                )
            else:
                raise EntryNotFoundError(
                    f"{filename} not found on {medicalnet_huggingface_repo_basename}{resnet_depth}"
                ) from None
        checkpoint = torch.load(pretrained_path, map_location=torch.device(device))
    else:
        raise NotImplementedError("Supported resnet_depth are: [10, 18, 34, 50, 101, 152, 200]")
    logger.info(f"{filename} downloaded")
    return checkpoint.get("state_dict")


def get_medicalnet_pretrained_resnet_args(resnet_depth: int):
    bias_downsample = resnet_depth in (18, 34)
    shortcut_type = "A" if resnet_depth in [18, 34] else "B"
    return bias_downsample, shortcut_type


def _load_state_dict(model: nn.Module, model_name: str, datasets23: bool = True) -> None:
    search_res = re.search(r"resnet(\d+)", model_name)
    if search_res:
        resnet_depth = int(search_res.group(1))
        datasets23 = model_name.endswith("_23datasets")
    else:
        raise ValueError("model_name argument should contain resnet depth. Example: resnet18 or resnet18_23datasets.")

    model_state_dict = get_pretrained_resnet_medicalnet(resnet_depth, device="cpu", datasets23=datasets23)
    model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
    model.load_state_dict(model_state_dict)

if __name__ == '__main__':
    x = torch.randn(8, 1 ,60, 60, 60)
    model = ResNet(
        block='bottleneck',
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type='B',
        widen_factor=1.0,
        num_classes=560,
        feed_forward=True,
        bias_downsample=True,
        act=('relu', {'inplace': True}),
        norm='batch'
    )
    print(model)
    y = model(x)
    print(y.shape)
