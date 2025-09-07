import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU block (optionally with downsample via stride)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CCAiMModel(nn.Module):
    """
    Compact convolutional classifier aimed to be ~9M parameters by default.

    Structure (default channels):
      stem: 3 -> 64
      stages: 64 -> 128 -> 256 -> 512 -> 512
      each stage contains 2 ConvBlock layers (first may downsample)

    Notes:
      - num_classes: number of output classes
      - width_mult: multiply all channel counts by this factor (int)
      - input image size: assumed >= 64x64 (for sensible feature map sizes)
    """

    def __init__(self, num_classes: int = 10, width_mult: int = 1):
        super().__init__()
        # Base channel configuration chosen to target ~9M params
        base_channels = [64, 128, 256, 512, 512]
        # allow simple integer width multiplier (1 keeps ~9M params)
        if width_mult != 1:
            base_channels = [max(8, int(c * width_mult)) for c in base_channels]

        self.stem = ConvBlock(3, base_channels[0], kernel_size=3, stride=1, padding=1)

        stages = []
        in_ch = base_channels[0]
        for out_ch in base_channels[1:]:
            # first layer in stage downsamples (stride=2) to reduce spatial size
            stages.append(ConvBlock(in_ch, out_ch, stride=2))
            # second layer keeps same spatial resolution
            stages.append(ConvBlock(out_ch, out_ch, stride=1))
            in_ch = out_ch

        self.features = nn.Sequential(*stages)

        # projection conv to a compact bottleneck before the classifier
        self.proj = nn.Conv2d(in_ch, 1024, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(1024)
        self.proj_relu = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.proj_relu(self.proj_bn(self.proj(x)))
        x = self.global_pool(x)  # B x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Utility: count parameters (trainable)
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick smoke/test example (runs only when script executed directly)
if __name__ == '__main__':
    # create model for 10 classes (modify num_classes as needed)
    model = CCAiMModel(num_classes=10, width_mult=1)
    print('Model:', model.__class__.__name__)
    print('Trainable parameters:', count_parameters(model))

    # test forward with a dummy batch (e.g. 8 images 3x224x224)
    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    print('Output shape:', y.shape)
