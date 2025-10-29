import torch
import torch.nn as nn
import math


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Implementação baseada no paper "Squeeze-and-Excitation Networks"
    """
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Número de canais de entrada
            reduction: Fator de redução para o bottleneck
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: multiply input by attention weights
        return x * y.expand_as(x)


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention Block para visão computacional
    Implementação simplificada de self-attention espacial
    """
    def __init__(self, channels):
        """
        Args:
            channels: Número de canais de entrada
        """
        super(SelfAttentionBlock, self).__init__()
        self.channels = channels
        
        # Query, Key, Value transformations
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: input feature maps (B, C, H, W)
        Returns:
            out: attention applied features (B, C, H, W)
        """
        batch_size, C, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        
        # Compute attention map
        energy = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = self.softmax(energy)  # B x (H*W) x (H*W)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)  # B x C x H x W
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class ResnextBlock(nn.Module):
    """
    ResNeXt block com grouped convolutions.
    Implementação inspirada no paper "Aggregated Residual Transformations for Deep Neural Networks"
    Suporta diferentes tipos de attention mechanisms.
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=32, width_per_group=4, 
                 block_type="residual", se_reduction=16):
        """
        Args:
            in_channels: Número de canais de entrada
            out_channels: Número de canais de saída
            stride: Stride para a convolução (usado para downsampling)
            groups: Número de grupos para grouped convolution (cardinality)
            width_per_group: Largura de cada grupo
            block_type: Tipo de bloco - "residual", "se_attention" ou "self_attention"
            se_reduction: Fator de redução para SE block (usado apenas se block_type="se_attention")
        """
        super(ResnextBlock, self).__init__()
        
        self.block_type = block_type
        
        # Calcular o número de canais intermediários
        width = groups * width_per_group
        
        # Primeira convolução 1x1 (bottleneck)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # Segunda convolução 3x3 com grouped convolution
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        # Terceira convolução 1x1 (expansão)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Attention mechanism baseado no block_type
        if self.block_type == "se_attention":
            self.attention = SEBlock(out_channels, reduction=se_reduction)
        elif self.block_type == "self_attention":
            self.attention = SelfAttentionBlock(out_channels)
        else:  # residual
            self.attention = None
    
    def forward(self, x):
        identity = x
        
        # Bottleneck pathway
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Grouped convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Expansion
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply attention mechanism based on block_type
        if self.block_type == "residual":
            # Standard residual connection
            out += self.shortcut(identity)
        elif self.block_type == "se_attention":
            # Apply SE attention then add shortcut
            out = self.attention(out)
            out += self.shortcut(identity)
        elif self.block_type == "self_attention":
            # Add shortcut first, then apply self-attention
            out += self.shortcut(identity)
            out = self.attention(out)
        
        out = self.relu(out)
        
        return out


class AnyNetStage(nn.Module):
    """
    Stage do AnyNet - sequência de blocos ResNeXt
    """
    def __init__(self, in_channels, out_channels, num_blocks, stride=1, 
                 groups=32, width_per_group=4, block_type="residual", se_reduction=16):
        """
        Args:
            in_channels: Número de canais de entrada
            out_channels: Número de canais de saída
            num_blocks: Número de blocos ResNeXt no stage
            stride: Stride para o primeiro bloco (downsampling)
            groups: Número de grupos para grouped convolution
            width_per_group: Largura de cada grupo
            block_type: Tipo de bloco - "residual", "se_attention" ou "self_attention"
            se_reduction: Fator de redução para SE block
        """
        super(AnyNetStage, self).__init__()
        
        layers = []
        # Primeiro bloco com possível downsampling
        layers.append(ResnextBlock(in_channels, out_channels, stride, 
                                   groups, width_per_group, block_type, se_reduction))
        
        # Blocos restantes
        for _ in range(1, num_blocks):
            layers.append(ResnextBlock(out_channels, out_channels, 1, 
                                      groups, width_per_group, block_type, se_reduction))
        
        self.stage = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.stage(x)


class AnyNet(nn.Module):
    """
    Arquitetura AnyNet usando blocos ResNeXt
    """
    def __init__(self, num_classes=10, stem_channels=32, 
                 stage_channels=[64, 128, 256, 512],
                 stage_depths=[2, 3, 4, 3],
                 groups=32, width_per_group=4, 
                 block_type="residual", se_reduction=16):
        """
        Args:
            num_classes: Número de classes para classificação
            stem_channels: Canais do stem inicial
            stage_channels: Lista com número de canais para cada stage
            stage_depths: Lista com número de blocos para cada stage
            groups: Número de grupos para grouped convolution
            width_per_group: Largura de cada grupo
            block_type: Tipo de bloco - "residual", "se_attention" ou "self_attention"
            se_reduction: Fator de redução para SE block
        """
        super(AnyNet, self).__init__()
        
        # Stem inicial
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True)
        )
        
        # Construir stages
        self.stages = nn.ModuleList()
        in_channels = stem_channels
        
        for i, (out_channels, depth) in enumerate(zip(stage_channels, stage_depths)):
            stride = 2 if i > 0 else 1  # Primeiro stage sem downsampling
            stage = AnyNetStage(in_channels, out_channels, depth, stride, 
                               groups, width_per_group, block_type, se_reduction)
            self.stages.append(stage)
            in_channels = out_channels
        
        # Global Average Pooling e classificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[-1], num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Funções auxiliares para criar modelos pré-configurados
def create_anynet_small(num_classes=10, block_type="residual"):
    """Cria uma versão pequena do AnyNet"""
    return AnyNet(
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[64, 128, 256, 512],
        stage_depths=[2, 2, 3, 2],
        groups=8,
        width_per_group=4,
        block_type=block_type
    )


def create_anynet_medium(num_classes=10, block_type="residual"):
    """Cria uma versão média do AnyNet"""
    return AnyNet(
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[128, 256, 512, 1024],
        stage_depths=[2, 3, 4, 3],
        groups=32,
        width_per_group=4,
        block_type=block_type
    )


def create_anynet_large(num_classes=10, block_type="residual"):
    """Cria uma versão grande do AnyNet"""
    return AnyNet(
        num_classes=num_classes,
        stem_channels=64,
        stage_channels=[256, 512, 1024, 2048],
        stage_depths=[3, 4, 6, 3],
        groups=32,
        width_per_group=8,
        block_type=block_type
    )
