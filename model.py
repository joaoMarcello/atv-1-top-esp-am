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


class AnyNetHead(nn.Module):
    """
    Head (classificador) da AnyNet
    """
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels: Número de canais de entrada (saída do último stage)
            num_classes: Número de classes para classificação
        """
        super(AnyNetHead, self).__init__()
        
        # Global Average Pooling e classificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CoralHead(nn.Module):
    """
    CORAL (Consistent Rank Logits) Head para classificação ordinal
    
    Baseado no paper "Rank-consistent Ordinal Regression for Neural Networks"
    Implementa um esquema de classificação ordinal onde as classes têm uma ordem natural.
    
    Ao invés de prever K classes independentes, CORAL usa K-1 tarefas de classificação binária
    onde cada tarefa prevê se y > k para k = 0, 1, ..., K-2.
    
    Referência: https://arxiv.org/abs/1901.07884
    """
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels: Número de canais de entrada (saída do último stage)
            num_classes: Número de classes ordinais (K)
        """
        super(CoralHead, self).__init__()
        
        if num_classes < 2:
            raise ValueError("num_classes deve ser >= 2 para CORAL")
        
        self.num_classes = num_classes
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature extractor compartilhado
        self.fc = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        # Camada final para CORAL: K-1 neurônios de saída (um para cada threshold)
        # Cada neurônio prevê P(Y > k) para k = 0, 1, ..., K-2
        self.coral_fc = nn.Linear(in_channels // 2, 1, bias=False)
        
        # Biases independentes para cada threshold (K-1 biases)
        # Estes são os thresholds ordinais que definem as fronteiras entre classes
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1))
    
    def forward(self, x):
        """
        Forward pass do CORAL Head
        
        Args:
            x: Features de entrada (B, C, H, W)
            
        Returns:
            logits: Tensor (B, K-1) com logits para cada threshold ordinal
        """
        # Global Average Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Feature extraction compartilhada
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Projeção para espaço 1D compartilhado
        logits = self.coral_fc(x)  # (B, 1)
        
        # Adicionar os biases independentes para cada threshold
        # Broadcasting: (B, 1) + (K-1,) -> (B, K-1)
        logits = logits + self.coral_bias
        
        return logits
    
    def predict_proba(self, logits):
        """
        Converte os logits CORAL em probabilidades para cada classe
        
        Args:
            logits: Tensor (B, K-1) com logits de cada threshold
            
        Returns:
            probas: Tensor (B, K) com probabilidades de cada classe
        """
        # Aplicar sigmoid para obter P(Y > k) para cada threshold
        probas_gt = torch.sigmoid(logits)  # (B, K-1)
        
        # Calcular probabilidades cumulativas: P(Y >= k+1)
        # probas_gt[:, k] = P(Y > k) = P(Y >= k+1)
        
        # P(Y = 0) = 1 - P(Y > 0)
        proba_0 = 1 - probas_gt[:, 0:1]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k) para k = 1, ..., K-2
        probas_middle = probas_gt[:, :-1] - probas_gt[:, 1:]
        
        # P(Y = K-1) = P(Y > K-2)
        proba_last = probas_gt[:, -1:]
        
        # Concatenar todas as probabilidades
        probas = torch.cat([proba_0, probas_middle, proba_last], dim=1)
        
        return probas
    
    def predict(self, logits):
        """
        Prediz a classe ordinal mais provável
        
        Args:
            logits: Tensor (B, K-1) com logits de cada threshold
            
        Returns:
            predictions: Tensor (B,) com a classe predita para cada exemplo
        """
        probas = self.predict_proba(logits)
        predictions = torch.argmax(probas, dim=1)
        return predictions

# ...existing code...

class CoralLoss(nn.Module):
    """
    CORAL Loss para classificação ordinal
    
    Implementa a loss function do paper "Rank-consistent Ordinal Regression for Neural Networks"
    
    A loss é calculada como a soma das binary cross-entropies para cada threshold ordinal.
    Para cada exemplo com label y, queremos:
    - P(Y > k) = 1 para k < y
    - P(Y > k) = 0 para k >= y
    
    Referência: https://arxiv.org/abs/1901.07884
    """
    def __init__(self):
        super(CoralLoss, self).__init__()
    
    def forward(self, logits, targets):
        """
        Calcula a CORAL loss
        
        Args:
            logits: Tensor (B, K-1) com logits de cada threshold
            targets: Tensor (B,) com labels ordinais (valores de 0 a K-1)
            
        Returns:
            loss: Scalar tensor com a loss média do batch
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1) + 1
        
        # Criar matriz de labels binários para cada threshold
        # levels[i, k] = 1 se targets[i] > k, caso contrário 0
        levels = torch.zeros_like(logits)
        
        for i in range(batch_size):
            # Para label y, queremos Y > k para k = 0, 1, ..., y-1
            # Exemplo: se y=2, então Y > 0 = True, Y > 1 = True, Y > 2 = False
            for k in range(num_classes - 1):
                if targets[i] > k:
                    levels[i, k] = 1.0
        
        # Calcular binary cross-entropy para cada threshold
        # BCE = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, 
            levels,
            reduction='mean'
        )
        
        return loss


class CoralLossV2(nn.Module):
    """
    Versão alternativa da CORAL Loss com importance weighting
    
    Permite ponderar diferentemente os erros em diferentes thresholds.
    Útil quando alguns thresholds são mais importantes que outros.
    """
    def __init__(self, importance_weights=None):
        """
        Args:
            importance_weights: Tensor (K-1,) com pesos para cada threshold.
                               Se None, usa pesos uniformes.
        """
        super(CoralLossV2, self).__init__()
        self.importance_weights = importance_weights
    
    def forward(self, logits, targets):
        """
        Calcula a CORAL loss com importance weighting
        
        Args:
            logits: Tensor (B, K-1) com logits de cada threshold
            targets: Tensor (B,) com labels ordinais
            
        Returns:
            loss: Scalar tensor com a loss ponderada
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1) + 1
        
        # Criar matriz de labels binários
        levels = torch.zeros_like(logits)
        
        for i in range(batch_size):
            for k in range(num_classes - 1):
                if targets[i] > k:
                    levels[i, k] = 1.0
        
        # Calcular BCE para cada threshold sem redução
        loss_per_threshold = nn.functional.binary_cross_entropy_with_logits(
            logits, 
            levels,
            reduction='none'
        )
        
        # Aplicar pesos de importância se fornecidos
        if self.importance_weights is not None:
            if self.importance_weights.device != loss_per_threshold.device:
                self.importance_weights = self.importance_weights.to(loss_per_threshold.device)
            loss_per_threshold = loss_per_threshold * self.importance_weights
        
        # Média sobre batch e thresholds
        loss = loss_per_threshold.mean()
        
        return loss


class AnyNet(nn.Module):
    """
    Arquitetura AnyNet usando blocos ResNeXt
    """
    def __init__(self, num_classes=5, stem_channels=32, 
                 stage_channels=[64, 128, 256, 512],
                 stage_depths=[2, 3, 4, 3],
                 groups=32, width_per_group=4, 
                 block_type="residual", se_reduction=16, stem_kernel_size=3,
                 head_type="normal_head"):
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
            stem_kernel_size: Tamanho do kernel da convolução do stem
            head_type: Tipo de head - "normal_head" ou "coral_head"
        """
        super(AnyNet, self).__init__()
        
        if head_type not in ["normal_head", "coral_head"]:
            raise ValueError(f"head_type deve ser 'normal_head' ou 'coral_head', recebido: {head_type}")
        
        # Calcular padding para manter dimensões espaciais
        padding = stem_kernel_size // 2
        
        # Stem inicial com LayerNorm
        self.stem_conv = nn.Conv2d(3, stem_channels, kernel_size=stem_kernel_size, 
                                   stride=1, padding=padding, bias=False)
        self.stem_norm = nn.GroupNorm(1, stem_channels)  # GroupNorm com 1 grupo = LayerNorm
        self.stem_relu = nn.ReLU(inplace=True)
        
        # Criar Sequential para compatibilidade
        self.stem = nn.Sequential(
            self.stem_conv,
            self.stem_norm,
            self.stem_relu
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
        
        # Head (classificador) - escolher baseado no head_type
        if head_type == "coral_head":
            self.head = CoralHead(stage_channels[-1], num_classes)
        else:  # normal_head
            self.head = AnyNetHead(stage_channels[-1], num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.head(x)
        
        return x


# Funções auxiliares para criar modelos pré-configurados
def create_anynet_small(num_classes=10, block_type="residual", stem_kernel_size=3, head_type="normal_head"):
    """Cria uma versão pequena do AnyNet"""
    return AnyNet(
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[64, 128, 256, 512],
        stage_depths=[2, 2, 3, 2],
        groups=8,
        width_per_group=4,
        block_type=block_type,
        stem_kernel_size=stem_kernel_size,
        head_type=head_type
    )


def create_anynet_medium(num_classes=10, block_type="residual", stem_kernel_size=3, head_type="normal_head"):
    """Cria uma versão média do AnyNet"""
    return AnyNet(
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[128, 256, 512, 1024],
        stage_depths=[2, 3, 4, 3],
        groups=32,
        width_per_group=4,
        block_type=block_type,
        stem_kernel_size=stem_kernel_size,
        head_type=head_type
    )


def create_anynet_large(num_classes=10, block_type="residual", stem_kernel_size=3, head_type="normal_head"):
    """Cria uma versão grande do AnyNet"""
    return AnyNet(
        num_classes=num_classes,
        stem_channels=64,
        stage_channels=[256, 512, 1024, 2048],
        stage_depths=[3, 4, 6, 3],
        groups=32,
        width_per_group=8,
        block_type=block_type,
        stem_kernel_size=stem_kernel_size,
        head_type=head_type
    )
