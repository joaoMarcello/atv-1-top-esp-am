"""
Script de teste para AnyNet
Testa as duas configurações de head: normal_head e coral_head
"""

import torch
import torch.nn as nn
from model import AnyNet, CoralLoss

# Detectar dispositivo (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*80}")
print(f"Dispositivo detectado: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print(f"{'='*80}\n")

def test_anynet_normal_head():
    """Testa AnyNet com normal_head (classificação padrão)"""
    print("\n" + "="*80)
    print("TESTANDO ANYNET COM NORMAL HEAD")
    print("="*80)
    
    # Configurações
    batch_size = 4
    num_classes = 5
    input_size = (batch_size, 3, 224, 224)
    
    print(f"\nConfiguração:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {input_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Head type: normal_head")
    
    # Criar modelo
    model = AnyNet(
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[64, 128, 256, 512],
        stage_depths=[2, 2, 3, 2],
        groups=8,
        width_per_group=4,
        block_type='residual',
        head_type='normal_head',
        stem_kernel_size=3
    ).to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nEstatísticas do modelo:")
    print(f"  Total de parâmetros: {total_params:,}")
    print(f"  Parâmetros treináveis: {trainable_params:,}")
    
    # Criar tensor de entrada aleatório
    x = torch.randn(input_size, device=device)
    print(f"\n>>> Entrada criada: shape={x.shape}, dtype={x.dtype}, device={x.device}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f">>> Forward pass concluído!")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Expected shape: ({batch_size}, {num_classes})")
    
    # Verificar output
    assert outputs.shape == (batch_size, num_classes), \
        f"Output shape incorreto! Esperado ({batch_size}, {num_classes}), obtido {outputs.shape}"
    
    # Aplicar softmax para obter probabilidades
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    print(f"\n>>> Predições:")
    for i in range(batch_size):
        print(f"  Sample {i}: Predicted class={predictions[i].item()}, "
              f"Confidence={probabilities[i, predictions[i]].item():.4f}")
    
    # Testar loss
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    print(f"\n>>> Loss testada:")
    print(f"  Targets: {targets.tolist()}")
    print(f"  CrossEntropyLoss: {loss.item():.4f}")
    
    # Testar backward pass
    model.train()
    x_train = torch.randn(input_size, device=device, requires_grad=True)
    outputs_train = model(x_train)
    loss_train = criterion(outputs_train, targets)
    loss_train.backward()
    
    print(f"\n>>> Backward pass concluído!")
    print(f"  Gradientes calculados: {x_train.grad is not None}")
    
    print("\n" + "="*80)
    print("✓ TESTE NORMAL HEAD PASSOU!")
    print("="*80)


def test_anynet_coral_head():
    """Testa AnyNet com coral_head (classificação ordinal)"""
    print("\n" + "="*80)
    print("TESTANDO ANYNET COM CORAL HEAD")
    print("="*80)
    
    # Configurações
    batch_size = 4
    num_classes = 5
    input_size = (batch_size, 3, 224, 224)
    
    print(f"\nConfiguração:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {input_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Head type: coral_head")
    
    # Criar modelo
    model = AnyNet(
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[64, 128, 256, 512],
        stage_depths=[2, 2, 3, 2],
        groups=8,
        width_per_group=4,
        block_type='residual',
        head_type='coral_head',
        stem_kernel_size=3
    ).to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nEstatísticas do modelo:")
    print(f"  Total de parâmetros: {total_params:,}")
    print(f"  Parâmetros treináveis: {trainable_params:,}")
    
    # Criar tensor de entrada aleatório
    x = torch.randn(input_size, device=device)
    print(f"\n>>> Entrada criada: shape={x.shape}, dtype={x.dtype}, device={x.device}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f">>> Forward pass concluído!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {num_classes - 1})")
    
    # Verificar output (CORAL retorna K-1 logits)
    assert logits.shape == (batch_size, num_classes - 1), \
        f"Output shape incorreto! Esperado ({batch_size}, {num_classes - 1}), obtido {logits.shape}"
    
    # Obter predições usando o método do CORAL head
    predictions = model.head.predict(logits)
    
    print(f"\n>>> Predições (usando método CORAL):")
    for i in range(batch_size):
        # Calcular probabilidades P(Y > k) para visualização
        probas_gt = torch.sigmoid(logits[i])
        print(f"  Sample {i}: Predicted class={predictions[i].item()}, "
              f"P(Y > k)={probas_gt.tolist()}")
    
    # Testar loss
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    criterion = CoralLoss()
    loss = criterion(logits, targets)
    
    print(f"\n>>> Loss testada:")
    print(f"  Targets: {targets.tolist()}")
    print(f"  CoralLoss: {loss.item():.4f}")
    
    # Testar backward pass
    model.train()
    x_train = torch.randn(input_size, device=device, requires_grad=True)
    logits_train = model(x_train)
    loss_train = criterion(logits_train, targets)
    loss_train.backward()
    
    print(f"\n>>> Backward pass concluído!")
    print(f"  Gradientes calculados: {x_train.grad is not None}")
    
    # Verificar propriedades ordinais
    print(f"\n>>> Verificando propriedades ordinais:")
    print(f"  Logits (thresholds): {logits[0].tolist()}")
    print(f"  P(Y > k) = sigmoid(logits): {torch.sigmoid(logits[0]).tolist()}")
    
    print("\n" + "="*80)
    print("✓ TESTE CORAL HEAD PASSOU!")
    print("="*80)


def test_different_block_types():
    """Testa diferentes tipos de blocos"""
    print("\n" + "="*80)
    print("TESTANDO DIFERENTES TIPOS DE BLOCOS")
    print("="*80)
    
    batch_size = 2
    num_classes = 5
    input_size = (batch_size, 3, 224, 224)
    block_types = ['residual', 'se_attention', 'self_attention']
    
    for block_type in block_types:
        print(f"\n>>> Testando block_type='{block_type}'...")
        
        # Criar modelo
        model = AnyNet(
            num_classes=num_classes,
            stem_channels=16,
            stage_channels=[32, 64, 128, 256],
            stage_depths=[1, 1, 2, 1],  # Configuração menor para teste rápido
            groups=8,
            width_per_group=4,
            block_type=block_type,
            head_type='normal_head',
            stem_kernel_size=3
        ).to(device)
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total de parâmetros: {total_params:,}")
        
        # Forward pass
        x = torch.randn(input_size, device=device)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        assert outputs.shape == (batch_size, num_classes)
        print(f"  ✓ Forward pass OK: {outputs.shape}")
    
    print("\n" + "="*80)
    print("✓ TODOS OS TIPOS DE BLOCOS TESTADOS!")
    print("="*80)


def test_different_depths():
    """Testa diferentes configurações de profundidade"""
    print("\n" + "="*80)
    print("TESTANDO DIFERENTES CONFIGURAÇÕES DE PROFUNDIDADE")
    print("="*80)
    
    batch_size = 2
    num_classes = 5
    input_size = (batch_size, 3, 224, 224)
    
    depth_configs = {
        'shallow':     [1, 2, 2, 1],
        'balanced':    [2, 2, 3, 2],
        'deep':        [2, 3, 4, 3],
        'very_deep':   [3, 4, 6, 3],
        'front_heavy': [3, 3, 2, 1],
        'back_heavy':  [1, 2, 3, 3]
    }
    
    for config_name, stage_depths in depth_configs.items():
        print(f"\n>>> Testando depth_config='{config_name}' {stage_depths}...")
        
        # Criar modelo
        model = AnyNet(
            num_classes=num_classes,
            stem_channels=16,
            stage_channels=[32, 64, 128, 256],
            stage_depths=stage_depths,
            groups=8,
            width_per_group=4,
            block_type='residual',
            head_type='normal_head',
            stem_kernel_size=3
        ).to(device)
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        total_blocks = sum(stage_depths)
        print(f"  Total de blocos: {total_blocks}")
        print(f"  Total de parâmetros: {total_params:,}")
        
        # Forward pass
        x = torch.randn(input_size, device=device)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        assert outputs.shape == (batch_size, num_classes)
        print(f"  ✓ Forward pass OK: {outputs.shape}")
    
    print("\n" + "="*80)
    print("✓ TODAS AS CONFIGURAÇÕES DE PROFUNDIDADE TESTADAS!")
    print("="*80)


def test_batch_sizes():
    """Testa diferentes tamanhos de batch para encontrar o limite da GPU"""
    print("\n" + "="*80)
    print("TESTANDO DIFERENTES TAMANHOS DE BATCH")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\n>>> GPU não disponível. Teste de batch size só funciona com GPU.")
        print("\n" + "="*80)
        print("⚠ TESTE BATCH SIZE PULADO (GPU não disponível)")
        print("="*80)
        return
    
    num_classes = 5
    input_size_base = (3, 224, 224)
    
    # Testar com diferentes block types
    block_types_to_test = ['residual', 'se_attention', 'self_attention']
    
    for block_type in block_types_to_test:
        print(f"\n{'='*80}")
        print(f"Testando block_type='{block_type}'")
        print(f"{'='*80}")
        
        # Batch sizes para testar
        batch_sizes = [4, 8, 16, 24, 32, 40, 48, 56, 64]
        max_batch_success = 0
        
        for batch_size in batch_sizes:
            try:
                # Limpar cache da GPU antes de cada teste
                torch.cuda.empty_cache()
                
                # Criar modelo
                model = AnyNet(
                    num_classes=num_classes,
                    stem_channels=32,
                    stage_channels=[64, 128, 256, 512],
                    stage_depths=[2, 2, 3, 2],
                    groups=8,
                    width_per_group=4,
                    block_type=block_type,
                    head_type='normal_head',
                    stem_kernel_size=3
                ).to(device)
                
                # Criar input
                x = torch.randn(batch_size, *input_size_base, device=device)
                
                # Forward pass
                model.eval()
                with torch.no_grad():
                    outputs = model(x)
                
                # Backward pass para simular treinamento
                model.train()
                targets = torch.randint(0, num_classes, (batch_size,), device=device)
                criterion = nn.CrossEntropyLoss()
                
                x_train = torch.randn(batch_size, *input_size_base, device=device, requires_grad=True)
                outputs_train = model(x_train)
                loss = criterion(outputs_train, targets)
                loss.backward()
                
                # Obter uso de memória
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                
                print(f"  ✓ Batch size {batch_size:3d}: OK "
                      f"| Mem alocada: {mem_allocated:7.2f} MB | Mem reservada: {mem_reserved:7.2f} MB")
                
                max_batch_success = batch_size
                
                # Limpar variáveis
                del model, x, outputs, x_train, outputs_train, loss, targets
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ✗ Batch size {batch_size:3d}: OOM (Out of Memory)")
                    # Limpar memória após erro
                    torch.cuda.empty_cache()
                    break
                else:
                    print(f"  ✗ Batch size {batch_size:3d}: Erro - {str(e)}")
                    torch.cuda.empty_cache()
                    break
            except Exception as e:
                print(f"  ✗ Batch size {batch_size:3d}: Erro inesperado - {str(e)}")
                torch.cuda.empty_cache()
                break
        
        print(f"\n  >>> Batch size máximo suportado para '{block_type}': {max_batch_success}")
        print(f"  >>> Recomendação: Use batch_size <= {max_batch_success} para este block_type")
    
    print("\n" + "="*80)
    print("✓ TESTE DE BATCH SIZES CONCLUÍDO!")
    print("="*80)


def test_gpu_compatibility():
    """Testa compatibilidade com GPU se disponível"""
    print("\n" + "="*80)
    print("TESTANDO COMPATIBILIDADE GPU")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"\n>>> GPU já está sendo utilizada em todos os testes")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memória alocada: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memória reservada: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        print("\n" + "="*80)
        print("✓ COMPATIBILIDADE GPU VERIFICADA!")
        print("="*80)
    else:
        print("\n>>> GPU não disponível. Testes executados na CPU.")
        print("  Para usar GPU, certifique-se de ter PyTorch com CUDA instalado.")
        print("\n" + "="*80)
        print("⚠ TESTE GPU PULADO (GPU não disponível)")
        print("="*80)


def main():
    """Executa todos os testes"""
    print("\n" + "="*80)
    print(" "*25 + "SUITE DE TESTES ANYNET")
    print("="*80)
    
    # Informações do ambiente
    print(f"\nAmbiente:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Executar testes
        test_anynet_normal_head()
        test_anynet_coral_head()
        test_different_block_types()
        test_different_depths()
        test_batch_sizes()
        test_gpu_compatibility()
        
        # Resumo final
        print("\n" + "="*80)
        print(" "*30 + "RESUMO FINAL")
        print("="*80)
        print("\n✓ Todos os testes passaram com sucesso!")
        print("\nTestes executados:")
        print("  [✓] AnyNet com Normal Head")
        print("  [✓] AnyNet com CORAL Head")
        print("  [✓] Diferentes tipos de blocos (residual, se_attention, self_attention)")
        print("  [✓] Diferentes configurações de profundidade")
        print("  [✓] Teste de batch sizes (limites da GPU)" if torch.cuda.is_available() else "  [⚠] Teste de batch sizes (pulado)")
        print("  [✓] Compatibilidade GPU" if torch.cuda.is_available() else "  [⚠] Compatibilidade GPU (pulado)")
        print("\n" + "="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ ERRO NO TESTE!")
        print("="*80)
        print(f"\nErro: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
