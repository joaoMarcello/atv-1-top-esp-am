"""
Script para calcular média e desvio padrão de um dataset de imagens.
Útil para definir os valores de normalização no transforms.Normalize().

Uso:
    python calculate_dataset_stats.py --csv_file data/train.csv --data_dir /path/to/images
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def get_args():
    """
    Configura e processa argumentos de linha de comando
    
    Returns:
        argparse.Namespace: Objeto contendo todos os argumentos configurados
    """
    parser = argparse.ArgumentParser(
        description='Calcula média e desvio padrão de um dataset de imagens',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Arquivo CSV contendo os nomes das imagens (primeira coluna)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Diretório contendo as imagens')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Tamanho para redimensionar as imagens')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Número de imagens a processar por vez')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Número máximo de imagens a processar (None = todas)')
    
    return parser.parse_args()


def calculate_mean_std(csv_file, data_dir, image_size=224, batch_size=100, max_images=None):
    """
    Calcula a média e desvio padrão de um dataset de imagens.
    
    Args:
        csv_file (str): Path para o CSV com nomes das imagens
        data_dir (str): Diretório contendo as imagens
        image_size (int): Tamanho para redimensionar as imagens
        batch_size (int): Número de imagens a processar por vez
        max_images (int): Número máximo de imagens (None = todas)
    
    Returns:
        tuple: (mean, std) como arrays numpy com 3 valores (RGB)
    """
    # Ler CSV
    print(f"Lendo CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    image_names = df.iloc[:, 0].values  # Primeira coluna são os nomes das imagens
    
    # Limitar número de imagens se especificado
    if max_images is not None and max_images < len(image_names):
        print(f"Usando apenas {max_images} imagens de {len(image_names)} disponíveis")
        image_names = image_names[:max_images]
    else:
        print(f"Processando {len(image_names)} imagens")
    
    # Transform para redimensionar e converter para tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # Converte para [0, 1] e muda para (C, H, W)
    ])
    
    # Acumuladores para média e desvio padrão
    # Vamos calcular online usando o método de Welford
    n_pixels = 0
    sum_channels = torch.zeros(3)
    sum_squared_channels = torch.zeros(3)
    
    # Processar imagens em batches
    print(f"\nCalculando estatísticas...")
    for i in tqdm(range(0, len(image_names), batch_size), desc="Processando batches"):
        batch_names = image_names[i:i+batch_size]
        
        for img_name in batch_names:
            try:
                # Carregar e processar imagem
                img_path = os.path.join(data_dir, img_name)
                
                # Verificar se arquivo existe
                if not os.path.exists(img_path):
                    print(f"\nAviso: Arquivo não encontrado: {img_path}")
                    continue
                
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)  # Shape: (3, H, W), valores em [0, 1]
                
                # Calcular estatísticas por canal
                # img_tensor.shape = (3, H, W)
                n_pixels += img_tensor.shape[1] * img_tensor.shape[2]  # H * W
                
                # Somar valores por canal
                sum_channels += img_tensor.sum(dim=[1, 2])  # Soma ao longo de H e W
                sum_squared_channels += (img_tensor ** 2).sum(dim=[1, 2])
                
            except Exception as e:
                print(f"\nErro ao processar {img_name}: {e}")
                continue
    
    # Calcular média e desvio padrão
    mean = sum_channels / n_pixels
    std = torch.sqrt((sum_squared_channels / n_pixels) - (mean ** 2))
    
    return mean.numpy(), std.numpy()


def main():
    """Função principal"""
    args = get_args()
    
    # Verificar se arquivos/diretórios existem
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {args.csv_file}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {args.data_dir}")
    
    print("="*80)
    print("CÁLCULO DE ESTATÍSTICAS DO DATASET")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  CSV file: {args.csv_file}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max images: {args.max_images if args.max_images else 'Todas'}")
    
    # Calcular estatísticas
    mean, std = calculate_mean_std(
        csv_file=args.csv_file,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_images=args.max_images
    )
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("RESULTADOS")
    print("="*80)
    print(f"\nMédia (mean) por canal RGB:")
    print(f"  R: {mean[0]:.6f}")
    print(f"  G: {mean[1]:.6f}")
    print(f"  B: {mean[2]:.6f}")
    print(f"\nDesvio padrão (std) por canal RGB:")
    print(f"  R: {std[0]:.6f}")
    print(f"  G: {std[1]:.6f}")
    print(f"  B: {std[2]:.6f}")
    
    print(f"\n" + "-"*80)
    print("Para usar no transforms.Normalize():")
    print("-"*80)
    print(f"transforms.Normalize(")
    print(f"    mean=[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}],")
    print(f"    std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print(f")")
    
    # Comparação com ImageNet (referência)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    mean_diff = np.abs(mean - imagenet_mean)
    std_diff = np.abs(std - imagenet_std)
    
    print(f"\n" + "-"*80)
    print("Comparação com ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):")
    print("-"*80)
    print(f"Diferença absoluta na média: R={mean_diff[0]:.6f}, G={mean_diff[1]:.6f}, B={mean_diff[2]:.6f}")
    print(f"Diferença absoluta no std:   R={std_diff[0]:.6f}, G={std_diff[1]:.6f}, B={std_diff[2]:.6f}")
    
    avg_mean_diff = np.mean(mean_diff)
    avg_std_diff = np.mean(std_diff)
    
    if avg_mean_diff < 0.05 and avg_std_diff < 0.05:
        print("\n✓ Valores próximos do ImageNet - pode usar normalização padrão do ImageNet")
    else:
        print("\n⚠ Valores diferentes do ImageNet - considere usar os valores calculados acima")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
