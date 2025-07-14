# VQ-VAE 训练配置查看和图像生成指南

## 1. 查看训练时用的配置

### 方法1: 从训练脚本查看默认配置
```python
import torch
import os

# 设置路径（根据你的实际路径调整）
vqvae_repo = "F:/ProjectsL/00PatternProject2025/00ImageCollection202505/VQVAEModel/vq-vae-2-pytorch"  # 你的VQ-VAE仓库路径
checkpoint_dir = f"{vqvae_repo}/checkpoint"

# 查看train_vqvae.py中的默认配置
with open(f"{vqvae_repo}/train_vqvae.py", "r") as f:
    content = f.read()
    
# 打印VQVAE类的初始化参数（你的配置）
print("VQ-VAE 模型配置:")
print("in_channel=3")
print("channel=128") 
print("n_res_block=2")
print("n_res_channel=32")
print("embed_dim=64")
print("n_embed=512")
print("decay=0.99")
print("epoch=50")
print("size=256")
```

### 方法2: 从保存的模型文件查看配置
```python
# 加载最新的模型文件查看配置
latest_ckpt = f"{checkpoint_dir}/vqvae_050.pt"  # 你训练的最后一个epoch

if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location='cpu')
    print("模型文件信息:")
    print(f"文件路径: {latest_ckpt}")
    print(f"文件大小: {os.path.getsize(latest_ckpt) / (1024*1024):.2f} MB")
    
    # 查看模型状态字典的键
    if 'model' in ckpt:
        model_state = ckpt['model']
        print(f"模型参数数量: {len(model_state)}")
        print("模型层结构:")
        for key in list(model_state.keys())[:10]:  # 显示前10个层
            print(f"  {key}: {model_state[key].shape}")
    
    # 如果保存了配置信息
    if 'config' in ckpt:
        print("训练配置:")
        for key, value in ckpt['config'].items():
            print(f"  {key}: {value}")
```

## 2. 在Jupyter Notebook中运行sample.py生成图像

### 完整的JN代码（一行式命令）:

```python
import os
import torch

# ============ 路径设置 ============
vqvae_repo = "F:/ProjectsL/00PatternProject2025/00ImageCollection202505/VQVAEModel/vq-vae-2-pytorch"
checkpoint_dir = f"{vqvae_repo}/checkpoint"
gen_dir = "generated"  # 生成图像保存目录

# 创建输出目录
os.makedirs(gen_dir, exist_ok=True)

# ============ 检查可用的模型文件 ============
print("可用的VQ-VAE模型文件:")
vqvae_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('vqvae_') and f.endswith('.pt')]
vqvae_files.sort()
for f in vqvae_files[-5:]:  # 显示最后5个文件
    print(f"  {f}")

# 使用最新的VQ-VAE模型
latest_vqvae = f"{checkpoint_dir}/vqvae_050.pt"  # 你训练的第50个epoch
print(f"使用VQ-VAE模型: {latest_vqvae}")

# ============ 检查PixelSNAIL模型 ============
# 假设你还没有训练PixelSNAIL，先检查是否存在
pixelsnail_files = [f for f in os.listdir(checkpoint_dir) if 'pixelsnail' in f.lower() and f.endswith('.pt')]
if pixelsnail_files:
    pixelsnail_ckpt = f"{checkpoint_dir}/{pixelsnail_files[-1]}"
    print(f"找到PixelSNAIL模型: {pixelsnail_ckpt}")
else:
    print("警告: 未找到PixelSNAIL模型文件，你可能需要先训练PixelSNAIL")
    pixelsnail_ckpt = f"{checkpoint_dir}/pixelsnail_top.pt"  # 默认名称

# ============ 生成图像命令 ============
# 方法1: 如果你有完整的sample.py脚本
sample_cmd = f'!python "{vqvae_repo}/sample.py" --vqvae_ckpt "{latest_vqvae}" --pixelsnail_ckpt "{pixelsnail_ckpt}" --num_samples 10 --output "{gen_dir}" --size 256'
print("生成图像命令:")
print(sample_cmd)

# 方法2: 直接执行（如果PixelSNAIL已训练）
if os.path.exists(pixelsnail_ckpt):
    exec_cmd = f'python "{vqvae_repo}/sample.py" --vqvae_ckpt "{latest_vqvae}" --pixelsnail_ckpt "{pixelsnail_ckpt}" --num_samples 10 --output "{gen_dir}" --size 256'
    print(f"执行命令: {exec_cmd}")
    # !{exec_cmd}  # 取消注释来执行
else:
    print("请先训练PixelSNAIL模型")
```

### 如果你还需要训练PixelSNAIL:

```python
# ============ 训练PixelSNAIL ============
img_folder = "你的图像文件夹路径"  # 替换为你的图像文件夹路径

# 训练PixelSNAIL top level
pixelsnail_top_cmd = f'!python "{vqvae_repo}/train_pixelsnail.py" "{img_folder}" --vqvae_ckpt "{latest_vqvae}" --hierarchy top --epoch 30 --batch 32 --size 256'
print("PixelSNAIL Top训练命令:")
print(pixelsnail_top_cmd)

# 训练PixelSNAIL bottom level  
pixelsnail_bottom_cmd = f'!python "{vqvae_repo}/train_pixelsnail.py" "{img_folder}" --vqvae_ckpt "{latest_vqvae}" --hierarchy bottom --epoch 30 --batch 32 --size 256'
print("PixelSNAIL Bottom训练命令:")
print(pixelsnail_bottom_cmd)
```

### 最终生成图像的一行命令:

```python
# 当所有模型都训练完成后，生成图像
!python "{vqvae_repo}/sample.py" --vqvae_ckpt "{checkpoint_dir}/vqvae_050.pt" --pixelsnail_top "{checkpoint_dir}/pixelsnail_top.pt" --pixelsnail_bottom "{checkpoint_dir}/pixelsnail_bottom.pt" --num_samples 10 --output "{gen_dir}" --size 256
```

## 3. 常见问题解决

### 如果sample.py参数不同:
```python
# 查看sample.py的实际参数
!python "{vqvae_repo}/sample.py" --help
```

### 检查生成的图像:
```python
import matplotlib.pyplot as plt
from PIL import Image

# 查看生成的图像
gen_files = [f for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
if gen_files:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, filename in enumerate(gen_files[:10]):
        row, col = i // 5, i % 5
        img = Image.open(os.path.join(gen_dir, filename))
        axes[row, col].imshow(img)
        axes[row, col].set_title(filename)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("未找到生成的图像文件")
```

## 注意事项

1. **路径调整**: 将所有路径替换为你的实际路径
2. **模型检查**: 确保VQ-VAE和PixelSNAIL模型都已训练完成
3. **参数匹配**: sample.py的参数可能与你的实际脚本不同，使用`--help`查看正确参数
4. **内存管理**: 生成大量图像时注意GPU内存使用