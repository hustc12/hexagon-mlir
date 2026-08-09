import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载并清洗数据
df = pd.read_csv('ALPS_Prefetcher_Data.csv').dropna(subset=['Model Name'])

# 简写模型名称
model_map = {
    'google/vit-base-patch16-224': 'ViT-Base',
    'Swin Transformer / swin-tiny-patch4-window7-224': 'Swin-Tiny',
    'TinyLlama-1.1B-Chat-v1.0': 'TinyLlama-1.1B',
    'SRGAN / Real-ESRGAN': 'SRGAN',
    'GraphSAGE (GNN)': 'GraphSAGE'
}
df['Model Name'] = df['Model Name'].replace(model_map)
df = df.sort_values('ALPS-Prefetch (ms)', ascending=False)

# 2. 准备绘图数据 (Latency)
models = df['Model Name'].tolist()
lat_nn = df['Hexagon NN (ms)'].tolist()
lat_mlir = df['Hexagon-MLIR (ms)'].tolist()
lat_vanilla = df['Hexagon-MLIR (Vanillar Prefetch) (ms)'].tolist()
lat_alps = df['ALPS-Prefetch (ms)'].tolist()

# 准备加速比数据 (ALPS 相比于各基准的倍数)
sp_nn = df['Speedup (Hexagon NN)'].tolist()
sp_mlir = df['Speedup (Hexagon-MLIR)'].tolist()
sp_vanilla = df['Speedup (Hexagon-MLIR Vanillar Prefetch)'].tolist()

y = np.arange(len(models))
height = 0.2 

# 3. 开始绘图 (还原截图风格并增强标注)
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# 采用专业对比色
color_nn = '#d9d9d9'      # 浅灰 (Hexagon NN)
color_mlir = '#7facd6'    # 中蓝 (Hexagon-MLIR)
color_vanilla = '#2f5597' # 深蓝 (Vanilla Prefetch)
color_alps = '#ed7d31'    # 橙色 (ALPS-Prefetch)

# 绘制四组水平柱子
rects1 = ax.barh(y + 1.5*height, lat_nn, height, label='Hexagon NN', color=color_nn, edgecolor='black', linewidth=0.5)
rects2 = ax.barh(y + 0.5*height, lat_mlir, height, label='Hexagon-MLIR', color=color_mlir, edgecolor='black', linewidth=0.5)
rects3 = ax.barh(y - 0.5*height, lat_vanilla, height, label='Vanilla Prefetch', color=color_vanilla, edgecolor='black', linewidth=0.5)
rects4 = ax.barh(y - 1.5*height, lat_alps, height, label='ALPS-Prefetch (Ours)', color=color_alps, edgecolor='black', linewidth=0.8)

# 4. 细节调整
ax.set_xlabel('Inference Latency (ms) [Log Scale]', fontsize=12, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=11)
ax.set_title('End-to-End Latency & Per-Baseline Speedup of ALPS', fontsize=15, fontweight='bold', pad=25)
ax.set_xscale('log')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.grid(True, which="both", linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 5. 在所有柱子上标注 ALPS 实现的加速比
def add_speedup_labels(rects, speedups, text_color):
    for i, rect in enumerate(rects):
        width = rect.get_width()
        # 标注格式为：X.Xx (代表 ALPS 比该柱子快多少倍)
        ax.annotate(f'{speedups[i]}x',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), 
                    textcoords="offset points",
                    ha='left', va='center', fontsize=8.5, fontweight='bold', color=text_color)

# 为各基准柱子添加 ALPS 对其的加速比标注
add_speedup_labels(rects1, sp_nn, '#767171')
add_speedup_labels(rects2, sp_mlir, '#2e75b6')
add_speedup_labels(rects3, sp_vanilla, '#1f4e79')

# 为 ALPS 柱子添加绝对延迟标注
for i, rect in enumerate(rects4):
    ax.annotate(f'{lat_alps[i]:.1f}ms',
                xy=(lat_alps[i], rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=8.5, fontweight='bold', color='#c45911')

# 6. 图例设置
ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True, bbox_to_anchor=(1, 1.02))

plt.tight_layout()
plt.savefig('alps_full_speedup_labels.png')
plt.show()