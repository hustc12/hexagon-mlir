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
# 按延迟降序排列，以便从左到右展示
df = df.sort_values('ALPS-Prefetch (ms)', ascending=False)

# 2. 准备绘图数据
models = df['Model Name'].tolist()
lat_nn = df['Hexagon NN (ms)'].tolist()
lat_mlir = df['Hexagon-MLIR (ms)'].tolist()
lat_vanilla = df['Hexagon-MLIR (Vanillar Prefetch) (ms)'].tolist()
lat_alps = df['ALPS-Prefetch (ms)'].tolist()

# 准备加速比数据
sp_nn = df['Speedup (Hexagon NN)'].tolist()
sp_mlir = df['Speedup (Hexagon-MLIR)'].tolist()
sp_vanilla = df['Speedup (Hexagon-MLIR Vanillar Prefetch)'].tolist()

x = np.arange(len(models))
width = 0.2  # 柱子宽度

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

# 配色方案 (与之前保持一致)
color_nn = '#d9d9d9'      # 浅灰
color_mlir = '#7facd6'    # 中蓝
color_vanilla = '#2f5597' # 深蓝
color_alps = '#ed7d31'    # 亮橙 (ALPS)

# 绘制四组竖向柱子
rects1 = ax.bar(x - 1.5*width, lat_nn, width, label='Hexagon NN', color=color_nn, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x - 0.5*width, lat_mlir, width, label='Hexagon-MLIR', color=color_mlir, edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + 0.5*width, lat_vanilla, width, label='Vanilla Prefetch', color=color_vanilla, edgecolor='black', linewidth=0.5)
rects4 = ax.bar(x + 1.5*width, lat_alps, width, label='ALPS-Prefetch (Ours)', color=color_alps, edgecolor='black', linewidth=0.8)

# 4. 细节调整
ax.set_ylabel('Inference Latency (ms) [Log Scale]', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11) # 旋转模型名称防止重叠
ax.set_title('End-to-End Latency & Per-Baseline Speedup of ALPS', fontsize=15, fontweight='bold', pad=30)
ax.set_yscale('log') # Y轴使用对数坐标

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.grid(True, which="both", linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 5. 在柱子上方添加标注
def add_vertical_labels(rects, speedups, text_color, is_alps=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if is_alps:
            label = f'{height:.1f}ms'
            txt_color = '#c45911'
        else:
            label = f'{speedups[i]}x'
            txt_color = text_color
        
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 向上偏移 5 点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color=txt_color, rotation=0)

add_vertical_labels(rects1, sp_nn, '#767171')
add_vertical_labels(rects2, sp_mlir, '#2e75b6')
add_vertical_labels(rects3, sp_vanilla, '#1f4e79')
add_vertical_labels(rects4, None, is_alps=True, text_color='#c45911')

# 6. 图例设置
ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)

plt.tight_layout()
plt.savefig('alps_vertical_latency_comparison.png')
plt.show()