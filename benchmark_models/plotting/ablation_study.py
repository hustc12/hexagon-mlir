import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载并清洗数据
df = pd.read_csv('ALPS_Prefetcher_Data.csv').dropna(subset=['Model Name'])

# 2. 挑选 4 个代表性模型
target_models = ['Falcon-RW-1B', 'google/vit-base-patch16-224', 'Mamba-130M', 'Qwen2.5-0.5B']
df_ab = df[df['Model Name'].isin(target_models)].copy()
df_ab['Model Name'] = df_ab['Model Name'].replace({'google/vit-base-patch16-224': 'ViT-Base', 'Qwen2.5-0.5B': 'Qwen2.5'})

# 按 Baseline 延迟降序排列
df_ab = df_ab.sort_values('Hexagon-MLIR (ms)', ascending=False)

# 3. 映射 4 个消融阶段
lat_base = df_ab['Hexagon-MLIR (ms)'].tolist()             # (1) Baseline
lat_vanilla = df_ab['Hexagon-MLIR (Vanillar Prefetch) (ms)'].tolist() # (2) Vanilla
lat_vdae = df_ab['V-DAE Only (ms)'].tolist()               # (3) V-DAE only
lat_alps = df_ab['ALPS-Prefetch (ms)'].tolist()            # (4) ALPS (Ours)

models = df_ab['Model Name'].tolist()
x = np.arange(len(models))
width = 0.22 # 柱子宽度

# 4. 开始绘图
fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
colors = ['#BDC3C7', '#7FACD6', '#2F5597', '#E67E22'] # 灰, 浅蓝, 深蓝, 橙

rects1 = ax.bar(x - 1.5*width, lat_base, width, label='Baseline (No Optimization)', color=colors[0], edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x - 0.5*width, lat_vanilla, width, label='Vanilla Prefetch', color=colors[1], edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + 0.5*width, lat_vdae, width, label='V-DAE only', color=colors[2], edgecolor='black', linewidth=0.5)
rects4 = ax.bar(x + 1.5*width, lat_alps, width, label='ALPS (Ours)', color=colors[3], edgecolor='black', linewidth=0.8)

# 5. 样式调整
ax.set_ylabel('Latency (ms) [Log Scale]', fontsize=24, fontweight='bold')
ax.tick_params(axis='y', labelsize=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='center', fontsize=22, fontweight='bold')
ax.set_title('Ablation Study: 4-Stage Performance Breakdown', fontsize=28, fontweight='bold', pad=30)
ax.set_yscale('log')
ax.set_ylim(bottom=100, top=10**4.8) # 预留更多空间给旋转的标注

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.grid(True, which="both", linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# 6. 数值标注
def add_labels(rects, text_color):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6), 
                    textcoords="offset points",
                    ha='left', va='bottom', fontsize=14, fontweight='bold', color=text_color, rotation=45)

add_labels(rects1, '#555555')
add_labels(rects2, '#2E75B6')
add_labels(rects3, '#1F4E79')
add_labels(rects4, '#C45911')

# 7. 说明框置于右上方
ax.legend(loc='upper right', frameon=True, fontsize=20, shadow=True)

plt.tight_layout()
plt.savefig('alps_ablation.pdf', format='pdf', bbox_inches='tight')
# plt.show()