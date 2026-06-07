import matplotlib.pyplot as plt
import numpy as np

# 实验模拟数据
k_values = [1, 2, 3, 4, 6, 8, 12, 16, 32]
gemm_latency = [1.0, 0.72, 0.71, 0.71, 0.71, 0.72, 0.73, 0.75, 0.78]
mha_latency = [1.0, 0.85, 0.75, 0.68, 0.67, 0.67, 0.69, 0.72, 0.76]

plt.figure(figsize=(7, 5), dpi=300)

# 设置学术配色
plt.plot(k_values, gemm_latency, marker='o', linestyle='-', color='#2F5597', linewidth=2, label='GEMM (Compute-intensive)')
plt.plot(k_values, mha_latency, marker='s', linestyle='--', color='#E67E22', linewidth=2, label='Attention (Memory-bound)')

# 标注拐点
plt.annotate('Optimal K=2', xy=(2, 0.72), xytext=(3, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
plt.annotate('Optimal K=4', xy=(4, 0.68), xytext=(6, 0.58),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

# 阴影区表示性能退化区
plt.axvspan(6, 32, color='gray', alpha=0.1, label='Resource Pressure Zone')

plt.xscale('log', base=2)
plt.xticks(k_values, k_values)
plt.xlabel('Prefetch Distance K (Tiles)', fontsize=12, fontweight='bold')
plt.ylabel('Normalized Execution Time', fontsize=12, fontweight='bold')
plt.title('Sensitivity Analysis of Prefetch Distance K', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, shadow=True)

# 移除冗余边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('sensitivity_k_final.png')
plt.show()