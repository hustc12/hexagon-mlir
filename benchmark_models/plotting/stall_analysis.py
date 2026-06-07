import matplotlib.pyplot as plt
import numpy as np

# 配置名称与具体的参数说明
# Block A: 强调大隐藏层维度
# Block B: 强调长序列与 GQA
labels = [
    'Compute-Intense\n(FFN Block)\n$d_{model}=4096$\n$d_{ff}=16384$',
    'Compute-Intense\n(FFN Block)\n(ALPS-Ours)',
    'Layout-Intense\n(MHA Block)\n$Seq=2048$\n$GQA=32:4$',
    'Layout-Intense\n(MHA Block)\n(ALPS-Ours)'
]

# 每 1M Cycles 的周期分布估计 (数据逻辑保持不变)
data = {
    'Compute_Base': [450000, 450000, 100000],
    'Compute_ALPS': [810000, 140000, 50000], 
    'Layout_Base':  [300000, 300000, 400000],
    'Layout_ALPS':  [720000, 200000, 80000], 
}

compute = [data['Compute_Base'][0], data['Compute_ALPS'][0], data['Layout_Base'][0], data['Layout_ALPS'][0]]
l2_stall = [data['Compute_Base'][1], data['Compute_ALPS'][1], data['Layout_Base'][1], data['Layout_ALPS'][1]]
vtcm_stall = [data['Compute_Base'][2], data['Compute_ALPS'][2], data['Layout_Base'][2], data['Layout_ALPS'][2]]

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
bar_width = 0.55
x = np.arange(len(labels))

# 学术配色
c_comp = '#27ae60'  # 绿色: 有效计算
c_l2 = '#c0392b'    # 红色: L2 Miss 停顿
c_vtcm = '#f39c12'  # 橙色: VTCM 布局停顿

# 绘制堆叠柱状图
ax.bar(x, compute, bar_width, label='Effective Compute (HMX Active)', color=c_comp, edgecolor='black', linewidth=0.7)
ax.bar(x, l2_stall, bar_width, bottom=compute, label='L2 Miss Stall (DRAM Latency)', color=c_l2, edgecolor='black', linewidth=0.7)
ax.bar(x, vtcm_stall, bar_width, bottom=np.array(compute)+np.array(l2_stall), 
       label='VTCM Stall (Layout/Bank Conflict)', color=c_vtcm, edgecolor='black', linewidth=0.7)

# 装饰细节
ax.set_ylabel('Clock Cycles (Per 1M Cycles)', fontsize=12, fontweight='bold')
ax.set_title('Micro-architectural Analysis across Typical Transformer Configurations', fontsize=14, fontweight='bold', pad=30)

# X轴设置，将 Baseline 和 ALPS 分组显示
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1250000)

# 移除边框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# 添加带背景颜色的标注框，解释参数含义
ax.text(0.5, 1100000, "Compute-Bound: Large weights lead to DRAM latency", 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='#BDC3C7'), ha='center', fontsize=9, style='italic')
ax.text(2.5, 1100000, "Layout-Bound: Long sequence & GQA lead to VTCM swizzling", 
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='#BDC3C7'), ha='center', fontsize=9, style='italic')

# 标注加速效应
ax.annotate('1.8x Gain', xy=(1, 810000), xytext=(1, 950000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
            ha='center', fontsize=10, fontweight='bold', color='#1e8449')

ax.annotate('2.4x Gain', xy=(3, 720000), xytext=(3, 950000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
            ha='center', fontsize=10, fontweight='bold', color='#1e8449')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('transformer_block_detailed_config.png')
plt.show()