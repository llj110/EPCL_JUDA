import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 假设模型输出预测
unlab_ema_out = [torch.randn(2, 4, 112, 112, 80), torch.randn(2, 4, 112, 112, 80),torch.randn(2, 4, 112, 112, 80),torch.randn(2, 4, 112, 112, 80)]  # [batch_size, channels, depth, height, width]
unlab_ema_out_pred = (unlab_ema_out[0] + unlab_ema_out[1] + unlab_ema_out[2] + unlab_ema_out[3])
unlab_ema_out_soft = torch.softmax(unlab_ema_out_pred, dim=1)

# 方差不确定性
unlab_ema_out_var = sum((x - unlab_ema_out_pred)**2 for x in unlab_ema_out_pred) / 4

# 熵不确定性
entro_uncertainty = -torch.sum(unlab_ema_out_soft * torch.log(unlab_ema_out_soft + 1e-16), dim=1)
entro_norm_uncertainty = entro_uncertainty / entro_uncertainty.sum(dim=(1, 2, 3), keepdim=True)
reliability_map1 = (1 - entro_norm_uncertainty) / np.prod(np.array(entro_norm_uncertainty.shape[-3:]))
# # 方差不确定性
mean_var_uncertainty = torch.mean(unlab_ema_out_var, dim=1)
var_norm_uncertainty = mean_var_uncertainty / mean_var_uncertainty.sum(dim=(1, 2, 3), keepdim=True)
# 联合不确定性
combined_uncertainty = torch.exp(-var_norm_uncertainty) * (1 - entro_norm_uncertainty)
reliability_map2 = combined_uncertainty / combined_uncertainty.sum(dim=(1, 2, 3), keepdim=True)

# 选择一个切片进行可视化，例如第40个深度切片
slice_idx = 30

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# 可视化基于熵的确定性
axes[0].imshow(reliability_map1[0, :, :, slice_idx].cpu().numpy(), cmap='hot', interpolation='nearest')
axes[0].set_title('Entropy', fontsize=16, fontweight='bold', fontname='Arial')
# 可视化可靠性map
axes[1].imshow(reliability_map2[0, :, :, slice_idx].cpu().numpy(), cmap='hot', interpolation='nearest')
axes[1].set_title('Joint Uncertainty Quantification', fontsize=16, fontweight='bold', fontname='Arial')
#
# axes[2].imshow(reliability_map[0, :, :, slice_idx].cpu().numpy(), cmap='hot', interpolation='nearest')
# axes[2].set_title('Reliability Map (Sample A)')
plt.tight_layout()
plt.savefig('uncertainty.pdf')
plt.show()
