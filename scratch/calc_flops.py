import sys
sys.path.insert(0, ".")
import torch
from src.models.dscnn import DSCNN, DSCNNConfig

model = DSCNN()
model.eval()

# Convention:
# 1 MAC = 1 Multiply + 1 Accumulate = 2 FLOPs
# or report MACs directly.
# Input shape: [1, 1, 40, 47]

# Let's compute layer-by-layer MACs:
# 1. Stem Conv2d: in=1, out=28, kernel=3x3, stride=2, padding=1
# Output spatial: floor((40 + 2 - 3)/2) + 1 = 20, floor((47 + 2 - 3)/2) + 1 = 24 -> [20, 24]
# MACs = out_h * out_w * (in_c * k_h * k_w) * out_c
# = 20 * 24 * (1 * 3 * 3) * 28 = 480 * 9 * 28 = 120,960 MACs

# 2. For each of the 4 DS blocks:
#    a. Depthwise Conv2d: in=28, out=28, kernel=3x3, stride=1, padding=1 (groups=28)
#       Output spatial: [20, 24]
#       MACs = out_h * out_w * (1 * 3 * 3) * out_c = 20 * 24 * 9 * 28 = 120,960 MACs
#    b. Pointwise Conv2d: in=28, out=28, kernel=1x1, stride=1, padding=0
#       Output spatial: [20, 24]
#       MACs = out_h * out_w * (in_c * 1 * 1) * out_c = 20 * 24 * 28 * 28 = 376,320 MACs
#    Total MACs per DS block = 120,960 + 376,320 = 497,280 MACs

# 3. Global Average Pooling:
#    Reduces [20, 24, 28] -> [1, 1, 28]
#    Additions = 20 * 24 * 28 = 13,440 additions

# 4. FC Layer:
#    in=28, out=12
#    MACs = 28 * 12 = 336 MACs

stem_macs = 20 * 24 * (1 * 3 * 3) * 28
block_macs = (20 * 24 * 9 * 28) + (20 * 24 * 28 * 28)
total_block_macs = 4 * block_macs
fc_macs = 28 * 12
total_macs = stem_macs + total_block_macs + fc_macs
total_flops = 2 * total_macs

print(f"Stem Conv MACs: {stem_macs:,}")
print(f"Single DS Block MACs: {block_macs:,}")
print(f"4 DS Blocks MACs: {total_block_macs:,}")
print(f"FC MACs: {fc_macs:,}")
print(f"Total MACs: {total_macs:,} ({total_macs/1e6:.3f} MMACs)")
print(f"Total FLOPs (2x MACs): {total_flops:,} ({total_flops/1e6:.3f} MFLOPs)")
