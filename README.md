# Image Matching and 3D Reconstruction

对杂乱图片数据集自动进行**多场景 3D 重建**的完整流水线。

## 整体架构

```
原始图片数据集
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 1 · 多模型特征提取                      │
│  DINOv2 / MAST3R-ASMK / MAST3R-SPoC / ISC   │
│  输出：每张图的全局/局部特征向量                   │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 2 · 图像检索 & Shortlist 生成           │
│  多模型特征加权融合 → 相似度矩阵 → top‑k 候选对    │
│  输出：shortlist（候选图像对池）                  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 3 · Pre‑clustering（关键插入点）         │
│  Coarse MAST3R 快速几何验证 → 匹配图 → 连通分量   │
│  输出：若干场景簇（每个簇 = 同一场景的图片集合）      │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 4 · 关键点检测                          │
│  SuperPoint（256维） / ALIKE（128维）           │
│  输出：每张图的关键点坐标 + 局部描述子              │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 5 · MAST3R 稠密匹配                     │
│  输入：RGB 图像对 → 输出：稠密 3D 点图 + 匹配对应  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  Phase 6 · COLMAP 3D 重建                    │
│  每个连通分量（场景）独立进行稀疏/稠密重建          │
│  输出：各场景的 3D 点云 & 相机位姿               │
└─────────────────────────────────────────────┘
```

## 关键设计原则

1. **Shortlist 是候选池，不是最终场景归属**。图像检索给出的是基于外观相似度的候选对，真正的场景归属由 Phase 3 几何验证决定。
2. **COLMAP 不会主动找图**。必须先通过连通分量确定每个场景的图片集合，再将集合送入 COLMAP。COLMAP 只使用显式输入给它的图像对。
3. **场景划分标准**：几何验证失败（如无法计算有效基础矩阵、内点数量不足）即判定为不同场景。
4. **多模型融合是加权求和，而非取交集**。保留各模型的互补信息，提高召回率。

---

## 代码文件说明（`code/` 目录）

### Phase 1 · 特征提取模型训练

四个特征提取模型各自独立训练，输出全局特征向量用于后续检索。

#### `dino.ipynb` — DINOv2 自监督训练

| 项目 | 说明 |
|------|------|
| **基础模型** | `dinov2-base`（HuggingFace Transformers） |
| **训练方式** | DINO 自蒸馏（student-teacher），EMA 动量更新 teacher |
| **数据增强** | Multi-crop：2 个 global crops (224×224) + 6 个 local crops (96×96) |
| **损失函数** | Cross-entropy between softened teacher & student distributions，含 center-momentum 去偏 |
| **投影维度** | 256 |
| **输出** | 全局特征向量（CLS token 经投影头） |
| **模型保存** | 每 epoch 保存 `dino_epoch_{N}.pth`（含 student/teacher state_dict + optimizer + center） |
| **内容要求** | ① 数据加载（多尺度裁剪增强）；② Student-Teacher 模型构建；③ EMA 更新 & Center 更新；④ 训练循环 + checkpoint 保存；⑤ 全局特征提取函数 |

---

#### `mast3r_asmk.ipynb` — MAST3R-ASMK 特征提取训练

| 项目 | 说明 |
|------|------|
| **基础模型** | MAST3R backbone（croco 预训练权重） |
| **训练方式** | ASMK（Aggregated Selective Match Kernel）聚合：提取 MAST3R 局部特征 → 选择性聚合 → 全局 ASMK 描述子 |
| **数据流程** | 图片 → MAST3R encoder → 局部特征图 → 多尺度池化 → 选择性匹配核聚合 |
| **损失函数** | 对比损失（contrastive loss）或 triplet loss，拉近同一场景图片、推远不同场景 |
| **输出** | 全局 ASMK 描述子（可做余弦相似度检索） |
| **模型保存** | 每 epoch / 最佳轮次保存完整模型权重 + 聚合参数 |
| **内容要求** | ① MAST3R backbone 加载与冻结/微调策略；② 局部特征提取与多尺度处理；③ ASMK 聚合模块（选择性匹配核）；④ 对比学习训练循环；⑤ checkpoint 保存；⑥ 全局描述子推理函数 |

---

#### `mast3r_spoc.ipynb` — MAST3R-SPoC 特征提取训练

| 项目 | 说明 |
|------|------|
| **基础模型** | MAST3R backbone（croco 预训练权重） |
| **训练方式** | SPoC（Sum-Pooled Convolutional features）：对 MAST3R 特征图空间维度求和池化得到全局描述子 |
| **与 ASMK 区别** | SPoC 更简单直接——全局求和池化，无选择性匹配核；速度更快但判别力略低于 ASMK |
| **损失函数** | 对比损失（contrastive loss）或 triplet loss |
| **输出** | 全局 SPoC 描述子 |
| **模型保存** | 每 epoch / 最佳轮次保存完整模型权重 |
| **内容要求** | ① MAST3R backbone 加载；② SPoC 池化层（空间求和 + L2 归一化）；③ 对比学习训练循环；④ checkpoint 保存；⑤ 全局描述子推理函数 |

---

#### `isc.ipynb` — ISC（Image Similarity Challenge）特征提取训练

| 项目 | 说明 |
|------|------|
| **基础模型** | 可选 ResNet-50 / EfficientNet 等标准 backbone |
| **训练方式** | 度量学习：使用 ArcFace / CosFace / SubCenter ArcFace 等边界损失训练全局特征 |
| **数据增强** | 标准增强（随机裁剪、翻转、颜色抖动） + 可能的增强策略（如 RandAugment） |
| **损失函数** | ArcFace margin loss（或 SubCenter ArcFace），将特征映射到超球面 |
| **输出** | L2 归一化的全局特征向量 |
| **模型保存** | 最佳轮次保存 backbone + margin head 权重 |
| **内容要求** | ① 数据加载与增强管道；② Backbone + ArcFace/SubCenter ArcFace head 构建；③ 度量学习训练循环（含 margin 调度策略）；④ checkpoint 保存（最佳 mAP/recall 轮次）；⑤ 全局特征推理函数 |

---

### Phase 2 & 4 · 图像检索 + 关键点检测

#### `feature_retrieval_shortlist.ipynb` — 多模型融合检索 & 关键点提取

| 项目 | 说明 |
|------|------|
| **输入** | 原始图片数据集 + 四个已训练模型的 checkpoint |
| **核心流程** | ① 加载四个模型各自的最佳 checkpoint；② 分别提取全局特征；③ 计算余弦相似度矩阵；④ 加权融合四个相似度矩阵（非取交集）；⑤ 每张图取 top‑k 最相似候选 → **shortlist**；⑥ 对 shortlist 中每张图运行 SuperPoint 和 ALIKE 提取关键点 + 描述子 |
| **关键点模型对比** | **SuperPoint**：先检测后描述，精度高，输出 256 维描述子；**ALIKE**：端到端检测+描述，速度更快，输出 128 维描述子 |
| **输出** | shortlist（候选图像对列表） + 每张图的 SuperPoint 关键点/描述子 + ALIKE 关键点/描述子 |
| **内容要求** | ① 四个特征模型的加载代码（各自加载最优 checkpoint 到 eval 模式）；② 批量全局特征提取；③ 余弦相似度矩阵计算（N×N）；④ 加权融合策略（权重可配置，默认均匀或基于验证集调优）；⑤ top‑k 筛选生成 shortlist；⑥ SuperPoint 模型加载与关键点批量提取；⑦ ALIKE 模型加载与关键点批量提取；⑧ 关键点可视化（随机抽样验证）；⑨ 结果保存（shortlist 及关键点数据） |

---

### Phase 3 & 5 · 几何验证 & 稠密匹配

#### `pre_clustering.ipynb` — Coarse MAST3R 几何验证与场景预聚类

| 项目 | 说明 |
|------|------|
| **输入** | shortlist（候选图像对） + 原始图片 |
| **核心流程** | ① 对 shortlist 中每对图像运行 coarse MAST3R 推理；② 计算几何一致性分数（内点数量 / 基础矩阵验证）；③ 阈值筛选：内点数 > τ_inlier 且几何一致性 > τ_geo → 保留边；④ 构建匹配图（节点=图片，边=通过几何验证的对）；⑤ 求连通分量 → 每个分量 = 一个场景簇；⑥ 孤立图片处理策略 |
| **阈值建议** | τ_inlier ≥ 20（内点数量下界）；τ_geo ≥ 0.3（几何一致性分数下界）；具体数值需在验证集上调优 |
| **输出** | 场景聚类结果（连通分量列表），每个分量包含：图片路径列表 + 内部匹配对列表 |
| **内容要求** | ① Coarse MAST3R 模型加载；② 批量图像对推理；③ 几何一致性计算（基础矩阵估计 + 内点计数）；④ 匹配图构建（邻接矩阵 / 边列表）；⑤ 连通分量算法（BFS/DFS/Union-Find）；⑥ 孤立图片处理（标记为 "未匹配" 或单独成场景）；⑦ 聚类结果可视化和统计；⑧ 结果保存 |

---

#### `mast3r_matching.ipynb` — MAST3R 稠密匹配

| 项目 | 说明 |
|------|------|
| **输入** | 图像对（来自连通分量内部匹配对） + 对应的 SuperPoint/ALIKE 关键点坐标 |
| **核心流程** | ① 加载 MAST3R 模型（精细模式）；② 输入 RGB 图像对 → 前向推理；③ 输出：稠密 3D 点图（pixel-aligned）、特征描述子（dense descriptors）、置信度图（confidence map）；④ 基于关键点坐标提取对应位置的局部描述子；⑤ 构建双边匹配关系（mutual nearest neighbors）；⑥ 输出像素级匹配对应 |
| **输出** | 每对图像的稠密 3D 点图 + 特征描述子 + 置信度图 + 匹配对（像素级对应） |
| **内容要求** | ① 精细 MAST3R 模型加载（含预训练权重）；② 图像对预处理（resize / 归一化）；③ 前向推理得到点图、描述子、置信度；④ 关键点位置描述子提取；⑤ 双向最近邻匹配（mutual nearest neighbor）；⑥ 匹配置信度过滤；⑦ 点图可视化（如深度图 / 点云预览）；⑧ 匹配结果保存 |

---

### Phase 6 · 3D 重建

#### `colmap_reconstruction.ipynb` — COLMAP 3D 重建流水线

| 项目 | 说明 |
|------|------|
| **输入** | 每个场景簇（连通分量）的图片集合 + MAST3R 匹配结果 |
| **核心流程** | ① 遍历每个连通分量；② 为每个分量创建独立工作目录；③ 将 MAST3R 匹配结果转换为 COLMAP 可接受的匹配格式（database 导入 / match 文件）；④ 运行 COLMAP sparse reconstruction（mapper）；⑤ 运行 COLMAP dense reconstruction（image_undistorter + patch_match_stereo + stereo_fusion） 或使用 MAST3R 稠密点云替代；⑥ 导出 3D 点云（PLY）和相机位姿 |
| **注意事项** | COLMAP 只使用显式给它的图像对——来自同一连通分量内的匹配对。不会自动跨分量匹配。 |
| **输出** | 每个场景的：稀疏点云 + 稠密点云（.ply）+ 相机内外参数 + 稀疏重建可视化 |
| **内容要求** | ① COLMAP 环境准备（pycolmap 或 subprocess 调用）；② 每个场景分量创建独立目录；③ 图片复制 / 软链接到场景目录；④ 匹配信息导入 COLMAP database（或直接写 match 文件）；⑤ Sparse reconstruction（特征提取 + 匹配 + 增量式 SfM）；⑥ Dense reconstruction（可选，输出稠密点云）；⑦ 点云导出与可视化（Open3D / pycolmap）；⑧ 所有场景的批量处理循环；⑨ 重建质量报告（注册图片数 / 稀疏点数 / 重投影误差） |

---

## 项目目录结构

```
Image-Matching-and-3D-Reconstruction/
├── README.md
├── code/
│   ├── dino.ipynb                       # DINOv2 自监督训练
│   ├── mast3r_asmk.ipynb                # MAST3R-ASMK 特征训练
│   ├── mast3r_spoc.ipynb                # MAST3R-SPoC 特征训练
│   ├── isc.ipynb                        # ISC 度量学习训练
│   ├── feature_retrieval_shortlist.ipynb # 多模型融合检索 + 关键点提取
│   ├── pre_clustering.ipynb             # Coarse MAST3R 几何验证 + 场景聚类
│   ├── mast3r_matching.ipynb            # MAST3R 稠密匹配
│   └── colmap_reconstruction.ipynb      # COLMAP 3D 重建
├── models/                              # 预训练基础模型权重
├── checkpoints/                         # 训练 checkpoint 保存目录
│   ├── dinov2/
│   ├── mast3r_asmk/
│   ├── mast3r_spoc/
│   └── isc/
└── image-matching-challenge-2025/       # 数据集
    └── train/
```

## 待明确 / 可调参数

| 问题 | 说明 |
|------|------|
| **Pre‑clustering 几何阈值** | 内点数量 τ_inlier 和几何一致性 τ_geo 的具体取值，建议在验证集上做 grid search |
| **多模型融合权重** | DINOv2 / ASMK / SPoC / ISC 四个模型的加权系数，默认可均匀（0.25），最优需在验证集上调优 |
| **孤立图片处理** | 无法与任何其他图片通过几何验证的图片，可标记为 "未归类" 或单独作为单图场景 |
| **Shortlist 的 k 值** | 每张图保留多少个候选邻居；k 太小会漏匹配，k 太大增加几何验证开销 |

## 环境依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0 + CUDA
- HuggingFace Transformers + Timm
- MAST3R (croco pretrained)
- COLMAP / pycolmap
- Open3D (点云可视化)
- Kornia / OpenCV (几何验证)
