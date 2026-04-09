# 工业级推荐系统深度学习指南

> **目标读者**：有Python基础，不熟悉推荐系统，想通过这个项目学透工业级推荐系统的开发者。
> **对标岗位**：TikTok ML Engineer Intern、京东ML实习生、快手推荐算法实习生。

---

## 全文重点索引

| 标记 | 含义 |
|------|------|
| 🔴 **【面试必考】** | TikTok/京东面试高频考点，必须掌握 |
| 🟠 **【重要原理】** | 理解项目架构必须掌握的核心原理 |
| 🟡 **【加分项】** | 掌握后面试中能体现深度的知识点 |
| 🟢 **【了解即可】** | 背景知识，时间不够可以跳过 |
| ⚠️ **【踩坑警告】** | 容易出错的地方，必须注意 |
| 💡 **【面试表达技巧】** | 如何在面试中表达这个知识点能获得更好印象 |

---

## 目录

1. [环境配置](#1-环境配置)
2. [项目概述与系统架构](#2-项目概述与系统架构)
3. [数据层：特征工程与数据集](#3-数据层特征工程与数据集)
4. [召回模块：双塔模型 + Faiss](#4-召回模块双塔模型--faiss)  *(Iteration 2 完成后填写)*
5. [排序模块：DeepFM + DIN](#5-排序模块deepfm--din)  *(Iteration 3 完成后填写)*
6. [实验与分析：消融实验设计](#6-实验与分析消融实验设计)  *(Iteration 4 完成后填写)*
7. [完整Pipeline与Demo](#7-完整pipeline与demo)  *(Iteration 5 完成后填写)*
8. [面试准备：高频题汇总与Pitch模板](#8-面试准备高频题汇总与pitch模板)  *(Iteration 5 完成后填写)*
9. [延伸阅读：参考论文清单](#9-延伸阅读参考论文清单)  *(Iteration 5 完成后填写)*

---

## 1. 环境配置

### 1.1 这个章节解决什么问题

在你跑通任何代码之前，需要确保：
1. Python版本和所有依赖库兼容
2. GPU被PyTorch正确识别（RTX 5060是最新Blackwell架构，有特殊坑）
3. 所有工具链安装正确

> ⚠️ **【踩坑警告】**
> RTX 50系（Blackwell架构）的 Compute Capability 是 12.0（sm_120），这是2025年发布的全新架构。PyTorch 2.5及以下完全不支持；PyTorch 2.6 开始添加初步支持，但**必须使用 cu128 编译版本**。如果你用 `pip install torch` 默认安装，大概率安装的是 cu121 版本，`torch.cuda.is_available()` 会返回 `False`。

### 1.2 实际检测结果

本项目开发环境的实际检测数据：

| 项目 | 检测结果 |
|------|---------|
| 操作系统 | Windows 11 Pro 10.0.26200 |
| GPU型号 | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU架构 | Blackwell (sm_120, Compute Capability 12.0) |
| 显存 | 8151 MiB (~8 GB) |
| NVIDIA驱动版本 | 591.74 |
| CUDA Runtime版本 | 12.8 |
| CUDA Driver支持上限 | 13.1 |
| Python版本 | 3.10.20 (Anaconda) |
| PyTorch版本 | 2.11.0+cu128 |
| conda环境位置 | D:\Anaconda3\envs\recsys |

> 🟠 **【重要原理】**
> **Compute Capability（计算能力）**是NVIDIA对GPU架构世代的编号。常见对应关系：
> - sm_86：RTX 30系（Ampere），PyTorch早期版本支持
> - sm_89：RTX 40系（Ada Lovelace），PyTorch 2.0+ 支持
> - sm_120：RTX 50系（Blackwell），PyTorch 2.6+ cu128 支持
>
> 选择PyTorch版本时，必须确保其编译的CUDA版本不低于你GPU需要的CUDA版本。

### 1.3 为什么选择 PyTorch 2.11.0+cu128

**决策过程**：

1. 检测到 Compute Capability 12.0 → 确认是 Blackwell 架构
2. 确认 PyTorch 2.6+ 才开始支持 sm_120
3. 需要 cu128（CUDA 12.8）编译版本，不能用 cu121 或 cu117
4. 安装命令：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
5. 验证：`torch.cuda.is_available()` 返回 `True` ✓

> ⚠️ **【踩坑警告】**
> **Windows上faiss-gpu不可用**。`faiss-gpu` 的 pip 包只提供 Linux 版本。
> 解决方案：使用 `faiss-cpu`（本项目采用），或在Linux/WSL2环境下使用faiss-gpu。
> faiss-cpu 对于10万级别以内的候选集性能完全够用；超过百万规模才需要GPU加速的faiss。

### 1.4 Python 3.10 与各依赖库兼容性验证

| 库 | 要求Python版本 | 实际安装版本 | 是否兼容 |
|----|---------------|-------------|---------|
| torch | ≥ 3.9 | 2.11.0+cu128 | ✓ |
| numpy | ≥ 3.9 | 2.2.6 | ✓ |
| pandas | ≥ 3.9 | 2.3.3 | ✓ |
| scikit-learn | ≥ 3.9 | 1.7.2 | ✓ |
| gradio | ≥ 3.10 | 6.11.0 | ✓ |
| faiss-cpu | ≥ 3.8 | 1.13.2 | ✓ |
| tensorboard | ≥ 3.9 | 2.20.0 | ✓ |

**结论：Python 3.10 与所有依赖库完全兼容，无需重建conda环境。**

### 1.5 安装步骤（可复现）

```bash
# 1. 激活conda环境
conda activate recsys  # 或使用完整路径

# 2. 安装 PyTorch（关键：必须用 cu128，不能用默认版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其余依赖
pip install pandas scikit-learn tqdm tensorboard
pip install gradio matplotlib seaborn pyyaml
pip install faiss-cpu  # Windows上faiss-gpu不可用
pip install jupyter ipykernel

# 4. 验证安装
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU mode\"}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 1.6 代码层面GPU规范

所有训练代码必须遵守以下模式：

```python
import torch

# 1. 设备检测（每个训练脚本开头必须有）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("[WARNING] CUDA not available, falling back to CPU. Training will be slower.")

# 2. 模型和数据都要 .to(device)
model = MyModel().to(device)
inputs = inputs.to(device)
labels = labels.to(device)

# 3. 永远不要hardcode 'cuda'，永远使用变量device
# 错误写法: model.cuda()
# 正确写法: model.to(device)
```

> 🔴 **【面试必考】**
> **Q: 为什么不直接写 `model.cuda()` 而要用 `torch.device`？**
> A: `model.cuda()` 在没有GPU的环境会直接报错，导致代码不可复现；而 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 可以自动fallback到CPU，代码在任何机器上都能运行。工业界CI/CD环境有时没有GPU，这个写法保证测试和部署的一致性。

### 1.7 学习优先级总结

| 知识点 | 优先级 | 备注 |
|--------|--------|------|
| GPU Compute Capability概念 | 🟠 | 面试时可能被问到"你的GPU是什么架构" |
| PyTorch CUDA版本选择逻辑 | 🟠 | 配置新环境必备 |
| faiss-gpu Windows不可用 | ⚠️ | 直接踩坑，需要知道解决方案 |
| device自动检测写法 | 🔴 | 所有代码必须遵守，面试也会考到 |
| conda环境管理 | 🟢 | 假设读者已熟悉 |

---

## 2. 项目概述与系统架构

### 2.1 这个项目解决什么问题

> 🔴 **【面试必考】**
> **Q: 为什么推荐系统需要分召回和排序两个阶段？一个大模型统一打分不行吗？**
>
> A: 以TikTok为例，视频库有数十亿条内容，用户每次打开App需要在**200ms内**完成推荐。如果用精排模型（DeepFM/DIN，参数量百万级）对所有视频打分，即使每个视频只需1ms，也需要10^6秒，完全不可行。
>
> 因此工业界标准方案是**漏斗架构**：
> - **召回层**（Retrieval）：从亿级物品快速筛出几百个候选，追求**覆盖率**，要求极低延迟（<50ms）
> - **排序层**（Ranking）：对百级候选精细打分，追求**准确率**，可以用复杂特征
> - **重排层**（Re-ranking）：考虑多样性、实时上下文，本项目不实现

### 2.2 本项目的系统架构

```
用户请求
    │
    ▼
特征提取
    │
    ├──[召回阶段]──────────────────────────────────┐
    │  ┌──────────────┐    ┌──────────────────┐   │
    │  │  User Tower  │    │  Item Tower      │   │
    │  │  (用户双塔)   │    │  (物品双塔,离线)  │   │
    │  └──────┬───────┘    └────────┬─────────┘   │
    │         │ user_embedding      │ item embeddings │
    │         └─────────────────────┘             │
    │                    │                         │
    │              Faiss Index Search              │
    │              top-100 candidates              │
    │                    │                         │
    └────────────────────┼─────────────────────────┘
                         │
                    [排序阶段]
                         │
               ┌─────────────────┐
               │  DIN / DeepFM   │
               │  (精排打分)      │
               └────────┬────────┘
                         │
                    Top-10 推荐
```

### 2.3 各模块文件映射

| 模块 | 核心文件 | 作用 |
|------|---------|------|
| 数据层 | `src/data/download_data.py` | 生成/下载KuaiRec数据 |
| 数据层 | `src/data/feature_engineering.py` | 特征计算与数据集划分 |
| 数据层 | `src/data/dataset.py` | PyTorch Dataset包装 |
| 召回 | `src/models/two_tower.py` | 双塔模型定义 |
| 召回 | `src/retrieval/faiss_index.py` | Faiss向量检索 |
| 召回训练 | `src/training/train_retrieval.py` | 双塔训练脚本 |
| 排序 | `src/models/deepfm.py` | DeepFM模型 |
| 排序 | `src/models/din.py` | DIN模型 |
| 排序训练 | `src/training/train_ranking.py` | 精排训练脚本 |
| 评估 | `src/evaluation/metrics.py` | Recall@K, AUC, GAUC |
| 推理 | `main.py` | 完整端到端推理 |

> 💡 **【面试表达技巧】**
> 介绍项目架构时，先说"为什么分两阶段"（时间复杂度约束），再说"每个阶段用什么"，最后说"怎么评估"。这比直接背模型名字要有说服力得多。

---

## 3. 数据层：特征工程与数据集

### 3.1 这个模块解决什么问题

工业界推荐系统的特征工程占据了大量工程时间。本项目模拟的场景：

> 快手/TikTok的推荐数据包含用户行为日志（观看、点赞、关注、评论、分享），需要从原始交互记录中提炼出模型可用的特征，同时避免**数据泄露**（用未来信息预测过去）。

KuaiRec 官网：https://kuairec.com/ （快手开源的真实推荐数据集）

### 3.2 数据集字段说明（KuaiRec Schema）

| 字段 | 类型 | 含义 | 取值范围 |
|------|------|------|---------|
| `user_id` | int | 用户ID | 0-499 |
| `video_id` | int | 视频ID | 0-999 |
| `watch_ratio` | float | 观看比例（>1代表重播） | 0.0-2.0 |
| `like` | int | 是否点赞 | 0/1 |
| `follow` | int | 是否关注 | 0/1 |
| `comment` | int | 是否评论 | 0/1 |
| `share` | int | 是否分享 | 0/1 |
| `timestamp` | int | Unix时间戳 | 30天范围 |
| `video_category` | int | 视频类别 | 0-19 |
| `video_duration` | int | 视频时长（秒） | 5-300 |

### 3.3 标签构造

> 🔴 **【面试必考】**
> **Q: 推荐系统的正样本是怎么定义的？**
>
> A: 这没有唯一答案，取决于业务目标：
> - **观看时长**：`watch_ratio >= 0.7`（本项目采用），代表用户真正看完了视频
> - **显式反馈**：点赞/评论/转发（正样本稀少，但质量高）
> - **点击**：只要点击就算正（样本丰富，但有"标题党"问题）
>
> 本项目用 `watch_ratio >= 0.7` 作为正样本，模拟**隐式反馈**推荐场景（无明确点赞，以观看行为为代理信号）。

本项目数据分布：正样本约14%，负样本约86%，**严重不平衡**，这是真实推荐数据的典型特征。

### 3.4 负样本构造策略

> 🔴 **【面试必考】**
> **Q: 为什么负样本构造对召回模型很重要？有哪些策略？**

| 策略 | 原理 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|---------|
| **随机负采样** | 从全库随机采样未交互物品 | 简单实现，效果稳定 | 采样到热门物品概率低，模型学不到难负样本 | 召回模型初期训练 |
| **In-batch负采样** | 同一batch内其他样本的正例item作为负例 | 天然包含热门物品（因为热门item出现概率高），高效（无额外计算） | 存在假负样本（被误标为负的真正正例） | 双塔模型首选 |
| **曝光未点击** | 系统展示但用户未互动的物品 | 质量最高的负样本，减少选择偏差 | 需要曝光日志，并非所有系统都有 | 精排模型 |
| **难负样本挖掘** | 训练后期，用相似度高但未交互的样本 | 防止模型过早收敛 | 需要两阶段训练，复杂度高 | 召回模型后期 |

> ⚠️ **【踩坑警告】**
> In-batch负采样存在**假负样本问题**：batch中两个用户都喜欢的同一个视频，会被互相当作负样本。解决方法：加logQ修正（参考Google双塔论文），或过滤掉已知正例。

> 💡 **【面试表达技巧】**
> 面试时说："我同时实现了随机负采样和in-batch负采样，并在消融实验中对比了两者的Recall@10，结论是in-batch在我们的数据规模下效果更好，但代价是需要处理假负样本问题。" 这比说"我用了in-batch负采样"要深度得多。

### 3.5 特征工程设计决策

#### 用户特征（25维）
```
[0]     avg_watch_ratio      用户平均完播率（活跃度代理指标）
[1]     like_rate            用户点赞率
[2]     follow_rate          用户关注率
[3]     comment_rate         用户评论率
[4]     share_rate           用户分享率
[5:25]  category_preferences 用户对20个类别的偏好分布（归一化）
```

#### 物品特征（3维密集 + 2维类别）
```
dense [0]  historical_ctr    历史平均完播率
dense [1]  item_like_rate    物品被点赞率
dense [2]  log_popularity    log10(1+交互数)，归一化
categorical: item_category   类别ID（送入Embedding层）
categorical: item_dur_bkt    时长分桶 [<15s, 15-30s, 30-60s, 60-120s, >120s]
```

#### 序列特征（20步）
```
history_seq   用户最近20次交互的video_id（1-indexed，0为padding）
history_len   实际序列长度（用于DIN attention masking）
```

> 🟠 **【重要原理】**
> **为什么序列特征重要？** 用户的兴趣是动态变化的。静态的用户画像（平均偏好）无法捕捉"用户刚才在看篮球视频，接下来更可能看体育内容"这种上下文依赖。序列特征是DIN模型的核心输入，在精排阶段带来显著的效果提升。

> ⚠️ **【踩坑警告】**
> **数据泄露陷阱**：用户统计特征（avg_watch_ratio, like_rate等）必须只用训练集的数据计算，绝对不能用全量数据。本项目的 `FeatureEngineer._fit_user_stats()` 只接受 `train_df`，验证集和测试集的用户直接复用训练期计算的统计量。

### 3.6 关键代码解析

```python
# feature_engineering.py - 核心：时间顺序切分
def _temporal_split(self, df):
    n = len(df)
    train_end = int(n * 0.8)  # 按时间切，不是随机切！
    val_end = int(n * 0.9)
    # 关键：df已按timestamp排序，所以前80%是"历史"，后20%是"未来"
    # 这才是真实场景下模型的评估方式
    return df[:train_end], df[train_end:val_end], df[val_end:]

# 序列构建的核心：记录每个用户看到当前样本时的历史
user_cursor = {}  # {user_id: 已处理的交互数}
for idx, row in enumerate(split_df.itertuples()):
    pos_in_hist = user_cursor.get(uid, 0)
    seq, seq_len = self._get_history_at(uid, pos_in_hist, seq_map)
    # seq是该用户在这条交互之前的最近20个视频
    user_cursor[uid] = pos_in_hist + 1  # 更新游标
```

> 🟡 **【加分项】**
> 本项目的序列特征使用了**右对齐padding**（历史在右边，零在左边）。这是因为DIN在做attention时，真实历史总在序列末尾，attention mask可以直接用 `history_len` 屏蔽padding位置。有些实现用左对齐，不同方式在RNN/Transformer中各有优劣。

### 3.7 面试问题精选

> 🔴 **【面试必考】**
> **Q: 推荐系统里为什么要按时间切分数据集，而不是随机切分？**
>
> A: 推荐系统的本质是"用历史行为预测未来行为"。随机切分会造成数据泄露：未来的数据（本应是测试集）被混入训练集。比如用户在1月31日的行为出现在训练集里，但模型需要预测的是1月20日——这在现实中是不可能的。
>
> 按时间切分才能真实评估模型的"泛化到未来"的能力。
>
> 追问：那如果某个用户在训练集里没有历史记录（冷启动），怎么办？
> A: 对于新用户，我们用全局平均特征作为默认值（代码中的`_default_user_feat`）。工业界更常见的方案是：新用户展示热门内容，积累几次交互后再个性化。

> 🔴 **【面试必考】**
> **Q: 你的数据正负样本比例是14% vs 86%，怎么处理样本不平衡？**
>
> A: 几种方案：
> 1. **不做处理，直接训练**：对于AUC指标影响不大，BCE Loss本身能处理不平衡
> 2. **过采样正例**：在Dataset的`__getitem__`里对正例采样多次
> 3. **调整Loss权重**：`BCELoss(pos_weight=torch.tensor([6.1]))` （6.1 = 86/14）
> 4. **重采样负例**：随机丢弃部分负样本，使正负比接近1:1
>
> 本项目召回训练只用正例对（RetrievalDataset只返回正样本），精排训练用全量样本+weighted BCE。

> 🟠 **【重要原理】**
> **Q: in-batch负采样的伪代码是什么？为什么它能节省计算？**
>
> A: 在一个batch里，有B个(user, pos_item)对：
> ```python
> # user_emb: (B, D), item_emb: (B, D)
> scores = user_emb @ item_emb.T  # (B, B)
> # scores[i][j] = user_i 和 item_j 的相似度
> # 对角线是正样本分数，非对角线是负样本分数
> labels = torch.eye(B)  # 对角线为1
> loss = cross_entropy(scores, labels)
> ```
> 节省计算的原因：**一次前向传播算了B个正样本和B*(B-1)个负样本的分数**，相比逐一采样负例效率高很多。当B=256时，相当于每个样本有255个负例。

> 🟡 **【加分项】**
> **Q: KuaiRec数据集有什么特别之处，为什么选它？**
>
> A: KuaiRec是业内罕见的"完全观测数据集"（Fully-Observed Dataset）。大多数推荐数据集只有用户实际看过的视频（存在曝光偏差/选择偏差），但KuaiRec包含了一部分用户对**所有视频**的评分，可以无偏地评估推荐系统的真实效果。这对于评估反事实推荐（"如果给用户推这个视频，他会不会看？"）非常有价值。

> 🟢 **【了解即可】**
> **Q: 为什么用户的category_preferences是20维向量而不是一个int？**
>
> A: 因为用户的兴趣通常是多样的，不是单一类别。一个用户可能60%看体育、30%看美食、10%看科技。用分布向量（histogram）比用单一标签更能表达这种多样性。这种表示方式也叫"兴趣画像向量"。

### 3.8 踩坑与注意事项

- **Windows路径编码**：读取YAML时必须指定 `encoding="utf-8"`，否则在中文路径下会报GBK解码错误
- **序列padding值**：video_id在序列中是1-indexed（真实ID+1），这样0可以作为padding token，对应embedding中的 `padding_idx=0`
- **测试集用户未出现在训练集**：用 `_default_user_feat`（全局均值）兜底，不要用0向量（会引入偏差）

### 3.9 本章学习优先级总结

| 知识点 | 优先级 | 原因 |
|--------|--------|------|
| 召回+排序两阶段架构 | 🔴 | 面试开场必考 |
| 为什么时间切分 | 🔴 | 数据泄露问题高频考点 |
| 负样本构造策略对比 | 🔴 | in-batch vs 随机，深度考察 |
| 正负样本不平衡处理 | 🔴 | 实际工程必须考虑 |
| 特征工程设计（用户/物品/序列） | 🟠 | 理解后续模型输入的基础 |
| KuaiRec数据集特点 | 🟡 | 体现对业界数据集的了解 |
| 序列padding方向（左对齐vs右对齐） | 🟡 | 实现细节，体现工程深度 |
| 数据稀疏性问题 | 🟢 | 背景知识 |

### 3.10 延伸阅读

- **KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems (CIKM 2022)**：快手开源数据集论文，核心贡献是提供无偏的全观测评估数据
- **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (RecSys 2019)**：Google的双塔in-batch负采样+频率修正方案（logQ correction）
- **Self-Attentive Sequential Recommendation (ICDM 2018)**：序列推荐中self-attention的应用

---

## 4. 召回模块：双塔模型 + Faiss

*(Iteration 2 完成后填写)*

---

## 5. 排序模块：DeepFM + DIN

*(Iteration 3 完成后填写)*

---

## 6. 实验与分析：消融实验设计

*(Iteration 4 完成后填写)*

---

## 7. 完整Pipeline与Demo

*(Iteration 5 完成后填写)*

---

## 8. 面试准备：高频题汇总与Pitch模板

*(Iteration 5 完成后填写)*

---

## 9. 延伸阅读：参考论文清单

*(Iteration 5 完成后填写)*

---

*文档版本：v0.1 | 最后更新：Iteration 0 | 作者：recsys-project*
