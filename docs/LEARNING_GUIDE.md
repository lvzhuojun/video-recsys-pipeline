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
2. [项目概述与系统架构](#2-项目概述与系统架构)  *(Iteration 1 完成后填写)*
3. [数据层：特征工程与数据集](#3-数据层特征工程与数据集)  *(Iteration 1 完成后填写)*
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

*(Iteration 1 完成后填写)*

---

## 3. 数据层：特征工程与数据集

*(Iteration 1 完成后填写)*

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
