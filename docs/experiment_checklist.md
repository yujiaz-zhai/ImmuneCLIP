# ImmuneCLIP 实验清单

> 对应论文：*Cleaning Is Not Immunization: Adaptation-Stable Defense Against Backdoor Rebound in Purified CLIP Models*
> 对应大纲：`7-8/paper-outline/03_ImmuneCLIP_USENIX2027_正文顺序完整写作大纲_v5.md`（v5.2 Single-Proxy）
> 目标会议：USENIX Security 2027
> 方法形态：单 Proxy（closed-set unknown-target）+ update-set 方向免疫 + reachable-checkpoint 免疫

本清单是实验的唯一执行依据。第四节表格全部填满即宣告正文实验完成；第五节表格全部填满即宣告附录实验完成。

**填写约定**：`[ ]` 表示待填数据；`—` 表示该项不适用；`PF` 表示 purification failure；`P` 表示通过 eligibility gate。

---

# 一、实验资料

## 1.1 基线

### 1.1.1 攻击基线

| 编号 | 攻击 | 代码/资产来源 | 定位 | 用于 |
|---|---|---|---|---|
| ATK-1 | alignment-enhanced BadCLIP surrogate | 自建，基于 ATK-2 代码库 | 主压力测试；梯度耦合式持久攻击 | 正文主表 |
| ATK-2 | BadCLIP（明确关闭持久性增强） | `https://github.com/LiangSiyuan21/BadCLIP` | 优化触发器、无持久性设计 | 正文主表 |
| ATK-3 | BadNet | PAR/CleanCLIP 代码内置（PAR 中标识 `random`） | 非优化 patch 触发器、经典基线 | 正文主表 |
| ATK-4 | BadNet-Stripes | `https://github.com/nmndeep/PerturbAndRecover`（`badnet_rs`） | 结构化触发器；ViT-B/32 有现成 checkpoint | 正文 Table 4 |
| ATK-5 | Blended-Text | 同上（`water_patt`） | 全图混合触发器 | 正文 Table 4 |
| ATK-6 | Blended-Triangles | 同上（`tri_patt`） | 触发器形态补充 | 附录 |
| ATK-7 | BadEncoder | `https://github.com/jinyuan-jia/BadEncoder` | encoder 级、reference-input 机制，与 caption 投毒完全不同 | 附录（时间不足可砍） |

**明确不做实验、仅在 Related Work 出现**：BadCLIP++（无公开代码/checkpoint）、SBL（`https://github.com/mail-research/SBL-resilient-backdoors`，原始实现为单模态分类，迁移不稳定）、mmPoison（`https://github.com/zqypku/mm_poison`，加入 ATK-7 后机制冗余）。

**ATK-1 的正文表述纪律**：统一称 `alignment-enhanced BadCLIP surrogate`，首次出现时必须列出与 BadCLIP++ 的最小差异（基于公开 BadCLIP 代码 + 仅加入已记录的 gradient-alignment 训练项，不含其未公开组件），不 claim 新攻击。

### 1.1.2 防御基线

| 编号 | 防御 | 代码来源 | 定位 | 用于 |
|---|---|---|---|---|
| DEF-0 | No defense | — | 上界参考 | 正文 |
| DEF-1 | Naive clean FT | 自有脚本 | compute-matched 对照 | 正文 |
| DEF-2 | CleanCLIP | `https://github.com/nishadsinghi/CleanCLIP` | 经典 FT 净化；对强攻击常清洗失败 | 正文 |
| DEF-3 | PAR | `https://github.com/nmndeep/PerturbAndRecover` | 主要 successful purifier | 正文 |
| DEF-4 | InverTune | `https://github.com/Leey21/InverTune` | 第二个 successful purifier；反演类 | 正文 |
| DEF-5 | ImmuneCLIP（ours） | 自有 `immuneclip_new` | 接在 DEF-3 / DEF-4 之后 | 正文 |
| DEF-6 | RVPT | `https://github.com/zhangzf01/RVPT` | 参数高效 prompt 防御 | 仅 Discussion 论证，不进主表 |

**DEF-6 降级理由**：RVPT 输出的是深层视觉 prompt 而非净化后的 encoder 权重，且需下游标注数据训交叉熵。下游用户一旦微调 encoder，prompt 即被丢弃或失效，作为 rebound 基线不对等。其价值是 Discussion 中的附加论点：不修改 encoder 权重的防御根本不会被"带过"下游适配。论文venue标注 `[VERIFY]`（arXiv 2412.20392，含 NeurIPS checklist）。

**不进主表**：CleanerCLIP（无官方代码）、CBPT / Neural Antidote（未找到公开代码）、BDetCLIP（test-time 检测）、RoCLIP / SafeCLIP（training-time 防御）、DECREE / MM-BD（检测类）。以上仅在 Related Work 分类中出现。

### 1.1.3 下游适配基线

| 编号 | 适配方式 | 说明 | 用于 |
|---|---|---|---|
| ADP-1 | Full contrastive FT | 全视觉塔 + 文本塔对比微调 | 正文主设定 |
| ADP-2 | Projection-only / partial FT | 仅训练 projection 层或部分层 | 正文 |
| ADP-3 | LoRA | 仅 ViT-B/32；若未纳入 update bank 则标为 out-of-bank | 正文 Table 4 |
| ADP-4 | MSCOCO retrieval adaptation | 真实下游任务，非同域继续预训练 | 正文 Table 4 |

**术语纪律**：只训练 projection 层时必须称 `projection-only fine-tuning`，不得称 standard linear probe。

## 1.2 模型架构

| 编号 | 架构 | 获取方式 | 用途 |
|---|---|---|---|
| ARCH-1 | RN50（OpenAI 预训练） | `open_clip`：`https://github.com/mlfoundations/open_clip`，`RN50` + `pretrained='openai'`；或 `https://github.com/openai/CLIP` | 主实验；与 BadCLIP/CleanCLIP/PAR/InverTune 一致 |
| ARCH-2 | ViT-B/32（OpenAI 预训练） | 同上 `ViT-B-32` + `openai`；HF `https://huggingface.co/openai/clip-vit-base-patch32` | 架构泛化 |

RN101 / ViT-B/16 / ViT-L-14-336 只在正文实验全部完成后考虑，否则不做。不扩展到 SigLIP / ALBEF / FLAVA。

### 1.2.1 可直接复用的现成 checkpoint（高优先级，省算力）

PAR 官方发布的 ViT-B/32 中毒与清洗 checkpoint，target 均为 `banana`：

| 攻击 | 中毒 checkpoint | PAR 清洗后 checkpoint |
|---|---|---|
| BadNet-Stripes | `https://nc.mlcloud.uni-tuebingen.de/index.php/s/Q6rnTj5bDKeKigp` | `https://nc.mlcloud.uni-tuebingen.de/index.php/s/EpKfgbbsCZJXCRx` |
| Blended-Triangles | `https://nc.mlcloud.uni-tuebingen.de/index.php/s/XaZe8ZCgmM2p3Cf` | `https://nc.mlcloud.uni-tuebingen.de/index.php/s/g2zwG2F323eTMoT` |
| Blended-Text | `https://nc.mlcloud.uni-tuebingen.de/index.php/s/GHKDMzizzmT5mk8` | `https://nc.mlcloud.uni-tuebingen.de/index.php/s/Qxc4FppPsmBHQK3` |

另有 ViT-L/14-336 的 BadNet-Stripes（`.../W83tntA6sFMDL8Z`）与 Blended-Text（`.../TqineSP7YsbaaMF`）中毒 checkpoint，无清洗版本，仅备用。

使用这批资产可同时满足：架构泛化（RN50→ViT-B/32）、触发器形态泛化、第二个 successful-purification rebound 案例，且不需要自行投毒与重跑 PAR。

## 1.3 数据集

| 编号 | 数据 | 来源 | 用途 |
|---|---|---|---|
| DATA-1 | CC3M / GCC 500K 子集 | `https://ai.google.com/research/ConceptualCaptions/`；本地 `GCC_Training500K` | 投毒预训练、purifier clean set、同域下游适配 |
| DATA-2 | ImageNet-1K val | `https://www.image-net.org/` | 零样本 CA@1/@5 与 ASR 主口径 |
| DATA-3 | MSCOCO 2017 | `https://cocodataset.org/#download` | 真实下游任务：图文检索适配 + triggered retrieval ASR |
| DATA-4 | STL10 / GTSRB / SVHN | `torchvision.datasets` | 仅 ATK-7（BadEncoder）专用评测口径 |

Flickr30K 仅作为 MSCOCO 目标类过滤失败时的备选，不与 MSCOCO 同时做。

### 1.3.1 数据卫生要求（每次实验前强制检查）

1. 下游适配集执行目标类字符串过滤，并补同义词、上位词、语义相关词审计；
2. 对下游适配集做图像级目标语义抽检，记录抽检规模与命中数；
3. purifier clean set 与下游适配集按 **image ID 严格去重**，输出去重清单；
4. 报告下游集样本数与目标概念计数（当前 CC3M 10K 严格集为 banana caption = 0）；
5. ImageNet 评测子集**固定且全实验一致**，不允许不同方法用不同评测规模。

## 1.4 其余必备实验资料

| 编号 | 资料 | 说明 |
|---|---|---|
| ASSET-1 | 触发器资产 | BadCLIP 优化触发器、BadNet patch、PAR 四种结构化触发器（`badnet_rs` / `blended_rs` / `tri_patt` / `water_patt`） |
| ASSET-2 | Stage 0 Proxy 产物 | `proxy_trigger.pt` + 完整 metadata：`mode`、`target_index`、`scan_candidates`、最终选出的 `target_index/target_name`、候选集合 $\mathcal Y$ 定义 |
| ASSET-3 | 干净参考模型 | 未中毒的 ARCH-1 / ARCH-2，用于 KD reference、clean control、purified-clean control |
| ASSET-4 | Update probe bank 配置 | clean batch 列表、学习率集合、optimizer 集合、可训练参数范围、bank 刷新频率 $K_{\text{update}}$ |
| ASSET-5 | 固定评测子集清单 | ImageNet val 固定子集的样本 ID 列表 + 哈希 |
| ASSET-6 | eligibility gate 阈值 | 预注册的 $\tau_{\text{ASR}}$ 与 utility floor $\gamma$，实验开始前冻结 |
| ASSET-7 | 环境锁 | torch / CUDA / open_clip 版本、`requirements.lock` |
| ASSET-8 | 算力 | 2–3 × RTX 4090 |

### 1.4.1 Stage 0 威胁模型口径（重要）

已核对 `stage0_blackbox_invert.py` 的生成记录：`mode=scan_then_invert`、`target_index=-1`、`scan_candidates=all`，最终自动选出 `954 / banana`。因此当前威胁模型是：

> **unknown-trigger, closed-set unknown-target**

即防御者不知道真实 trigger 与 target，只知道目标位于预定义候选集合 $\mathcal Y$ 中，并由扫描自动选出 top-1。

正文不得写成开放世界 `unknown-target`。必须在附录报告 target identification 的 Recall@1 / Recall@5（见 A2），不能只展示 banana 这一例。若流程使用了模型梯度或参数，不得称 black-box defense，改称 `target-agnostic scan-then-invert`。

### 1.4.2 eligibility gate 阈值冻结（须在跑实验前定稿）

$$\text{Gate}=\text{P}\iff A_0\le\tau_{\text{ASR}}\ \wedge\ \text{CA}_0\ge\gamma$$

- 主设定建议 $\tau_{\text{ASR}}=0.10$。**注意**：PAR 在 ATK-1 上的 delivery ASR 约 0.064，若取 $\tau_{\text{ASR}}=0.05$ 会把 PAR 判为 PF，主 rebound 案例将不成立。因此 0.05 只能作为敏感性检查行，不能作为主阈值。
- $\gamma$ 建议用相对形式 $\gamma=c\cdot\text{CA}_{\text{pre}}$，$c$ 取值须使 PAR / InverTune 均能通过，且实验前冻结。`c = [ ]`
- 阈值必须按 **attack–purifier–seed 单元**统一应用，不允许只保留通过的 seed。

---

# 二、评价指标

四个核心指标每次实验必算并落盘，另有一个强制记录量。**任何一次正式 run 缺少其中任意一项，视为无效 run，需要重跑**——这是本清单存在的首要目的。

## M1 Clean Utility

$$\text{CA@1},\ \text{CA@5}\quad(\text{ImageNet-1K zero-shot})$$

在每个测量点计算。MSCOCO 适配实验额外报告 R@1 / R@5 / R@10。禁止只报 clean loss 代替效用。

## M2 ASR Trajectory

$$A_t=\text{ASR}_{q^\star}(\theta_t),\quad t\in\{0,5,10,20,30,50,100,200,300\}$$

- $A_{\text{pre}}$：净化前中毒模型 ASR；
- $A_0$：delivery ASR（净化后 / 免疫后，下游第 0 步）；
- 前 50 步密采样，因为当前反弹主要发生在早期；
- 使用真实 trigger 与真实 target 计算，属于 **oracle evaluation gate**，仅用于研究测量，不代表现实部署可执行同一验收。

## M3 Worst Post-FT ASR 与 Rebound-Δ（主安全指标）

$$A_{\text{post}}=\max_{1\le t\le T}A_t,\qquad \Delta R=\max_{t}\left(A_t-A_0\right)_+$$

$A_{\text{post}}$ 是通过 eligibility gate 后的**首要排序指标**。若需包含交付点，另称 $A_{\text{life}}=\max_{0\le t\le T}A_t$（Worst Lifecycle ASR），两者不得混用。

## M4 Normalized AURC

$$\text{AURC}=\frac{1}{T}\int_0^T\left(A_t-A_0\right)_+\,dt$$

首次出现写全称 `area under the rebound curve`，避免与 risk–coverage AURC 混淆。必须在固定适配预算下比较；若不同方法有效更新速度差异明显，补充以参数路径长度为横轴的版本。

## M5 强制记录量：Reactivation Susceptibility

$$\widehat\rho_{\text{SP}}(\mathbf x)=\max_k\left[\left\langle\nabla_\theta S_{\widehat q}(\theta),\widehat{\mathbf u}_k\right\rangle\right]_+$$

不属于论文主指标，但**每次有梯度可算的 run 都必须记录**（至少在 $t=0$ 与 $t\in\{10,50\}$）。E5 的理论对齐图完全依赖它，事后补算需要重跑全部轨迹。同时记录 $S_{\widehat q}$ 与真实 ASR 的配对值以做 calibration。

## 附带落盘项（非指标，但缺失即需重跑）

`GPU-hours`、`peak memory`、`wall-clock`、`seed`、`code commit`、`data hash`、`ckpt hash`、完整 config dump、`Revival Step` $T_{0.5}$。

---

# 三、标准化实验流程

## 3.0 通用规范

### run_id 命名

```
{arch}_{attack}_{purifier}_{method}_{adapt}_{seed}_{tag}
例：rn50_align_par_immunev5_fullft_s42_main
```

### 目录结构

```
runs/{run_id}/
  config.json          # 完整配置 + code commit + data/ckpt hash
  ckpt/                # delivery checkpoint
  eval_step{t}.json    # 每个测量点的 M1–M5
  traj.json            # 汇总轨迹 + A_post / ΔR / AURC / T_0.5
  train.log
  figs/
```

### `traj.json` 必备字段

```json
{
  "run_id": "", "arch": "", "attack": "", "purifier": "", "method": "",
  "adapt": "", "seed": 0,
  "a_pre": 0.0, "ca_pre": 0.0,
  "a_0": 0.0, "ca_0": 0.0, "gate": "P|PF",
  "steps": [0,5,10,20,30,50,100,200,300],
  "asr": [], "ca1": [], "ca5": [],
  "a_post": 0.0, "rebound_delta": 0.0, "aurc": 0.0, "revival_step": null,
  "rho_sp": {"0": 0.0, "10": 0.0, "50": 0.0},
  "path_length": 0.0, "gpu_hours": 0.0, "peak_mem_gb": 0.0
}
```

### 强制断言（不通过直接报错退出，不允许静默继续）

1. checkpoint 加载成功且 `state_dict` 键完全匹配，加载失败不得 fallback；
2. 下游数据集 target 计数为 0；
3. purifier clean set 与下游集 image ID 交集为空；
4. 评测子集哈希与 ASSET-5 一致；
5. `total_loss` 日志包含全部启用项（含 $\mathcal L_{\text{reach}}$），不得只记录部分项；
6. 若训练 text tower，则 text classifier 必须按步重建；否则断言 text tower 已冻结；
7. 方向导数符号通过有限差分校验：$S_q(\theta+\Delta\theta)-S_q(\theta)$ 与 $\langle\nabla S_q,\Delta\theta\rangle$ 符号一致。

## 3.1 主流程 P0–P5

```
P0  资产准备
    获取/训练中毒 checkpoint；数据卫生检查；固定评测子集
    ↓  测 M1 + M2 → 得 A_pre, CA_pre

P1  防御基线净化（DEF-2/3/4）
    ↓  测 M1 + M2 + M5 → 得 A_0, CA_0, ρ̂_SP(0)
    ↓  应用 eligibility gate → 判定 P / PF
    ※ 判为 PF 的 checkpoint 不进入 rebound 主分析，单独记为 purification failure

P2  反弹微调（ADP-1/2/3/4，clean 数据）
    ↓  按固定 step grid 测 M1 + M2，t∈{10,50} 加测 M5
    ↓  算 M3 + M4 → 得 A_post, ΔR, AURC, T_0.5

P3  ImmuneCLIP 免疫（DEF-5，输入为 P1 的 P 类 checkpoint）
    Stage 0：scan_then_invert 得单 Proxy q̂（记录 ASSET-2 全部字段）
    Stage 1：L_util + λ_a·L_anchor + λ_d·L_dir + λ_r·L_reach
    ↓  测 M1 + M2 + M5 → 免疫后 delivery 指标
    ↓  重新应用 eligibility gate（免疫不得抬高即时 ASR）

P4  免疫后反弹再测试
    ↓  与 P2 **完全相同**的适配配置（同数据、同优化器、同 lr、同 step grid、同 seed）
    ↓  测 M1–M5 → 得免疫后 A_post, ΔR, AURC

P5  对照组（与 P2/P4 同配置）
    C1 clean control：干净模型直接走 P2
    C2 purified-clean control：干净模型先过同一 purifier 再走 P2
    C3 compute-matched clean FT：purifier 后用与 ImmuneCLIP 等量算力做纯干净 FT，再走 P4
```

C3 是必需控制，用于排除"提升只来自额外训练预算"。

## 3.2 消融流程

原则：**只改一个开关，其余全部冻结，并共享 P0/P1 产物**。

```
固定：arch, attack, purifier checkpoint, Proxy q̂, update bank 配置,
      适配配置, step grid, seed 集合
逐级：A → H（见 E4）
每级仅重跑 P3 + P4，不重跑 P0/P1/P2
```

消融的 Go/No-Go 判据写在 E4 中。任一级若无独立增益，对应模块在论文中降级或删除，不为"理论完整性"保留。

## 3.3 参数实验流程

原则：**单变量扫描 + 单场景选参 + 选定后全局冻结**。

```
1. 选定唯一调参场景：ARCH-1 + ATK-1 + DEF-4(InverTune) + ADP-1
2. 单变量扫描（其余取默认）：
   λ_a ∈ {0.1, 0.5, 1, 5}
   λ_d ∈ {0.1, 0.5, 1, 5}
   λ_r ∈ {0, 0.1, 0.5, 1}
   τ_s ∈ {0.01, 0.05, 0.1}
   update bank 大小 K ∈ {2, 4, 8}
   reach steps h ∈ {0, 1, 2}
3. 选参准则：在 CA@1 下降不超过预设容差的前提下最小化 A_post
4. 冻结：选定值写入默认 config，**不再针对其他攻击/架构单独调参**
5. 所有其他场景一律使用冻结后的同一组超参
```

第 4 步是审稿可信度的关键：若每个攻击单独调参，泛化结论无效。

---

# 四、进入正文的实验清单

共 8 项，按优先级排列。每项含 4.1–4.5 五个部分。全部表格填满即正文实验完成。

**统一实验条件**：ARCH-1 为主，seed 采用配对分层设计（固定同一 attack/purifier checkpoint 后配对 adaptation seed），至少 3 seeds，报告均值 ± 标准差或 95% CI 并说明误差类型。失败 run 不得静默删除。

---

## E1 净化与良性适配的安全组合失效测量

### E1.1 实验描述

**标题**：Successful purification does not survive benign downstream adaptation.

**描述**：对多个攻击 × 多个净化器 × 多种合法适配，测量净化交付时的安全状态是否在完全干净的下游微调中保持。严格区分 purification failure 与 rebound failure。本实验同时产出全文 hero figure，是 C1 / C2 / C9a / C10 的主要证据来源。

**所需资料**：ATK-1/2/3、DEF-2/3/4、ADP-1/2、ARCH-1、DATA-1、DATA-2、ASSET-1/3/5/6、流程 P0–P2 + P5。

### E1.2 实验主表格

论文版本：full FT 进正文，projection-only 移附录。

| Attack | Purifier | Adapt | $A_{\text{pre}}$ | CA$_0$ | $A_0$ | Gate | $A_{\text{post}}$ | $\Delta R$ | AURC | CA$_T$ |
|---|---|---|---|---|---|---|---|---|---|---|
| Align-surrogate | None | full | [ ] | [ ] | [ ] | — | [ ] | [ ] | [ ] | [ ] |
| Align-surrogate | CleanCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align-surrogate | PAR | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align-surrogate | InverTune | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | CleanCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | PAR | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | InverTune | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadNet | CleanCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadNet | PAR | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadNet | InverTune | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align-surrogate | PAR | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align-surrogate | InverTune | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | PAR | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadNet | PAR | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| *C1 clean control* | None | full | — | [ ] | [ ] | — | [ ] | [ ] | [ ] | [ ] |
| *C2 purified-clean* | PAR | full | — | [ ] | [ ] | — | [ ] | [ ] | [ ] | [ ] |
| *C2 purified-clean* | InverTune | full | — | [ ] | [ ] | — | [ ] | [ ] | [ ] | [ ] |

判为 `PF` 的行**不得**把其高 $A_{\text{post}}$ 描述为 rebound。

### E1.3 实验图

**需要，且这是全文 hero figure（Fig.1b）。本实验图与表同等重要，缺图不可。**

推荐方案：**带置信区域的折线图**。

- 横轴：下游适配 step（0→300，前 50 步密采样）；纵轴：ASR@1；
- 曲线：PAR、InverTune、`PAR+ImmuneCLIP`、`InverTune+ImmuneCLIP`、clean control；
- 每条曲线用 3 seeds 的均值 + 半透明标准差阴影带；
- 标注：step-0 处的 eligibility gate 阈值横线、$\Delta R$ 箭头、文字框 `downstream data contains 0 target-class captions`；
- 灰度可读：5 种线型 + marker，阴影带用网格纹理。

选择理由：本实验的核心断言是"安全状态随适配轨迹失效"，这是时间序列断言而非单点对比，折线图是唯一能同时呈现交付点安全、早期快速反弹和对照组平坦的图型；置信带同时把"多 seed 稳健性"这一审稿必问点在同一张图内回答，避免额外表格。不使用柱状图，因为柱状图会丢掉"何时反弹"这一最关键信息。

### E1.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E1.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E2 方向干预：反弹的因果证据

### E2.1 实验描述

**标题**：Removing the positive reactivation component suppresses rebound.

**描述**：在下游适配中对实际参数更新做三种处理并对比 ASR 轨迹，把"梯度同向"从相关证据升级为方向特异的因果证据。这是全文抗反驳强度最高的实验，也是 §3 与 §4 的衔接点。

三组处理，令 $\widehat{\mathbf s}_t=\nabla S/\|\nabla S\|$、$\alpha_t=[\langle\Delta\theta_t,\widehat{\mathbf s}_t\rangle]_+$：

1. Normal：$\Delta\theta_t$
2. Reactivation-projected：$\Delta\theta_t-\alpha_t\widehat{\mathbf s}_t$（只移除正向分量）
3. Matched-component random control：在同一层支持内采样单位方向 $\widehat{\mathbf r}_t$，取 $\Delta\theta_t-\alpha_t\widehat{\mathbf r}_t$，再重标定至与第 2 组相同的最终更新范数

**所需资料**：ATK-1、DEF-3、ADP-1、ARCH-1、DATA-1/2、ASSET-1/5；需在 `run_downstream.py` 增加投影开关与随机对照开关。

**已知待修项**：现有脚本若删除的是 signed component、或随机组未做 component / final-norm 匹配，则现有右图只能作为 preliminary，正文版本必须按上述定义重跑；累计暴露量的符号必须先通过 3.0 断言 7。

### E2.2 实验主表格

| Intervention | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | CA$_T$ | $T_{0.5}$ |
|---|---|---|---|---|---|---|
| Normal benign update | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Reactivation-projected | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Matched-component random | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Shuffled-proxy direction | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### E2.3 实验图

**需要。本实验在论文中以图为主、表可压缩为一句数字或移入附录。**

推荐方案：**带置信区域的折线图 + 局部嵌入子图**。

- 主轴：三条 ASR 轨迹（Normal / Reactivation-projected / Matched-component random），各带 3 seeds 置信带；随机对照需多次采样 $\widehat{\mathbf r}_t$ 并画成置信带而非单线；
- 内嵌子图：同三组的 CA@1 轨迹，用于证明效用未崩塌（不使用 clean loss 代替）；
- 图内不放装饰性标题，caption 自包含。

选择理由：因果断言的说服力来自"处理组与两个对照组在同一时间轴上的分离"，折线图直接呈现这种分离；随机对照的置信带是堵死"只是步长变小"这一替代解释的关键视觉元素。CA@1 用内嵌子图而非独立图，是因为它只承担"效用未被破坏"的辅助断言，独立成图会浪费版面。

### E2.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E2.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E3 ImmuneCLIP 主结果

### E3.1 实验描述

**标题**：ImmuneCLIP reduces post-adaptation backdoor risk while preserving delivery-time safety and utility.

**描述**：在两个 successful purifier、两个攻击、两种适配上评估冻结后的 ImmuneCLIP（`anchor + update-set L_dir + one-step L_reach`），并与 purifier-only 及 compute-matched clean FT 对照。这是论文的主表。

**前置条件**：必须使用冻结后的新方法，不得用当前 `suppression + single-gradient L_fo` 的 v5 原型数据充当最终结果；`L_reach` 必须是真实 virtual optimizer step 上重算 $\widehat\rho_{\text{SP}}$，而非 `reach_radius × h × unit_dir` 的方向近似。

**所需资料**：ATK-1/2、DEF-3/4/5、ADP-1/2、ARCH-1、DATA-1/2、ASSET-2/3/4/5/6、流程 P3–P5。

### E3.2 实验主表格

| Attack | Purifier | Method | Adapt | CA$_0$ | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | CA$_T$ | $\widehat\rho_{\text{SP}}$ | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Align | PAR | purifier only | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | PAR | + compute-matched FT | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | PAR | + ImmuneCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | InverTune | purifier only | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | InverTune | + compute-matched FT | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | InverTune | + ImmuneCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | PAR | purifier only | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | PAR | + ImmuneCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | InverTune | purifier only | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | InverTune | + ImmuneCLIP | full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | PAR | purifier only | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | PAR | + ImmuneCLIP | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | InverTune | purifier only | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align | InverTune | + ImmuneCLIP | proj | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

排序主指标为 $A_{\text{post}}$，其次 $A_0$ 与效用。`Revival Step` 不进正文主表，落附录。

### E3.3 实验图

**条件性需要。** 表已是主要交付物；图只在存在明显安全—效用权衡时补充。

- 若 ImmuneCLIP 在 $A_{\text{post}}$ 与 CA 两个维度上同时优于所有基线，则**不出图**，避免与表重复浪费版面；
- 若存在权衡（例如降低 $A_{\text{post}}$ 伴随 CA 下降），推荐 **帕累托前沿图**：横轴 CA$_T$ / downstream utility，纵轴 $A_{\text{post}}$，点为各方法 × 各配置，连出前沿边界，理想方向标注在左上角。

选择理由：帕累托前沿图是唯一能一眼说明"我们不是靠牺牲效用换安全"的图型，而这恰是防御类论文最常被质疑之处；但若没有真实权衡，该图不承载新信息，此时数值表格更紧凑，符合 13 页硬约束。

### E3.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E3.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E4 逐级消融

### E4.1 实验描述

**标题**：Which components are necessary?

**描述**：从 purifier-only 逐级加到完整 ImmuneCLIP，证明本方法不是单梯度符号反转，且每个模块都有独立增益。这是回应"像 BadCLIP++ 防御版"这一最大拒稿风险的核心实验。

**场景**：仅在 `ATK-1 + DEF-4` 与 `ATK-2 + DEF-3` 两个场景上运行，共享 P0/P1 产物（见 3.2）。

**所需资料**：同 E3，外加 single-gradient 与 `L_traj` 两个对照实现开关。

### E4.2 实验主表格

场景一：Align-surrogate + InverTune + full FT

| # | Method | CA$_0$ | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | CA$_T$ | $\widehat\rho_{\text{SP}}$ | GPU-h |
|---|---|---|---|---|---|---|---|---|---|
| A | Purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| B | + compute-matched clean FT | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| C | + $\mathcal L_{\text{anchor}}$ / suppression only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| D | + single-gradient decorrelation | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E | + update-set $\mathcal L_{\text{dir}}$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| F | + directional approx $\mathcal L_{\text{traj}}$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| G | + one-step $\mathcal L_{\text{reach}}$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| H | Full ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

场景二：BadCLIP + PAR + full FT（同列，A–H）

| # | Method | CA$_0$ | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | CA$_T$ | $\widehat\rho_{\text{SP}}$ | GPU-h |
|---|---|---|---|---|---|---|---|---|---|
| A | Purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| C | + $\mathcal L_{\text{anchor}}$ only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| D | + single-gradient | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| E | + update-set $\mathcal L_{\text{dir}}$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| G | + one-step $\mathcal L_{\text{reach}}$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| H | Full ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

**Go/No-Go**：E 不优于 D → update-set 不作为 headline；G 不优于 E/F → reachable 删除或降级；C 已解释全部效果 → 重新审视方法创新；F 优于 G → 承认只做到方向近似，理论表述同步下调。

### E4.3 实验图

**不需要。** 表格是唯一合适的交付形式。

理由：本实验同时比较 8 个方案 × 6 个指标，任何图型都只能呈现其中一个指标，反而丢掉"安全提升是否以效用和算力为代价"这一必须同屏比较的信息；且方法名较长，柱状图会造成标签重叠。若后期确需补图，选 **横向条形图** 只画 $A_{\text{post}}$（横向可容纳长方法名，避免 X 轴文字倾斜），但优先级低于版面。

### E4.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E4.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E5 理论量与真实反弹的对齐

### E5.1 实验描述

**标题**：The theoretical risk quantity predicts actual rebound.

**描述**：验证 $\widehat\rho_{\text{SP}}$ 不是装饰性理论量，而是可测且能预测真实反弹的指标。若本实验不成立，条件性轨迹界必须从 headline theoretical guarantee 降级为 risk decomposition。

**数据点来源**：不同攻击强度（`λ_align = 0 / low / medium / high` 的 alignment-dose 序列）、不同 purifier、E4 各消融级别、不同下游 pipeline，全部复用已有 run 的 M5 记录，**不需要新的训练**。

**所需资料**：E1/E3/E4 全部 run 的 `traj.json` 中的 `rho_sp` 字段；alignment-dose 序列需额外训练 4 个攻击 checkpoint。

### E5.2 实验主表格

| Config | $\widehat\rho_{\text{SP}}(t{=}0)$ | $\widehat\rho_{\text{SP}}(t{=}50)$ | $A_{\text{post}}$ | AURC | held-out coverage angle | proxy–oracle grad cos |
|---|---|---|---|---|---|---|
| Align + PAR | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align + InverTune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align + PAR + ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align + InverTune + ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP + PAR | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP + PAR + ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\lambda_{\text{align}}=0$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\lambda_{\text{align}}=$ low | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\lambda_{\text{align}}=$ med | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\lambda_{\text{align}}=$ high | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

汇总统计：Spearman $\rho$ = `[ ]`，Pearson $r$ = `[ ]`，$n$ = `[ ]`；$S_q$ 与真实 ASR 的 calibration gap = `[ ]`；轨迹界经验松弛度 = `[ ]`。

### E5.3 实验图

**需要，且本实验以图为主、表移附录。**

推荐方案：**散点拟合图**。

- 横轴 $\widehat\rho_{\text{SP}}$，纵轴 AURC（或 $A_{\text{post}}$）；
- 每个点为一个配置，用 marker 形状区分类别（攻击强度 / purifier / 消融级别 / pipeline），免疫前后用空心与实心区分；
- 叠加拟合线与置信带，图内标注 Spearman $\rho$ 与 $n$；
- 若 alignment-dose 序列呈单调关系，用箭头或渐变色标出 dose 方向。

选择理由：本实验的断言是"两个连续量之间存在预测关系"，散点拟合图是学术界对该类断言的标准呈现方式，且能同时展示离散配置的聚集结构；加拟合线与相关系数把定性观察转为可检验的定量结论。不用热力图，因为配置维度不构成规整矩阵；不用柱状图，因为会丢掉相关性这一核心信息。

### E5.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E5.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E6 跨攻击、架构、任务与 pipeline 的泛化

### E6.1 实验描述

**标题**：Does the defense generalize beyond the tuned configuration?

**描述**：验证方法不是对自建攻击与自建适配的闭环过拟合。四类 held-out 维度：未见攻击 / 未见架构 / 真实下游任务 / 未见优化器与学习率。全部使用 3.3 冻结后的同一组超参，不得重新调参。

**关键提示**：ViT-B/32 部分直接使用 1.2.1 的 PAR 现成中毒与清洗 checkpoint，无需自行投毒与重跑 PAR，这是本项最省算力的路径。

**所需资料**：ATK-3/4/5、ARCH-2、ADP-3/4、DATA-3、1.2.1 全部 checkpoint、DEF-3/4/5。MSCOCO 需完成目标语义过滤与 `triggered retrieval ASR@K` 指标实现。

### E6.2 实验主表格

| Held-out 维度 | Setting | CA$_0$ | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | CA$_T$ | In/Out-of-bank |
|---|---|---|---|---|---|---|---|---|
| 未见攻击 | BadNet + PAR | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 未见攻击 | BadNet + PAR + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 触发器形态 | BadNet-Stripes + PAR (ViT-B/32) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 触发器形态 | BadNet-Stripes + PAR + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 触发器形态 | Blended-Text + PAR (ViT-B/32) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 触发器形态 | Blended-Text + PAR + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 架构 | ViT-B/32, full FT, purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 架构 | ViT-B/32, full FT, + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| Pipeline | ViT-B/32, LoRA, purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | out |
| Pipeline | ViT-B/32, LoRA, + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | out |
| Optimizer | held-out lr, + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Optimizer | held-out optimizer, + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | out |
| 真实任务 | MSCOCO retrieval, purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| 真实任务 | MSCOCO retrieval, + Immune | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |

MSCOCO 两行额外报告：R@1 / R@5 / R@10 = `[ ]`，triggered retrieval ASR@1 / @5 = `[ ]`。

### E6.3 实验图

**条件性需要。** 表为主。

若最终矩阵超过约 12 个有效单元、表格在两栏排版下不可读，推荐 **热力图**：行为 held-out 维度设定，列为 `purifier only` 与 `+ImmuneCLIP`，色深编码 $A_{\text{post}}$，单元内叠印数值。

选择理由：热力图是呈现"多设定 × 多方法"性能矩阵的标准选择，能让审稿人一眼看出免疫列整体变浅，即泛化是系统性的而非个例；单元内叠印数值可保留精确读数，避免纯色块被质疑不可量化。但在单元数较少时表格更精确，因此设为条件性。

### E6.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E6.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E7 自适应攻击者

### E7.1 实验描述

**标题**：What happens when the attacker knows ImmuneCLIP?

**描述**：至少实现一种针对性自适应攻击，报告防御边界与失败模式。不要求在所有自适应攻击下完胜，但必须诚实给出退化幅度。

四类自适应方向（至少实现第一类，其余尽力）：

1. 同时优化 delivery ASR、持久性与 proxy evasion；
2. 让真实 trigger 的风险梯度远离所使用的单 Proxy；
3. 把后门作用分散到未被选中的层；
4. 针对已知 update bank 优化 held-out 适配方向。

**所需资料**：ATK-1 训练代码（加入 evasion 项）、DEF-3/4/5、ARCH-1、ASSET-2/4。

### E7.2 实验主表格

| Adaptive variant | CA$_0$ | $A_0$ | Gate | $A_{\text{post}}$ | $\Delta R$ | proxy–oracle grad cos | $\epsilon_q$ 代理 | $\epsilon_u$ 代理 | 相对非自适应退化 |
|---|---|---|---|---|---|---|---|---|---|
| Non-adaptive（参考行） | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | — |
| A1 proxy-evasion | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| A2 layer-dispersion | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| A3 out-of-bank direction | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

同时报告：自适应攻击后是否仍能被 purifier 即时净化 = `[ ]`。

### E7.3 实验图

**不需要。** 自适应攻击的关键信息是"退化了多少、以及 coverage error 是否同步变大"，这是少量精确数值的比较，表格最合适。且自适应攻击变体数量少，作图信息密度过低，不值一整张图的版面。

### E7.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E7.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## E8 计算开销

### E8.1 实验描述

**标题**：What is the cost of immunization?

**描述**：报告 ImmuneCLIP 相对 purifier-only 的额外开销，并给出单卡可复现配置。所有数字必须实测，不得用"约 2–3 倍"的估计代替。

**所需资料**：全部 run 的 `gpu_hours` / `peak_mem_gb` / `wall-clock` 记录（已在 3.0 中强制落盘，无需额外实验）。

### E8.2 实验主表格

| Stage | GPU-h | Peak mem (GB) | Wall-clock | 相对 purifier-only 倍数 |
|---|---|---|---|---|
| Purifier (PAR) | [ ] | [ ] | [ ] | 1.0× |
| Purifier (InverTune) | [ ] | [ ] | [ ] | [ ] |
| Stage 0 proxy inversion | [ ] | [ ] | [ ] | [ ] |
| Update bank construction | [ ] | [ ] | [ ] | [ ] |
| $\mathcal L_{\text{dir}}$ | [ ] | [ ] | [ ] | [ ] |
| $\mathcal L_{\text{reach}}$ (1-step) | [ ] | [ ] | [ ] | [ ] |
| ImmuneCLIP total | [ ] | [ ] | [ ] | [ ] |
| 单卡可复现配置 | [ ] | [ ] | [ ] | [ ] |

### E8.3 实验图

**不需要。** 正文只需总额外开销的少量数字，表格已足够；分项细节移附录。若审稿明确要求可视化，选 **堆叠柱状图** 展示总开销的分项构成（Stage 0 / bank / dir / reach），但正文不预留版面。

### E8.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### E8.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

# 五、附录实验

共 6 项，格式与第四节一致。优先级低于正文实验，但 A1 与 A2 是审稿必问项，不可省。

---

## A1 排除替代解释

### A1.1 实验描述

**标题**：Rebound is not merely undoing purification or recovering utility.

**描述**：排除"下游微调只是把模型推回中毒 checkpoint"与"反弹只是效用恢复的副作用"两种替代解释。支撑 C10。

**所需资料**：ATK-1、DEF-3/4、ARCH-1、ASSET-3；需实现参数距离追踪与 checkpoint 插值评测脚本。

### A1.2 实验主表格

| 控制项 | 观测量 | 结果 |
|---|---|---|
| purified-clean control | $A_{\text{post}}$ / $\Delta R$ | [ ] |
| 参数距离轨迹 | $\|\theta_t-\theta_{\text{clean}}\|$ / $\|\theta_t-\theta_{\text{poisoned}}\|$ / $\|\theta_t-\theta_{\text{purified}}\|$ | [ ] |
| poisoned↔purified 插值 | ASR($\alpha$), $\alpha\in\{0,0.25,0.5,0.75,1\}$ | [ ] |
| utility-matched 对照 | 相同 CA 恢复量下的 ASR | [ ] |
| 跨数据源适配 | 与净化不同数据源/任务目标的 $A_{\text{post}}$ | [ ] |

### A1.3 实验图

**需要（附录）。** 推荐 **带置信区域的折线图**（双子图）：左子图为三条参数距离随 step 的轨迹，右子图为 poisoned↔purified 插值曲线 ASR($\alpha$)。

选择理由：这两个断言都是"某量沿一维连续变量如何变化"，折线图最直接；把两者并置成一张双子图可在附录内以最小版面同时否证"回退到中毒点"与"沿插值路径单调恢复"两种解释。

### A1.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### A1.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## A2 单 Proxy 保真度与 target 识别成功率

### A2.1 实验描述

**标题**：Single-proxy fidelity and closed-set target identification.

**描述**：由于当前 Stage 0 是 `scan_then_invert` 自动选 target（见 1.4.1），必须证明 target 识别不是只在 banana 一例上成功，并给出单 Proxy 与真实后门的方向偏差。这是 $\epsilon_q$ 的经验支撑，也是审稿必问项。

**关键纪律**：每次训练仍只使用一个 Proxy；跨 inversion seed 的结果只能描述为稳定性评测，不得描述为 proxy bank。

**所需资料**：ATK-1/2/3、多 target 中毒 checkpoint、ASSET-2、Stage 0 脚本（需支持多 seed 与候选集配置）。

### A2.2 实验主表格

| Setting | Target Recall@1 | Recall@5 | proxy–oracle grad cos | $A_0$ | $A_{\text{post}}$ |
|---|---|---|---|---|---|
| Align, seed 1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align, seed 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| Align, seed 3 | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadCLIP | [ ] | [ ] | [ ] | [ ] | [ ] |
| BadNet | [ ] | [ ] | [ ] | [ ] | [ ] |
| 另一 target（非 banana） | [ ] | [ ] | [ ] | [ ] | [ ] |
| Oracle trigger（上界） | — | — | 1.0 | [ ] | [ ] |
| Wrong target（下界） | — | — | [ ] | [ ] | [ ] |
| Shuffled / random proxy | — | — | [ ] | [ ] | [ ] |
| Proxy augmentation off | [ ] | [ ] | [ ] | [ ] | [ ] |

### A2.3 实验图

**需要（附录）。** 推荐 **小提琴图**：横轴为 setting（多 inversion seed / 多攻击 / wrong target / random proxy），纵轴为 proxy–oracle 风险梯度余弦，展示各组的概率密度分布。

选择理由：本实验的核心是"单 Proxy 的方向保真度是否稳定"，这是一个分布性质而非单点数值；小提琴图能直接呈现分布形状（例如是否出现双峰，即部分 seed 反演失败），比箱线图更能体现统计严谨性，也比条形图更能说明失败模式的存在性。

### A2.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### A2.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## A3 Reachable 变体消融

### A3.1 实验描述

**标题**：Which reachable formulation actually helps?

**描述**：区分方向近似、真实一步虚拟更新、多步、以及随机球/SAM 球扰动。用于回答"reach 是否必要、以及必须是哪种 reach"。当前 v8 的 `reach_steps=2` 未优于 v5，需要在修正全局聚合后重新判定。

**所需资料**：同 E4；需实现 `L_traj` / `L_reach` / SAM 球 / 随机球四种开关。

### A3.2 实验主表格

| Variant | CA$_0$ | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | $\widehat\rho_{\text{SP}}$ | GPU-h |
|---|---|---|---|---|---|---|---|
| no reach | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| directional approx（逐项 backward，现 v8） | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| directional approx（全局聚合） | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| real 1-step $\mathcal L_{\text{reach}}$ | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| real 2-step | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| random parameter ball | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| SAM perturbation | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| explicit HVP / cross-curvature | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

若 explicit HVP 无独立增益，从论文中删除而非保留。

### A3.3 实验图

**不需要。** 八个变体 × 多指标的比较，表格最精确；且这是内部机制取舍，不承担对外主张，不值附录图的版面。

### A3.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### A3.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## A4 超参数敏感性

### A4.1 实验描述

**标题**：Hyperparameter sensitivity.

**描述**：按 3.3 的单变量扫描流程执行，证明方法不依赖脆弱的调参，并给出冻结后的默认值。仅在唯一调参场景上做。

**所需资料**：ARCH-1 + ATK-1 + DEF-4 + ADP-1 的固定产物；DEF-5 各超参配置。

### A4.2 实验主表格

| 变量 | 取值 | CA$_0$ | $A_0$ | $A_{\text{post}}$ | AURC | CA$_T$ |
|---|---|---|---|---|---|---|
| $\lambda_a$ | 0.1 / 0.5 / 1 / 5 | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\lambda_d$ | 0.1 / 0.5 / 1 / 5 | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\lambda_r$ | 0 / 0.1 / 0.5 / 1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| $\tau_s$ | 0.01 / 0.05 / 0.1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| bank 大小 $K$ | 2 / 4 / 8 | [ ] | [ ] | [ ] | [ ] | [ ] |
| $K_{\text{update}}$ | 1 / 5 / 20 | [ ] | [ ] | [ ] | [ ] | [ ] |
| clean set 大小 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 层选择 top-k | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

冻结后的默认配置：`[ ]`

### A4.3 实验图

**需要（附录）。** 推荐 **双 Y 轴图**：横轴为超参取值（对数刻度），左轴 CA@1，右轴 $A_{\text{post}}$，每个超参一个子图，用分面网格排列。

选择理由：超参敏感性的关键断言是"存在一个宽区间同时保住效用与安全"，而 CA 与 ASR 量纲不同，双 Y 轴是呈现该权衡的标准做法；分面网格让多个超参共享坐标轴、在附录一张图内完成，避免每个超参单独出图。

### A4.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### A4.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## A5 逐 seed 与完整轨迹

### A5.1 实验描述

**标题**：Per-seed results and full trajectories.

**描述**：披露正文主表每一格的逐 seed 数值与完整轨迹，含 Revival Step 与路径长度归一化 AURC。不需要新实验，全部从 `traj.json` 导出。

**所需资料**：E1 / E3 / E4 / E6 的全部 `traj.json`。

### A5.2 实验主表格

| run_id | seed | CA$_0$ | $A_0$ | $A_{\text{post}}$ | $\Delta R$ | AURC | path-normalized AURC | $T_{0.5}$ |
|---|---|---|---|---|---|---|---|---|
| [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

按行展开全部正式 run。同时列出所有失败 / 弃用 run 及原因：`[ ]`

### A5.3 实验图

**不需要。** 完整轨迹以表格与 artifact 中的原始 JSON 披露即可；E1 与 E2 的折线图已承担轨迹可视化职责，重复作图无新信息。

### A5.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### A5.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

## A6 BadEncoder 跨机制泛化

### A6.1 实验描述

**标题**：Generalization to an encoder-level attack with a different mechanism.

**描述**：BadEncoder 通过 reference inputs 微调已有 encoder 注入后门，机制与 caption 投毒完全不同，且威胁模型即"攻击者发布中毒 encoder"，与本文供应链叙事吻合。用于最强意义上的 C9a / C9b。

**已知成本与风险（时间不足可直接砍掉本项）**：

1. 其 ASR 定义在自身下游任务（STL10 / GTSRB / SVHN 零样本），需单独一套 eligibility gate 与 rebound protocol；
2. 官方代码基于 torch 1.7 与内置旧 CLIP 模块，需移植；
3. PAR / CleanCLIP / InverTune 均未针对它设计，很可能清洗失败，届时只能记为 purification failure，不构成 rebound 案例；
4. 它只改图像编码器，文本塔干净，与本文视觉塔为中心的 update bank 兼容。

**所需资料**：`https://github.com/jinyuan-jia/BadEncoder`（已确认含 `clip/` 与 `zero_shot.py`，支持 CLIP 零样本评测）、DATA-4、ARCH-1 或 ARCH-2、DEF-3/4/5。

### A6.2 实验主表格

| Downstream task | Purifier | Method | CA$_0$ | $A_0$ | Gate | $A_{\text{post}}$ | $\Delta R$ | AURC |
|---|---|---|---|---|---|---|---|---|
| STL10 | PAR | purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| STL10 | PAR | + ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| GTSRB | PAR | purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| GTSRB | PAR | + ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| SVHN | PAR | purifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| SVHN | PAR | + ImmuneCLIP | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### A6.3 实验图

**不需要。** 本项的作用是提供跨机制的存在性证据，数值表格即可；若最终三个下游任务都成立且需要强化视觉表达，复用 E6 的热力图追加三行，不单独出图。

### A6.4 实验 log 及实验图所在路径

请把实验进行的 log 和产出的实验图的地址补充在此。

### A6.5 实验中问题记录

请把实验进行中遇到的问题补充在此。

---

# 六、完成判据

## 正文实验完成

- [ ] E1–E8 全部主表格填满，且每格含 3 seeds 的均值与误差
- [ ] E1 / E2 / E5 三张必需图产出，灰度可读、caption 自包含
- [ ] 全部 run 的 M1–M5 与附带落盘项完整，无缺项重跑
- [ ] eligibility gate 阈值已冻结并在 E1 中报告敏感性行
- [ ] 3.0 的七条强制断言在所有正式 run 中通过
- [ ] E4 的 Go/No-Go 判定已执行，未通过的模块已在大纲中降级或删除

## 附录实验完成

- [ ] A1 / A2 / A3 / A4 / A5 主表格填满（A6 可选）
- [ ] A1 / A2 / A4 三张附录图产出
- [ ] 全部失败与弃用 run 已在 A5 中披露

## 待定项（须在开跑前定稿）

- [ ] $\tau_{\text{ASR}}$ 与 $\gamma$ 的具体数值（1.4.2）
- [ ] Stage 0 候选集合 $\mathcal Y$ 的定义与冻结
- [ ] InverTune 的会议与年份核准（当前 `[VERIFY]`）
- [ ] RVPT 的会议与年份核准（当前 `[VERIFY]`）
- [ ] BadEncoder（A6）是否投入约一周预算
