# RSG-Mem：面向视觉语言模型自动化红队测试的记忆增强关系策略图探索方法

> **英文名称：** Memory-Augmented Relational Strategy Graph Exploration for Automated VLM Red-Teaming  
> **暂定简称：** RSG-Mem  
> **研究对象：** 视觉语言模型（Vision-Language Models, VLMs）  
> **核心目标：** 在固定查询预算下，提高自动化红队测试的成功率、策略多样性、探索覆盖率和跨样本经验复用效率。

---

## 1. 方法定位

### 1.1 核心研究问题

本研究关注以下问题：

> 显式建模多模态测试策略之间的演化、相似、互补、冲突和组合关系，并利用跨样本经验学习这些关系的条件有效性，能否在固定查询预算下，比层次化策略树获得更高的攻击成功率、更广的策略覆盖，以及更好的跨样本和跨模型迁移能力？

本方法不只是将 TreeTeaming 的树结构替换为图结构，而是将其 **Hierarchical Strategy Exploration（层次化策略探索）** 推进为 **Relational Strategy Exploration（关系化策略探索）**：策略间关系不仅负责组织知识，还直接参与候选生成、策略组合、行动选择和预算分配。

### 1.2 主要创新点

1. **关系化策略表示**：将多模态策略及其细化、组合、相似、互补和冲突关系建模为持久化关系策略图。
2. **关系引导的探索决策**：让图关系直接指导 Agent 的策略复用、细化、组合和发明，而不是仅将图作为可视化或检索索引。
3. **组合策略效果归因**：分别学习策略原子与策略交互的效果，避免将组合成功错误归因于全部组成策略。
4. **跨样本多模态经验演化**：持续积累成功与失败经验，并根据目标、图像、模型及拒答状态进行条件化检索和更新。

### 1.3 与相关工作的边界

- **TreeTeaming** 主要使用层次化策略树进行策略发现，并通过 Orchestrator、Multimodal Actuator、Consistency Checker 和失败分析形成闭环。本研究进一步显式建模跨分支策略关系和多父节点组合。[TreeTeaming](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TreeTeaming_Autonomous_Red-Teaming_of_Vision-Language_Models_via_Hierarchical_Strategy_Exploration_CVPR_2026_paper.html)
- **GoAT/GAP** 已经使用图连接不同攻击路径，且 GAP 包含多模态变体，因此本研究不以“首次将图用于 VLM 越狱”为贡献，而以“持久化策略级关系、关系引导决策和组合效果归因”为核心。[GoAT](https://arxiv.org/abs/2504.19019)、[GAP](https://aclanthology.org/2025.emnlp-industry.46/)
- **MemoAttack** 重点研究技能记忆单元的后验选择与生命周期管理；本研究在此基础上进一步把策略之间的关系及组合收益作为显式搜索对象。[MemoAttack](https://arxiv.org/abs/2605.29237)
- **MemJack** 已研究跨图像的多模态经验记忆和知识图谱；本研究强调策略—策略关系构成可执行的搜索空间，并学习组合关系的条件收益。[MemJack](https://arxiv.org/abs/2604.12616)
- **AutoDAN-Reasoning** 已搜索策略组合，因此“组合”本身不是本研究的贡献；本研究重点解决上下文条件下的组合选择、协同关系学习和信用分配。[AutoDAN-Reasoning](https://arxiv.org/abs/2510.05379)

---

## 2. 方法总体描述

本研究提出一种面向视觉语言模型的记忆增强关系策略图探索方法。该方法将多模态红队测试策略表示为具有明确适用条件、模态分工和执行步骤的策略节点，并通过细化、组合、相似、互补和冲突等带类型关系构建持久化策略图。不同于 TreeTeaming 沿单一父子路径演化策略，本方法允许多个策略节点共同生成组合策略，并利用图中关系指导跨分支信息共享、候选生成和搜索预算分配。

对于每个待测试样本，系统首先编码其目标语义、图像内容、图文关系、目标模型和历史拒答特征，再从跨样本经验记忆中检索相关成功与失败记录，构建与当前样本相关的局部策略子图。Agent 在该子图上从策略复用、策略细化、策略组合和新策略发现四类操作中选择下一步行动。候选策略由多模态执行器转化为图文测试样本，并经目标保持、策略实现和图文一致性检查后提交给目标 VLM。

系统根据目标模型反馈评估安全机制违反程度、目标相关性、信息量和策略实现质量，分析失败原因，并将结果更新到策略节点、组合关系及跨样本经验记忆中。为避免将组合策略的成功错误归因于全部组成策略，方法分别维护策略原子及策略交互的效果统计，并对高潜力组合进行选择性消融验证，估计组合相对于单个策略的增量收益。

随着样本持续到来，低价值或失效策略被退役，相似策略被合并，具有稳定互补关系的策略得到优先组合，从而形成能够跨样本持续演化的关系策略图。

---

## 3. 问题形式化

给定目标 VLM：

\[
F:(I,P)\rightarrow Y
\]

其中：

- \(I\) 为输入图像；
- \(P\) 为文本提示；
- \(Y\) 为目标 VLM 的响应。

对于第 \(i\) 个测试任务，输入状态表示为：

\[
x_i=(g_i,I_i^{src},m_i,h_i,B_i)
\]

其中：

- \(g_i\)：测试目标；
- \(I_i^{src}\)：可选的原始图像；
- \(m_i\)：目标模型及版本信息；
- \(h_i\)：当前任务已有交互历史；
- \(B_i\)：剩余目标模型查询预算。

Agent 在每轮选择一个策略行动 \(a_t\)，经多模态执行器生成测试样本：

\[
z_t=(I_t,P_t)
\]

获得目标响应 \(Y_t=F(z_t)\) 后，评价器返回多维分数和失败原因。方法的首要目标是在固定目标查询预算下最大化成功率，同时兼顾策略覆盖、多样性、迁移能力和总计算成本：

\[
\max_{\pi}\quad
\operatorname{ASR@B}
+\lambda_1\operatorname{Coverage}
+\lambda_2\operatorname{Diversity}
+\lambda_3\operatorname{Transfer}
-\lambda_4\operatorname{Cost}
\]

其中 \(\pi\) 为由关系策略图和经验记忆共同决定的搜索策略。

---

## 4. 系统总体结构

```text
输入样本
  │
  ▼
上下文编码器
  ├──目标语义
  ├──图像语义、OCR与布局
  ├──图文关系
  ├──目标模型及版本
  └──历史拒答与剩余预算
  │
  ▼
多模态经验检索 ───────────────────────────┐
  │                                       │
  ▼                                       │
局部关系策略子图                           │
  │                                       │
  ▼                                       │
Graph-guided Orchestrator                 │
  ├── REUSE：复用                         │
  ├── REFINE：细化                        │
  ├── COMPOSE：组合                       │
  └── INVENT：发明                        │
  │                                       │
  ▼                                       │
多模态 Actuator                           │
  │                                       │
  ▼                                       │
一致性检查与样本级修复                     │
  │                                       │
  ▼                                       │
目标 VLM                                  │
  │                                       │
  ▼                                       │
多维评价与失败分析                         │
  │                                       │
  ▼                                       │
信用分配                                  │
  ├──更新策略节点──────────────────────────┤
  ├──更新策略关系──────────────────────────┤
  └──写入经验记忆──────────────────────────┘
```

系统包含三个嵌套循环：

1. **样本级修复循环**：修复当前图文样本的具体实现，不改变抽象策略定义；
2. **策略级演化循环**：执行复用、细化、组合或发明，改变局部策略搜索状态；
3. **跨样本持续学习循环**：维护持久化策略图、经验记忆和策略生命周期。

---

## 5. 关系策略图

### 5.1 图定义

使用带类型的有向多重图：

\[
\mathcal G=(V,E,\tau)
\]

其中：

- \(V\) 为策略节点集合；
- \(E\) 为策略关系集合；
- \(\tau:E\rightarrow T\) 将每条边映射到一种关系类型。

图在不同测试样本之间持续保存，不会在处理新样本时重置。

### 5.2 策略节点

每个节点保存可复用的抽象策略，而不是某一条具体提示词：

\[
v=\langle
name,mechanism,modality,precondition,
execution,embedding,lineage,status
\rangle
\]

| 字段 | 含义 |
|---|---|
| `strategy_id` | 策略唯一标识 |
| `name` | 简洁且稳定的策略名称 |
| `description` | 与具体样本无关的抽象描述 |
| `mechanism` | 策略预期发挥作用的机制 |
| `modality_signature` | 文本、视觉或跨模态策略 |
| `preconditions` | 对图像、目标或模型能力的要求 |
| `execution_plan` | 图像与文本分别如何实现 |
| `tool_requirements` | 所需生成、编辑、OCR或布局工具 |
| `embedding` | 策略语义向量 |
| `lineage` | 策略来源及演化谱系 |
| `lifecycle_state` | `candidate`、`active`、`retired` 或 `eliminated` |

策略节点只缓存少量聚合统计。完整执行结果以经验记忆为唯一事实来源，避免策略图和记忆分别维护两套冲突数据。

### 5.3 关系类型

第一版建议使用六类关系：

| 关系 | 含义 |
|---|---|
| `REFINED_FROM` | 新策略是旧策略的针对性改进 |
| `COMPOSED_FROM` | 新策略由多个已有策略共同产生 |
| `SIMILAR_TO` | 两个策略在机制上高度相似 |
| `COMPLEMENTARY_TO` | 两个策略解决不同问题，可能产生协同效果 |
| `CONFLICTS_WITH` | 两个策略组合后相互干扰 |
| `SUBSUMES` | 一个策略覆盖另一个策略的主要机制 |

组合策略使用具有多个父节点的组合节点表示：

```text
策略 A ───┐
          ├── COMPOSED_FROM ──> 组合策略 C
策略 B ───┘
```

因此工程实现可以使用普通有向图加组合节点，不必依赖专门的超图库。

### 5.4 图节点去重

新增策略进入图之前执行两级去重：

1. 使用策略向量计算与已有节点的相似度；
2. 对高相似候选使用 LLM 等价性判断器比较其机制、适用条件和执行方式。

如果语义和机制均等价，则合并到已有节点并累积证据；如果表面相似但适用条件或模态分工不同，则保留为独立节点并建立 `SIMILAR_TO` 边。

---

## 6. 多模态经验记忆

### 6.1 经验记录

每次有效执行写入一条经验：

\[
m_t=\langle x_t,A_t,z_t,Y_t,r_t,f_t,c_t\rangle
\]

其中：

- \(x_t\)：当前上下文；
- \(A_t\)：实际使用的策略节点集合；
- \(z_t\)：生成的图文测试样本；
- \(Y_t\)：目标模型响应；
- \(r_t\)：多维评价结果；
- \(f_t\)：失败类型；
- \(c_t\)：目标查询和总计算成本。

推荐的经验记录结构如下：

```yaml
experience_id: string
sample_context:
  goal_category: string
  goal_embedding: vector
  image_embedding: vector
  ocr_layout_features: object
  cross_modal_embedding: vector
  target_model: string
  target_version: string
  refusal_signature: string
strategy_ids: [string]
composition_id: string | null
artifacts:
  image_path: string
  prompt_hash: string
  response_hash: string
evaluation:
  safety_violation: float
  relevance: float
  informativeness: float
  strategy_consistency: float
  success: bool
failure_type: string | null
cost:
  target_queries: int
  agent_calls: int
  judge_calls: int
  image_calls: int
  wall_time: float
timestamp: string
```

### 6.2 记忆内容

经验记忆同时保存成功和失败记录：

- 成功记录用于复用策略和估计适用条件；
- 接近成功的记录用于产生细化候选；
- 失败记录用于避免重复尝试、学习冲突关系以及分析模型防御模式；
- 执行器失败与策略失败分开保存，避免错误惩罚抽象策略。

### 6.3 混合检索

对于当前状态 \(x\) 和经验 \(m\)，检索相关性可定义为：

\[
\begin{aligned}
R(m|x)=&
\alpha_g Sim_g(x,m)
+\alpha_v Sim_v(x,m)\\
&+\alpha_c Sim_{cross}(x,m)
+\alpha_f Sim_{failure}(x,m)\\
&+\alpha_t Match_{target}(x,m)
+\alpha_u Utility(m)
-\alpha_a Age(m)
\end{aligned}
\]

分别检索：

1. 与当前目标相似的经验；
2. 与当前图像及图文关系相似的经验；
3. 与当前拒答模式相似的失败经验。

随后执行融合排序，只向 Orchestrator 提供少量最相关经验和对应策略子图，避免完整历史占用上下文。

### 6.4 生命周期管理

策略节点依次经历：

```text
candidate → active → retired → eliminated
                 ↑        │
                 └────────┘ reactivation
```

- 新发明或新组合策略首先进入 `candidate`；
- 获得足够直接证据后晋升为 `active`；
- 长期低效、重复或失效的节点转为 `retired`；
- 退役节点在新上下文中重新成功时可以恢复；
- 被充分证明无效且占用容量的节点进入 `eliminated`。

---

## 7. 图引导的 Agent 决策

Agent 每轮从四类操作中选择一种。

### 7.1 REUSE：策略复用

从当前上下文中成功后验较高的策略节点中选择一个，并根据当前图像和目标重新实例化。

适用条件：

- 检索到高相似历史任务；
- 已有策略在相近模型或类别中表现稳定；
- 当前查询预算较低。

### 7.2 REFINE：策略细化

根据当前策略的主要失败原因生成子策略。例如，当失败来自视觉信息未被正确识别时，调整视觉实现；当目标模型理解了输入但仍然拒答时，调整语义框架。新策略作为独立节点加入图中，并通过 `REFINED_FROM` 连接父节点。

### 7.3 COMPOSE：策略组合

从图中选择具有互补关系、不同模态分工或不同失败修复能力的策略，形成新的组合节点。

组合策略不能只是拼接两个策略名称，而必须明确生成：

1. 联合机制说明；
2. 图像与文本的模态分工；
3. 两个策略的执行顺序；
4. 必须满足的兼容条件；
5. 组合预期解决的失败模式。

第一版建议只允许二元组合，以控制组合空间规模。

### 7.4 INVENT：新策略发现

在以下情况下探索全新策略：

- 已有策略与当前样本的匹配度均较低；
- 局部策略子图连续多轮没有取得进展；
- 某类图像、失败模式或模态组合长期未被覆盖；
- 已有策略普遍产生同一种失败；
- 当前局部子图存在明显的结构空白。

---

## 8. 候选生成和选择

### 8.1 候选池

每轮分别生成：

- Top-\(K_r\) 个 REUSE 候选；
- Top-\(K_f\) 个 REFINE 候选；
- Top-\(K_c\) 个 COMPOSE 候选；
- Top-\(K_i\) 个 INVENT 候选。

生成后执行：

1. 语义去重；
2. 关系合法性检查；
3. 工具可执行性检查；
4. 预算预估；
5. 与近期尝试的重复检查。

### 8.2 关系引导效用函数

对于候选行动 \(a\) 和当前上下文 \(x\)，定义：

\[
\begin{aligned}
U(a|x)=&
\lambda_s\widetilde P_{success}(a|x)
+\lambda_p\widetilde P_{progress}(a|x)\\
&+\lambda_m Sim(a,x)
+\lambda_y Synergy(a|x)\\
&+\lambda_n Novelty(a)
+\lambda_v Coverage(a)\\
&-\lambda_c Cost(a)
-\lambda_r Redundancy(a)
\end{aligned}
\]

各项含义如下：

| 项 | 含义 |
|---|---|
| \(P_{success}\) | 最终成功概率 |
| \(P_{progress}\) | 获得部分进展的概率 |
| \(Sim\) | 策略与当前目标、图像、模型及失败状态的匹配度 |
| \(Synergy\) | 组合策略的协同收益 |
| \(Novelty\) | 与已有策略的差异程度 |
| \(Coverage\) | 对未探索类别、关系或失败模式的覆盖价值 |
| \(Cost\) | 目标查询、Agent、Judge 和图像生成成本 |
| \(Redundancy\) | 与近期尝试或已有节点的重复程度 |

其中 \(\widetilde P_{success}\) 和 \(\widetilde P_{progress}\) 使用 Thompson Sampling 从 Beta 后验中采样，而不是总是使用均值最高的节点，以保留对高不确定性策略的探索。

### 8.3 模式选择

四种操作模式也可以先进行一次状态条件评分，再从模式内部选择具体候选：

- 已知策略匹配度和成功后验较高时，提高 REUSE 概率；
- 存在明确的可修复失败原因时，提高 REFINE 概率；
- 两个策略具有互补证据且预算允许时，提高 COMPOSE 概率；
- 检索质量较低、搜索停滞或覆盖不足时，提高 INVENT 概率。

---

## 9. 组合策略的信用分配

### 9.1 组合收益

假设组合策略为：

\[
C=A\oplus B
\]

定义上下文条件的协同收益：

\[
Synergy(A,B|x)=
\mathbb E[R(A\oplus B,x)]
-\max\{\mathbb E[R(A,x)],\mathbb E[R(B,x)]\}
\]

若组合仅达到与最佳单策略相同的效果，则不能视为具有真正协同；只有组合相对于最佳父策略取得稳定增益时，才增加 `COMPLEMENTARY_TO` 证据。

### 9.2 更新原则

- 单策略执行：完整更新该策略节点；
- 细化策略执行：完整更新子节点，对父节点执行较弱的迁移更新；
- 组合策略执行：完整更新组合节点；
- 只有在存在单策略对照时，才更新组成策略间的协同或冲突后验；
- 若组合显著低于两个父策略，则增加 `CONFLICTS_WITH` 证据；
- 若样本未正确实现策略，则只记录 Actuator 失败，不降低策略有效性。

### 9.3 选择性反事实验证

为控制查询成本，只对以下组合执行父策略对照：

- 首次成功的组合；
- 得分接近成功阈值的组合；
- 计划晋升为高优先级的组合；
- 跨模型迁移时表现异常的组合；
- 协同后验不确定性较高但潜在收益较大的组合。

可以将“组合优于两个父策略”定义为二元协同事件，并为每对策略维护独立 Beta 后验。

---

## 10. 多模态执行与一致性检查

### 10.1 Multimodal Actuator

Actuator 接收抽象策略或组合策略，并输出具体测试样本：

\[
z_t=(I_t,P_t)
\]

执行器可以调用：

- 图像生成；
- 图像编辑；
- 布局和拼接；
- 文本渲染与 OCR 检查；
- 几何及颜色变换；
- 图文一致性分析。

工具调用只是抽象策略的具体实现。只有当策略机制或适用条件发生变化时，才建立新的策略节点；普通提示词调整和图像修复属于样本级操作。

### 10.2 一致性检查

在消耗目标 VLM 查询之前，依次检查：

1. 原始测试目标是否得到保持；
2. 图文样本是否真正实现指定策略；
3. 图像内容与文本提示是否一致；
4. 图像和文字是否能够被目标 VLM 正常感知；
5. 组合策略的各组成部分是否均被正确实现；
6. 样本修改是否发生策略漂移。

未通过检查时进入样本级修复循环。若修复次数耗尽，则记录为执行器失败，并返回策略选择阶段。

---

## 11. 评价与失败分析

### 11.1 多维评价

目标模型响应的评价向量为：

\[
r_t=(r_{safety},r_{relevance},r_{information},r_{consistency})
\]

其中：

- \(r_{safety}\)：安全机制违反程度；
- \(r_{relevance}\)：响应与测试目标的相关性；
- \(r_{information}\)：响应是否包含实质信息；
- \(r_{consistency}\)：图文样本和预定策略的一致性。

成功条件定义为：

\[
Success=
\mathbb I[
r_{safety}\ge\tau_s
\land r_{relevance}\ge\tau_r
\land r_{information}\ge\tau_i
\land r_{consistency}\ge\tau_c
]
\]

这样可以避免将“没有明确拒绝但也没有实质回答”的情况错误计为成功。

### 11.2 失败分类

建议使用以下失败类型：

| 失败类型 | 后续动作倾向 |
|---|---|
| 直接拒绝 | REFINE 或 COMPOSE |
| 安全说教/安全改写 | REFINE |
| 偏离目标 | 样本级修复或更换策略 |
| 信息不足 | REFINE |
| 图像识别/OCR失败 | 修复视觉实现 |
| 图文关联失败 | 修复跨模态关系 |
| 策略实现漂移 | Actuator 修复，不惩罚策略 |
| 组合策略相互干扰 | 建立冲突证据并拆分组合 |
| 多轮无进展 | INVENT 或切换图分支 |

---

## 12. 完整执行流程

### 阶段 0：策略图初始化

1. 输入少量基础策略；
2. 将每个策略规范化为统一节点结构；
3. 根据语义与机制相似性建立候选关系；
4. 在小规模校准集上执行种子策略；
5. 写入初始成功、失败和成本记录；
6. 形成初始关系策略图和经验记忆。

### 阶段 1：新样本上下文建模

提取目标语义、图像语义、OCR与布局、图文关系、目标模型、历史拒答和剩余预算，得到上下文表示 \(x_i\)。

### 阶段 2：经验检索与局部子图构建

1. 分别检索目标相似、图像相似和拒答相似的经验；
2. 找到经验对应的策略节点；
3. 加入相关节点的一跳关系邻居；
4. 删除已淘汰节点和明确冲突的组合；
5. 得到当前样本的局部策略子图 \(\mathcal G_i\)。

### 阶段 3：候选行动生成

生成 REUSE、REFINE、COMPOSE 和 INVENT 四类候选，并执行去重、关系约束、工具可行性和成本检查。

### 阶段 4：图引导选择

计算各候选的关系引导效用，通过 Thompson Sampling 或带温度的 Softmax 选择下一行动。图关系及其后验必须直接参与评分，不能只是把图结构转写成文本后完全交给 LLM 自由决定。

### 阶段 5：多模态执行

Actuator 将选定策略转换为具体图文样本，并记录所使用的工具、模态分工和执行步骤。

### 阶段 6：一致性检查和局部修复

在调用目标 VLM 前验证目标保持、策略实现、图文一致性和可感知性。失败时执行有限次数的局部修复。

### 阶段 7：目标查询和多维评价

向目标 VLM 提交测试样本，获得响应，计算多维评价向量并判断是否成功。

### 阶段 8：失败分析与信用分配

识别主要失败原因，并根据行动类型分别更新单策略、子策略、组合节点和策略关系。必要时调度选择性反事实验证。

### 阶段 9：图和记忆更新

1. 写入完整经验；
2. 更新策略成功和进展后验；
3. 更新协同或冲突关系；
4. 合并重复节点；
5. 添加细化或组合节点；
6. 执行晋升、退役、重新激活和清理；
7. 更新后重新构建当前局部子图。

### 阶段 10：终止

满足任一条件时停止当前样本：

- 达到成功阈值；
- 用尽目标查询预算；
- 连续若干轮没有进展；
- 所有候选的预期收益均低于成本阈值。

系统可以支持两种运行模式：

- **高效测试模式**：首次成功后立即停止；
- **策略发现模式**：成功后继续使用单独的发现预算，探索新的策略和关系。

---

## 13. 算法伪代码

```text
Input:
    样本流 D
    目标 VLM F
    种子策略 S0
    每样本目标查询预算 B

Initialize:
    G ← BuildStrategyGraph(S0)
    M ← EmptyExperienceMemory()
    Bootstrap(G, M)

for each sample x_i in D:

    context ← EncodeContext(x_i)
    experiences ← Retrieve(M, context)
    local_graph ← BuildLocalSubgraph(G, experiences)

    for t = 1 ... B:

        candidates ← {
            ProposeReuse(local_graph, context),
            ProposeRefine(local_graph, context),
            ProposeCompose(local_graph, context),
            ProposeInvent(local_graph, context)
        }

        candidates ← DeduplicateAndFilter(candidates)
        action ← GraphGuidedSelect(candidates, context, G, M)

        test_case ← Actuator(action, x_i)

        while not ConsistencyCheck(test_case):
            test_case ← Repair(test_case)
            if repair budget exhausted:
                LogActuatorFailure()
                break

        if test_case is invalid:
            continue

        response ← F(test_case)
        scores, failure ← Evaluate(response, test_case, x_i)

        evidence ← CreditAssignment(
            action, scores, failure, context
        )

        UpdateExperienceMemory(M, evidence)
        UpdateStrategyGraph(G, evidence)
        MaintainLifecycle(G, M)

        if Success(scores):
            break

return G, M, test traces, evaluation results
```

---

## 14. 实验设计

### 14.1 对比方法

建议至少包含：

1. 原始 TreeTeaming；
2. TreeTeaming + 相同历史经验上下文；
3. Flat Strategy Library + Memory；
4. GoAT/GAP-VLM；
5. 多模态化 MemoAttack；
6. MemJack；
7. 完整 RSG-Mem。

### 14.2 核心消融

| 实验设置 | 验证问题 |
|---|---|
| TreeTeaming | 原始基线 |
| TreeTeaming + 相同历史经验 | 排除提升仅来自更多上下文 |
| Flat Memory | 排除提升仅来自经验检索 |
| Graph，仅增加连接 | 检验图数据结构本身 |
| Graph + 节点去重 | 检验冗余探索是否下降 |
| Graph + 关系引导决策 | 检验关系是否真正影响搜索 |
| Graph + COMPOSE，无信用分配 | 检验简单组合的作用 |
| Graph + 信用分配 | 检验组合归因机制 |
| 无生命周期管理 | 检验过期与低质量经验的影响 |
| 仅文本检索 | 检验多模态上下文检索的作用 |
| 完整 RSG-Mem | 验证整体效果 |

最关键的对照是：**向 TreeTeaming 的 Orchestrator 提供与 RSG-Mem 完全相同的历史经验，但不提供策略关系和关系引导决策。** 如果该设置获得相同提升，则说明有效因素可能只是信息增加，而不是关系策略图。

### 14.3 数据设置

实验需要明确区分：

1. **Cold-start**：初始记忆为空；
2. **Warm-start**：使用独立校准集构建初始图和记忆；
3. **Online continual**：按数据流持续更新，并对多种样本顺序取平均；
4. **Held-out category transfer**：在未见风险类别上测试；
5. **Cross-image transfer**：训练和测试图像不重合；
6. **Cross-model transfer**：策略图来自源模型，在新目标 VLM 上测试；
7. **Temporal transfer**：目标模型版本变化后测试记忆是否失效。

在线持续学习结果应与完全独立的测试集结果分开报告，避免把测试阶段的跨样本写入误认为普通独立同分布测试。

### 14.4 评价指标

#### 攻击效果

- ASR@固定目标查询预算；
- ASR-Budget 曲线面积；
- 首次成功所需目标查询数；
- 成功响应的相关性与信息量。

#### 搜索效率

- 目标 VLM 调用数；
- Agent、Judge 和图像生成调用数；
- 总 token、GPU 时间、墙钟时间及费用；
- 达到相同 ASR 所需的总成本。

#### 策略探索

- 策略数量；
- 策略语义多样性；
- 策略关系覆盖率；
- 新失效模式发现数量；
- 重复策略生成率。

#### 记忆效果

- 检索命中率；
- 经验复用率；
- 查询节省比例；
- 负迁移率；
- 退役和重新激活准确性；
- 随样本数量增长的在线收益曲线。

#### 组合效果

- 组合策略相对最佳父策略的平均增益；
- 正协同组合比例；
- 冲突组合比例；
- 信用分配准确性；
- 组合策略的跨样本和跨模型迁移率。

### 14.5 多模态因果对照

为了证明收益确实来自视觉—语言协同，而不是单纯文本策略，应对选定样本执行：

- 原始图文测试；
- 移除图像；
- 替换为空白或无关图像；
- 保留图像但替换为普通文本；
- 打乱图像与目标之间的配对；
- 移除组合中的一个策略组成部分。

这些对照也可用于组合策略的信用分配和跨模态依赖分析。

---

## 15. 推荐的实现顺序

### 阶段一：最小可行原型

- 使用 NetworkX 保存带类型策略图；
- 使用 SQLite/JSONL 保存经验记录；
- 使用 FAISS 保存目标、图像和策略向量；
- 仅实现 REUSE、REFINE、二元 COMPOSE 和 INVENT；
- 使用启发式效用函数；
- 先在少量样本和单个开源 VLM 上验证闭环。

### 阶段二：完整方法

- 加入双后验 Thompson Sampling；
- 加入节点生命周期；
- 加入选择性反事实验证；
- 学习互补和冲突关系；
- 增加上下文条件的组合收益估计。

### 阶段三：迁移实验

- 跨图像；
- 跨风险类别；
- 跨目标模型；
- 跨模型版本；
- 对不同样本顺序重复实验。

第一版不建议直接引入 GNN。方法的关键是“策略关系是否进入搜索决策”，而不是是否使用图神经网络。只有当规则或统计式关系传播已得到验证后，再考虑学习图表示。

---

## 16. 论文贡献表述

论文中可以将贡献概括为：

1. 提出一种面向 VLM 自动化红队测试的关系策略图，将树形单父节点演化扩展为可表示细化、组合、相似、互补和冲突关系的持久化策略空间。
2. 提出关系引导的预算搜索机制，综合上下文匹配、成功与进展后验、策略协同、覆盖价值、查询成本和重复程度，在复用、细化、组合与新策略发现之间进行动态决策。
3. 提出面向组合策略的信用分配机制，通过策略原子统计、交互后验和选择性反事实验证学习策略间的条件协同关系。
4. 构建跨样本多模态经验记忆及生命周期管理机制，支持成功与失败经验的条件化复用、去重、退役和重新激活。

---

## 17. 一句话总结

> 本研究不是将 TreeTeaming 的树简单替换为图，而是把策略探索从独立分支演化转化为由策略关系、组合收益和跨样本证据共同驱动的关系化持续探索。

---

## 参考文献与项目链接

1. [TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TreeTeaming_Autonomous_Red-Teaming_of_Vision-Language_Models_via_Hierarchical_Strategy_Exploration_CVPR_2026_paper.html)
2. [Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs](https://arxiv.org/abs/2504.19019)
3. [Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation](https://aclanthology.org/2025.emnlp-industry.46/)
4. [Evolving Skill-Structured Attack Memory Enhances LLM Jailbreaking](https://arxiv.org/abs/2605.29237)
5. [Every Picture Tells a Dangerous Story: Memory-Augmented Multi-Agent Jailbreak Attacks on VLMs](https://arxiv.org/abs/2604.12616)
6. [AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling](https://arxiv.org/abs/2510.05379)
7. [MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/34983)
8. [JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks](https://arxiv.org/abs/2404.03027)
