# RSG-Mem Implementation

- **Changed branch:** `main`
- **Changed field:** `Implementation status`
- **Implementation status:** complete research prototype
- **Baseline commit:** `2c78457b898708f477e9c8514e1155e554edd852`

## 已实现方法

本仓库实现了 **RSG-Mem（Memory-Augmented Relational Strategy Graph）**：以可持久化的关系策略图代替严格策略树，用跨样本多模态经验记忆为图上的 `REUSE`、`REFINE`、`COMPOSE`、`INVENT` 四类操作提供检索证据、成功/进展后验和组合协同后验，在固定目标查询预算内完成候选生成、关系约束、Thompson Sampling 选择、图文样本执行、一致性修复、目标模型查询、多维评价、信用分配以及节点生命周期更新。

## 模块映射

| 方法环节 | 实现 |
|---|---|
| 关系策略图与六类边 | `src/rsg_mem/graph.py` |
| 跨样本经验、混合检索、双 Beta 后验 | `src/rsg_mem/memory.py` |
| 四操作候选池与关系引导效用函数 | `src/rsg_mem/controller.py` |
| 组合相对父策略的信用与协同/冲突更新 | `src/rsg_mem/credit.py` |
| 图文工具执行与产物溯源 | `src/rsg_mem/actuator.py` |
| 晋升、退役、重激活和容量清理 | `src/rsg_mem/lifecycle.py` |
| 完整在线迭代与预算控制 | `src/rsg_mem/engine.py` |
| Mock 与 OpenAI-compatible 适配器 | `src/rsg_mem/providers/` |
| 配置、CLI、种子图与样例 | `configs/`, `examples/`, `src/rsg_mem/cli.py` |
| 离线单元与端到端验证 | `tests/` |

## 核心实现约束

1. 策略节点保存样本无关的机制描述，具体提示词和图像编辑仅作为经验记录；
2. 组合节点保留多父节点谱系，互补与冲突关系直接进入候选效用计算；
3. 只有存在单策略历史对照时才更新组合协同后验，缺少对照时标记为 deferred；
4. 未通过一致性检查的样本在目标查询前修复或拒绝，不对策略有效性写入负反馈；
5. 目标查询、Agent、Judge、图像调用和墙钟耗时分别计费；
6. 图快照、SQLite 记忆、JSONL 轨迹和汇总结果均可重开复用。

## 可复现实验入口

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m rsg_mem demo --output runs/demo
PYTHONPATH=src python3 -m rsg_mem run \
  --config configs/mock.yaml \
  --samples examples/samples.jsonl \
  --seeds examples/seeds.yaml
```

完整方法定义见 [`docs/METHOD.md`](docs/METHOD.md)，运行说明见 [`README.md`](README.md)。
