"""
eval
----
知识图谱消融实验（实验设计见 docs/eval-design.md，标注规范见 docs/annotation-guide.md）。

  companies.txt  15 家目标公司
  snapshot.py    录制 / 回放工具返回结果，保证三组看到完全一样的数据
  variants.py    A（无产业链）/ B（纯 tool use）/ C（知识图谱）三种配置

评估代码不进入生产链路：生产代码只提供注入点（见 usage_tracker.py 与
OrchestratorLoop 的 tool_definitions / system_prompt / tool_impls / temperature）。
"""
