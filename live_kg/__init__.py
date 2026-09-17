"""
live_kg
--------
基于实时数据现场构建的公司知识图谱（每次生成报告时构建一张小图谱）。

本包沿用 FinDKG "先定义本体、再填充实例" 的思路（FinDKG 数据集因截止
2023-01-01、与实时数据时间错位，已不再参与流程，仅保留在 knowledgeGraph/ 作参考），
实例数据全部来自实时数据源 + 人工种子关系。

模块：
  ontology.py         本体：实体类型、关系（含两端类型约束）、事件类型枚举
  seed.py             读取并校验人工种子关系文件 seed_relations.csv
  graph.py            CompanyGraph：带本体校验的图谱构建器（networkx）
  queries.py          邻居发现（1~2 跳）与风险传导推理（确定性规则）
  event_extractor.py  LLM 从新闻中抽取事件
"""
