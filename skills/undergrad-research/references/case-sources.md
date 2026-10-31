# 案例来源与阅读范围

整理日期：2026-10-31。该表记录 skill 的实际依据，供查证；未来使用时以项目现状刷新事实。可复用的方法不依赖此本机路径。

## 本地依据

源项目：`D:/26秋YY高项`；核对时 Git HEAD：`ea3eca04cac6d8f8e9bf8b689e61c12397e1c08e`，工作区干净。源仓库配置为 `https://github.com/JJ704sd/26-Fall-YY-Advanced-Project-.git`；本轮没有进行远端同步或验证远端状态。下列路径均相对源项目。

| 文件 | 实际阅读/核对范围 | 用途与限制 |
| --- | --- | --- |
| README.md、analysis/README.txt、analysis/操作指南.txt | 全文 | 当前工程说明、运行方式与结论边界；状态仍要与产物核对 |
| analysis/研究推进提示词.txt | 全文 | 提炼研究推进经验；其中旧阶段的计划不当作当前待办，也不当作本轮授权 |
| 研究方案/实验设计_v1.json | 全文 | 问题、参数、对照、评价和早期计划；其 `methodology_only_not_executed` 不是当前工程状态 |
| analysis/polar_pipeline.py | parameters、区域特征、空间构图等相关实现与函数入口 | 阵元映射、归一化、数值处理与实际入口 |
| analysis/graph_models.py、analysis/tests/test_pipeline.py | 当前函数/测试入口及其与工程记录的对应 | DGI 与关键计算检查已经存在；本轮未把历史测试当作新测试结果 |
| analysis/configs/ppt_reference_4channels.json、ppt_total_diattenuation.json、ppt_similarity_edges.json | 三份完整配置 | 当前四种方法与三组单因素对照 |
| analysis/runs/运行记录索引.txt | 全文 | 识别主要实验、诊断和历史试跑 |
| analysis/reports/ppt_comparison_20261031_v6/实验结果说明.txt、comparison_summary.csv、data_alignment.csv、sources.json | 全文 | 结果、共同评价空间、已有输入对齐记录与数据来源；不是本轮重新运行实验 |
| analysis/runs/mmpd_audit_7roi_20261031_v2/metadata.json、summary.csv | 抽样/约定/容差、逐 ROI 数值 | 矩阵物理性条件和诊断范围 |
| analysis/mask_review/ppt_20261031/ | 文件清单；状态另见工程说明 | 已有原尺寸参考与自动掩膜草稿；本轮未把它们认定为人工审核 |
| 相关资料/2026高项偏振介绍_副本(1).pptx | 第 8 页文本；用户提供的第 8 至 11 页截图 | 研究目标、推荐路线/实验/评价；不是已执行结论 |
| 相关资料/5698548 (1)(1).pdf | 全 4 页文字；重点读物理第 1 至 3 页，渲染核对第 3 页 Table S1 | 确认为肝纤维化研究的补充材料；参数公式与类别，不能当作主论文或当前类器官验证 |
| 相关资料/基于全偏振显微成像的数字病理技术_中国激光 (4)(1).pdf | 物理第 1 页摘要、第 2 至 3 页基础/参数，以及第 7 至 9 页偏振超像素相关段落 | Stokes–Mueller、参数、分解与概念边界；非本项目效果证据，非声称整篇精读 |

以上文件的整理时 SHA-256 见同目录 [case-source-hashes.json](case-source-hashes.json)。仅匹配源文件版本；不是质量认证。原始 MAT 数据仍在本机，未打包进 skill。

## 原始文献与可访问性

2026-10-31 已读取 [SGC 原文页面](https://proceedings.mlr.press/v97/wu19e.html)、[GAE/VGAE 原文摘要页](https://arxiv.org/abs/1611.07308)、[DGI 原文摘要页](https://arxiv.org/abs/1809.10341) 和 [矩阵物理可实现性检查原文](https://academic.oup.com/bioinformatics/article/40/7/btae348/7687046)。这些页面支持方法定位，不等于复现全部论文。

本地补充材料给出主论文 DOI `10.1364/BOE.450294`，补充材料 DOI `10.6084/m9.figshare.19160606`；本轮网络未取得该主论文正文。中文综述 DOI 为 `10.3788/CJL231462`，相关概念以本地 PDF 为依据。SLIC 的 EPFL 项目页本轮访问超时，未把网络访问失败写成已读；案例中的具体多通道使用方式以本项目实现为准。

## 用户当前提供的材料

本轮 4 张附件分别说明研究概况、技术路线、实验方案和评价指标。已作为参考阅读，未把截图中的建议当作额外操作授权。skill 不依赖这些临时图片路径；可查项目内的演示文稿。

工具列表未提供专门的 `read_thread` 工具。本轮改用本机官方 App Server 的只读 `thread/read(includeTurns=true)`，经本地 `read_thread` 包装函数逐个读取全部 5 个引用，未续接、修改或创建聊天。实际取回完整历史后重点阅读用户请求与阶段答复，再以本地文件核对；没有把历史答复当作独立的科学验证。读取覆盖与响应哈希见 [thread-read-coverage.json](thread-read-coverage.json)。

| 聊天 ID | 读取时轮次 | 提炼并核对的经验 |
| --- | ---: | --- |
| 01a18bbf-e353-7d90-b3fe-b75a86fd0f20 | 7 | 从研究前提、仅设计方法，到用户明确要求代码实操，再到四方法对照与整理；以当前授权的执行层级交付，状态随实现更新 |
| 01a18bd8-4116-7891-be77-e10ef6d22927 | 1 | 区分综述与原始证据，围绕图表重建方法链；区分偏振超像素、空间超像素，以及结构敏感性与实际分辨能力 |
| 01a18bc0-dfae-71d2-9b88-2ac7b5da6577 | 2 | 公式、符号含义、实验阵元与归一化必须连起来说明；数值定义与相位延迟解释分别核对 |
| 01a18c05-6e02-7e51-94d1-93b58429caca | 1 | 术语沿实际流程说明；SLIC 初步分块与最终聚类类别分开；组成熵与空间排列分开 |
| 01a190c5-8ba7-77a3-988a-5703819bbcfc | 3 | 工程交接需要数据/环境边界、清单、验证与明确发布范围；历史公开授权不自动成为本轮新增内容的发布授权 |

完整响应曾短暂存于本轮临时目录，读取后已删除；skill 仅保留方法、引用 ID、覆盖信息和哈希，不附整段聊天或内部推理记录。读取后发生的新轮次不在此快照范围内。App Server 接口依据：[OpenAI 官方文档](https://learn.chatgpt.com/docs/app-server)。
