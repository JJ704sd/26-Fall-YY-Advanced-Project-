# 26-Fall-YY-Advanced-Project-
26-Fall-YY-Advanced-Project​ 

## 项目概况

本项目探索乳腺癌类器官切片的偏振图像分析，对比区域特征与图方法。现有结果来自同一批 7 个 ROI 的探索性实验；自动组织掩膜、样本间关系、偏振标定及组织学解释仍需核对。聚类类别不代表病理名称，也不是诊断准确率。

## 目录

- `analysis/`：分析代码、实验配置、测试、掩膜复核草稿、运行记录和结果汇总。研究背景及结果说明见 `analysis/README.txt`，运行步骤见 `analysis/操作指南.txt`。
- `analysis/runs/`：各次运行的配置、指标、图像、数组、模型权重与诊断记录；用途见该目录中的运行记录索引。
- `analysis/reports/ppt_comparison_20261031_v6/`：三组主要对照实验的汇总表、图和解释。
- `研究方案/`：研究方案、实验设计和数据目录记录。
- `相关资料/`：项目参考资料。

## 原始数据和环境

本仓库不包含 `类器官/` 下的原始采集文件（约 6.4 GiB，部分文件超过 GitHub 普通文件大小限制），也不包含本机 `analysis/.venv/`。原始文件仍保留在本地；`analysis/原始数据_SHA256清单.csv` 记录各文件的相对路径、字节数和 SHA-256，供核对数据版本使用。仓库中的实验图像和数组是基于原始数据生成的结果，不等同于完整的原始数据。

如需重新运行分析，请通过项目允许的数据渠道取得对应原始文件，并按清单还原到项目根目录的 `类器官/` 下。使用 Python 3.11，在项目根目录创建环境并安装依赖：

```powershell
python -m venv .\analysis\.venv
& .\analysis\.venv\Scripts\python.exe -m pip install -r .\analysis\requirements.txt
& .\analysis\.venv\Scripts\python.exe -m pytest .\analysis\tests -q
```

完整实验命令、数据假设和结果限制见 `analysis/README.txt` 与 `analysis/操作指南.txt`。依赖和执行环境的历史记录保存在各运行目录中。

## 研究辅助 skill

`skills/undergrad-research/` 提供大学生高级项目研究辅助：问题与前提梳理、论文和公式学习、数据检查、受控实验、结果解释与阶段汇报。通用工作方法与本项目偏振案例分开，附项目简报、论文卡、实验记录、阶段汇报四个模板；项目案例和聊天读取范围均有来源记录。

安装后的调用示例：

```text
请使用 $undergrad-research，结合当前项目材料，核对已有结果，推进本轮可完成的工作，并记录证据和下一步完成判据。
```

首次试用的实际项目产物见 `研究方案/高级项目研究推进简报_20261031.md`。skill 不附原始采集数据、论文全文或聊天全文；案例快照不替代使用时的项目核查。
