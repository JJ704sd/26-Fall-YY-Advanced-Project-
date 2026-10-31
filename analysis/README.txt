类器官偏振图像研究 工程说明
更新日期：2026-10-31

本工程针对7个类器官切片ROI，比较偏振区域特征与轻量图方法，研究区域组成及空间分布。当前结果是同一批数据上的探索性实验；自动组织掩膜、样本关系、偏振标定和组织学解释均需继续核对。

一、目录和入口

  类器官\1 至 类器官\7            原始数据，实验程序只读取
  相关资料、研究方案                用户资料及已有方案
  analysis\polar_pipeline.py       数据处理、SLIC、构图、聚类与评价入口
  analysis\graph_models.py          DGI图表示学习实现
  analysis\experiments\            完整矩阵物理性检查和实验汇总工具
  analysis\mask_review\            原尺寸参考图与待核对的自动掩膜草稿
  analysis\configs\                每组实验的可复现设置
  analysis\runs\                   每次运行单独输出配置、图像、数据及日志；用途见运行记录索引.txt
  analysis\reports\                跨实验汇总、图表和解释
  analysis\tests\                  公式、构图、模型和指标的关键检查
  analysis\操作指南.txt             环境、运行命令和输出说明
  analysis\研究推进提示词.txt       可以交给AI继续研究的任务说明

配置、随机种子、软件版本、数据文件信息和代码SHA-256分别记录在每次运行目录；已有运行目录不会被程序覆盖。原始MAT文件和此前试验结果没有移动。

二、为什么选择这些步骤

1. 偏振参数：PL、DL、qL、rL来自缪勒阵元，便于解释不同类型的偏振变化。D=sqrt(m12²+m13²+m14²)是总二向色性，DL只使用m12、m13，对应线性部分。本批数据中各ROI的超像素均值D与DL高度相关，加入D能否带来额外价值，需要用特征消融检验；不能把qL、rL等同于相位延迟。

2. SLIC和区域特征：先用四个基础偏振通道划分超像素，把像素压缩成相邻的小区域，再提取各通道的均值和标准差。加入D时保留原来的SLIC输入，才能单独观察D对节点特征的影响。SLIC输入做统一分位缩放只是为了分割；特征统计仍使用未按显示范围裁剪的参数数值。

3. 邻接图：接触的超像素建立空间边。等权边便于观察单纯加入邻域信息的效果；相似度边按标准化区域特征的距离赋权，使差异大的邻居影响较小。两种图都不连接不同ROI。

4. 四组方法：
   baseline：直接按区域特征聚类，用来判断原始偏振特征本身的作用。
   propagation：SGC式固定图传播再聚类，用来观察邻域平滑；过多传播可能模糊边界。
   gae：学习节点表示以重建图中的连接关系，利用可训练模型表达局部结构；重建连接好不等同于病理识别正确。
   dgi：在各ROI内部打乱节点特征，通过真实与扰动表示和图摘要的对比学习节点表示。它与GAE的训练目标不同，仍需检查随机种子稳定性。
   VGAE是可选的变分图自编码器，当前未实现。其他聚类算法如高斯混合模型也可后续比较，但不能将其作为已有结果。

5. 评价：轮廓系数和不同种子之间的ARI分别衡量特征分离和稳定性。面积按分析网格上的像素计算；熵描述组成多样性，邻接和连通性描述空间分布。类别编号在方法间无固定对应，指标也不是病理准确率。

三、本次受控实验及结果

运行配置：
  configs\ppt_reference_4channels.json       四参数、等权空间边
  configs\ppt_total_diattenuation.json       四参数加D、等权空间边
  configs\ppt_similarity_edges.json          四参数、偏振相似度边权

三组都使用7个ROI、5个随机种子、K=4、约500个超像素、GAE和DGI各训练100轮。比较前已逐ROI确认掩膜、SLIC标签和四个基础参数图完全相同。为了可比，轮廓系数统一在原始四参数的8维区域特征空间计算。

结果与图表：
  reports\ppt_comparison_20261031_v6\实验结果说明.txt
  reports\ppt_comparison_20261031_v6\comparison_summary.csv
  reports\ppt_comparison_20261031_v6\comparison_per_roi.csv
  reports\ppt_comparison_20261031_v6\method_comparison.png
  reports\ppt_comparison_20261031_v6\data_alignment.csv

在当前自动掩膜和参数设置下，加入相似度边权提高了三种图方法在7个ROI上的共同空间轮廓系数描述性平均，但不能据此认为组织识别更准确。加入D的结果与原四参数总体接近；D、DL的高相关性提示两者在本批数据中可能存在较多重复信息。真实优势仍需掩膜复核、设置敏感性分析和结构参考。

四、退偏与延迟的计算前提

PPT中提出的退偏和延迟需要完整缪勒矩阵分解。使用 muellerkit 0.1.2 按 I,Q,U,V 顺序进行物理性检查，并在7个未审核的自动掩膜内按原始图像每16个像素取样：24950个取样点中393个通过严格的相干矩阵半正定条件，约1.58%。这一数值依赖采样、掩膜、Stokes约定和容差，不能外推到全部像素，也不能单独确定原因。

对于未经筛选的分解，约49.80%的取样点具有大于0.001的延迟子矩阵正交性误差；该阈值仅用于数值诊断。矩阵重构误差接近零并不保证分解结果有可靠物理意义。因此，当前没有把退偏和延迟作为有效特征送入聚类；先向数据提供方核对仪器标定、阵元顺序、偏振约定和噪声，再决定是否使用物理约束估计或其他去噪方法。任何修正都要保留原矩阵并报告修正幅度。

诊断目录：runs\mmpd_audit_7roi_20261031_v2；代码：experiments\mmpd_diagnostics.py。summary.csv给出每个ROI的取样数、通过率、最小特征值及分解质量，report.html用于查看空间分布。未筛选的分解图仅供排查。

另在排除图像边缘后的同一批24862个取样点上，比较16个阵元分别做1×1、3×3和7×7局部平均。严格物理性检查通过率依次为1.58%、2.47%和4.43%。局部平均会改变原始矩阵并可能混合组织、腔隙和背景；通过率上升不能证明数据已经校准。逐ROI改变量和通过率见 runs\mueller_neighborhood_7roi_20261031，代码见 experiments\mueller_neighborhood_check.py。

五、运行与核查

  Set-Location -LiteralPath 'D:\26秋YY高项'
  & '.\analysis\.venv\Scripts\python.exe' -m pytest '.\analysis\tests' -q
  & '.\analysis\.venv\Scripts\python.exe' '.\analysis\polar_pipeline.py' --config '.\analysis\configs\ppt_reference_4channels.json'
  & '.\analysis\.venv\Scripts\python.exe' '.\analysis\polar_pipeline.py' --config '.\analysis\configs\ppt_total_diattenuation.json'
  & '.\analysis\.venv\Scripts\python.exe' '.\analysis\polar_pipeline.py' --config '.\analysis\configs\ppt_similarity_edges.json'
  & '.\analysis\.venv\Scripts\python.exe' '.\analysis\experiments\mmpd_diagnostics.py'
  & '.\analysis\.venv\Scripts\python.exe' '.\analysis\experiments\mueller_neighborhood_check.py'

如需将新生成的三次运行合并，给它们分别传入唯一的 --run-name，再将名称按“四参数、四参数加D、四参数加权图”顺序传入：

  & '.\analysis\.venv\Scripts\python.exe' '.\analysis\experiments\summarize_comparison.py' --runs 名称1 名称2 名称3 --audit 新诊断目录名 --neighborhood 新平均实验目录名 --name 新汇总目录名

资料：
  PPT：相关资料\2026高项偏振介绍_副本(1).pptx，第8至11页。项目内副本与用户提供的文件SHA-256一致：fcc537387580d8d745c3237ae124aca655fee726af9a3319a11a1ae6f9c1d4cf。
  SLIC：https://www.epfl.ch/labs/ivrl/research/slic-superpixels/
  SGC：https://proceedings.mlr.press/v97/wu19e.html
  GAE：https://arxiv.org/abs/1611.07308
  DGI：https://arxiv.org/abs/1809.10341
  Lu-Chipman：https://opg.optica.org/josaa/abstract.cfm?uri=josaa-13-5-1106
  物理性检查：https://academic.oup.com/bioinformatics/article/40/7/btae348/7687046
  分解代码库：https://pypi.org/project/muellerkit/0.1.2/
