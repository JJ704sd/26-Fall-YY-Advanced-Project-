from pathlib import Path
import csv
import json
import re
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PDF = OUT / '高项研究路径_v1.pdf'
pdfmetrics.registerFont(TTFont('Song', 'C:/Windows/Fonts/simsun.ttc'))
pdfmetrics.registerFont(TTFont('Hei', 'C:/Windows/Fonts/simhei.ttf'))
BLUE = colors.HexColor('#234B80')
GRAY = colors.HexColor('#586474')
styles = {
    'title': ParagraphStyle('title', fontName='Hei', fontSize=21, leading=29, textColor=BLUE, spaceAfter=10),
    'sub': ParagraphStyle('sub', fontName='Song', fontSize=10, leading=16, textColor=GRAY, spaceAfter=10),
    'h': ParagraphStyle('h', fontName='Hei', fontSize=14, leading=21, textColor=BLUE, spaceBefore=8, spaceAfter=8),
    'body': ParagraphStyle('body', fontName='Song', fontSize=10.2, leading=16.5, wordWrap='CJK', spaceAfter=7),
    'small': ParagraphStyle('small', fontName='Song', fontSize=8.8, leading=13.4, wordWrap='CJK', spaceAfter=5),
    'th': ParagraphStyle('th', fontName='Hei', fontSize=9.2, leading=14, textColor=colors.white, wordWrap='CJK'),
    'td': ParagraphStyle('td', fontName='Song', fontSize=9.1, leading=14.4, wordWrap='CJK'),
    'code': ParagraphStyle('code', fontName='Song', fontSize=9.4, leading=15, wordWrap='CJK', backColor=colors.HexColor('#F1F5FA'), borderPadding=7, spaceAfter=9),
}
story = []


def p(text, kind='body'):
    story.append(Paragraph(escape(text).replace('\n', '<br/>'), styles[kind]))


def heading(text):
    p(text, 'h')


def table(headers, rows, widths):
    data = [[Paragraph(escape(str(x)), styles['th']) for x in headers]]
    data += [[Paragraph(escape(str(x)), styles['td']) for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F1F5FA'), colors.white]),
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor('#CCD5DF')),
    ]))
    story.extend([t, Spacer(1, 9)])


def new_page(title):
    story.append(PageBreak())
    heading(title)


W = A4[0] - 92
p('乳腺癌类器官偏振图像研究路径', 'title')
p('方法论方案 第一版    2026年10月30日\n本版完成研究设计与实操安排 尚未进行图像分析 模型训练或效果评测', 'sub')
p('研究目标是利用多参数偏振图像，将类器官切片中的组织划分为具有不同偏振特征的区域，量化各区域的组成和空间分布，并通过对照实验判断图信息及图表示学习是否有帮助。')
heading('一 研究范围与总体路线')
p('已核对现有7个编号目录的文件头和变量目录：每个 FinalMM.mat 均列出 FinalM11 至 FinalM44 共16个数值阵元，以及 CalibratedM11；数值数组尺寸均为2008×2408。尚未核验全部像素的有效性、标定质量和归一化状态。')
p('7份 info.txt 的采集路径均包含 BR-029，暂按同一来源的7个 ROI 组织探索分析。独立患者及类器官数量需要向数据提供方确认，不能把7个 ROI 直接当作7名患者。')
p('文件与参数确认 → 组织掩膜与质量检查 → 偏振参数计算 → SLIC超像素与区域特征 → 空间图构建 → 三组方法对照 → 区域解释与异质性分析', 'code')
table(['首版选择', '具体安排'], [
    ['计算平台', '本地 Python 为主，VS Code 或 JupyterLab 查看过程；GPU用于后续GAE训练。'],
    ['起始参数', 'PL、DL、qL、rL 四个参数；进一步计算退偏和相位延迟作为后续扩展。'],
    ['比较方法', '直接K-Means、SGC式特征传播后K-Means、GAE后K-Means。'],
    ['分析范围', '二维切片内的偏振区域与空间分布；以区域类别A、B、C等作初始命名。'],
], [94, W-94])
heading('本版可以明确什么')
p('可以明确数据进入方法的方式、模块之间的关系、实验对照和结果解释边界。后续实操预计能生成参数热图、区域分布图、类别特征表、面积占比和空间指标。是否具有病理对应关系、图方法是否优于基础聚类，需要实验结果及参考标注支持。')

new_page('二 数据准备与偏振参数')
p('先统一斯托克斯分量顺序和阵元下标约定。以下采用1起始下标，并以归一化阵元 m_ij 表示公式输入。[1][2]')
table(['参数', '计算公式', '在本项目中的解释'], [
    ['PL', 'sqrt(m21² + m31²)', '线性起偏能力，描述样本将非偏振入射光转化为线偏振成分的幅值。'],
    ['DL', 'sqrt(m12² + m13²)', '线性二向色性，描述不同线偏振入射状态的衰减差异幅值。'],
    ['qL', 'sqrt(m42² + m43²)', '与双折射有关的线性偏振向圆偏振转换幅值。'],
    ['rL', 'sqrt(m24² + m34²)', '与双折射有关的圆偏振向线性偏振转换幅值。'],
], [38, 145, W-183])
p('如果输入仍为未归一化矩阵，应按约定先取 m_ij = M_ij / M11；如果采集流程已经完成该步骤，则直接使用相应阵元。FinalM11 的变量名本身不能证明归一化是否完成。CalibratedM11 单独作为光强与结构查看通道，不与 FinalM11 默认互换。')
p('qL、rL 不直接等于相位延迟 δ。首版使用“偏振差异”或“与双折射有关的耦合差异”描述结果。退偏、延迟及方向参数的计算，需要后续增加经核对的矩阵分解过程；方向参数还需处理角度周期性。')
heading('数据进入流程前的检查')
table(['检查项', '处理安排与通过条件'], [
    ['文件和变量', '只按名称读取需要的数值阵元；检查尺寸、缺失和非有限值。带(1)的副本先不计作新增样本。'],
    ['空间对应', '确认各阵元与光强图逐像素对应，保留坐标、裁剪和缩放记录。'],
    ['组织掩膜', '首轮建立经人工查看的组织实体掩膜；空白、培养基质与腔隙分别说明处理方式，腔隙可另作形态统计。'],
    ['标定和尺度', '确认切片厚度与像素尺寸。采集路径虽含20X，仍不能据此推定微米/像素。'],
    ['参考信息', '收集同一区域的明场、H&E或专家意见；相邻切片配准只能支持与配准精度相称的区域核对。'],
], [76, W-76])
p('本批数据已存在上述变量目录，但完整数值质量尚未验证。MAT文件还包含MATLAB对象元数据，后续应显式指定变量名读取，避免把对象元数据当作图像。', 'small')

new_page('三 超像素 区域特征与空间图')
heading('预处理与SLIC划分')
p('先计算参数并建立有效组织掩膜，再为模型输入做统一的特征缩放。保留未经显示裁剪的参数数值；热图显示范围与模型特征处理分别记录。跨ROI比较时，使用共同的特征处理规则，避免每幅图单独拉伸后掩盖幅值差异。')
p('首轮以4个偏振通道进行多通道SLIC，并以光强通道的SLIC作为敏感性比较。多参数图不是RGB颜色，使用 scikit-image 时明确关闭 convert2lab；设置组织mask并保持区域连通。[3]')
table(['项目', '起始设计'], [
    ['超像素尺度', '主设置约1000个/ROI；补充比较500和1500。实际数量受掩膜形状影响。'],
    ['紧凑度', '主设置1；仅在检查边界时比较0.1和10。先固定输入缩放，再讨论该参数。'],
    ['核心节点特征', 'PL、DL、qL、rL 的区域均值与标准差，共8维。'],
    ['可选特征', '后续逐组增加光强纹理、局部方向等，比较是否提供额外信息。'],
], [90, W-90])
p('SLIC的节点数量与最终聚类数K含义不同。节点面积、圆度会受到SLIC规则影响，因此首版不把它们直接当作组织形态；组织形态主要在最终类别的连通区域上统计。')
heading('构图规则')
p('每个超像素是一个节点，共享边界的两个超像素之间建立无向边。每个ROI单独构图，不连接不同ROI的空间位置。GAE使用共享编码器，之后在共同表示空间内统一聚类。')
p('图G0：空间邻接边等权。图G1：仍只连接空间邻居，以偏振特征相似度设置边权，例如 w_ij = exp(-||x_i-x_j||² / (2σ²))。σ可取相邻节点特征距离的中位数；距离退化时需先检查特征，而非静默加入任意常数。')
p('传播时为节点加入自环并归一化邻接矩阵。相似度应在统一缩放后的核心特征上计算；边权不能当作病理相似性的既定真值。')
heading('这一阶段的检查标准')
p('在原图上查看掩膜、超像素边界和图连接。确认小结构未被大范围合并、背景未混入节点、分离组织片段之间没有错误连边。若边界明显不合理，先调整划分和掩膜，再比较模型。')

new_page('四 方法对照与实验设置')
table(['方法', '如何使用', '主要回答的问题'], [
    ['M0 基础聚类', '统一标准化后的8维区域特征直接进入K-Means。', '已有偏振特征能区分哪些区域？'],
    ['M1 图传播', 'Z = A_norm^t X；先用t=2，再比较t=1、3；随后K-Means。', '邻域信息是否有帮助，是否出现过度平滑？'],
    ['M2 图自编码器', '两层GCN，8→32→16维；利用图连接重建目标学习表示，再K-Means。', '可学习的图表示是否提供进一步价值？'],
], [78, 220, W-298])
p('M1借用SGC的特征传播思想，不使用原论文中的监督分类器。GAE参考标准编码器与内积解码器；编码器可使用相似度边权，重建目标先明确为二元空间连接。重建损失下降只说明模型更符合构图目标，不能直接证明区域符合组织结构。[4][5]')
heading('首轮采用的小规模实验')
table(['设置', '安排'], [
    ['聚类数K', '以K=4作为起始工程设置，比较K=3、5、6；不预设存在4种真实组织成分。'],
    ['共同条件', '相同ROI、掩膜、超像素、核心特征和评价单位；K-Means统一使用n_init=20。'],
    ['随机性', '固定并记录5个随机种子：0、1、2、3、4；报告稳定性和均值、离散程度。'],
    ['GAE起始设置', '150轮，Adam学习率0.001，16维表示；稀疏传播，按边采样负例，避免全像素构图。'],
    ['跨ROI比较', '7个图共享编码器；同一种方法统一拟合聚类中心，保持跨ROI类别定义一致。'],
], [88, W-88])
p('先在G0等权空间图上比较M0、M1、M2；方法能稳定运行后，再固定其他设置，将G0替换为G1以单独观察相似度边权的作用。超像素尺度、K和传播次数分别改变，首轮不展开所有参数的组合搜索。')
heading('实验结果如何取舍')
p('不预设GAE一定优于直接聚类。只有当特征区分、稳定性、边界保留及参考结构核对形成一致证据时，才认为某种方法有优势。不同方法的类别编号不可直接对照；展示比较时可对齐颜色，ARI本身不受编号置换影响。')
p('DGI和VGAE列为后续扩展。首版先把上述三组比较讲清楚，再决定是否增加模型，以便将改进归因于具体环节。')

new_page('五 评价指标与预期效果')
table(['评价内容', '计算与解释'], [
    ['聚类区分程度', '每个ROI报告轮廓系数；主要比较共同基础特征空间中的分数，另列各方法自身表示空间中的分数。'],
    ['聚类稳定性', '同一ROI、同一节点划分下，对5次运行计算两两ARI。超像素设置改变时，需映射回共同像素区域再比较。'],
    ['区域组成', '类别面积占比 p_c = 该类像素数/有效组织像素数，按面积加权，不能以节点比例代替。'],
    ['组成多样性', 'H = -Σ p_c ln(p_c)，可同时列H/ln(K)。主要在相同K及掩膜规则下比较。'],
    ['空间分布', '统计类别连通域大小、数量、单位面积边界长度及类别接触关系；邻接关系优先按接触边界长度加权。'],
    ['组织解释', '展示各类参数分布和空间位置；结合原图、配准H&E或专家意见解释，有参考标注后再讨论一致性指标。'],
    ['计算开销', '实操时记录预处理时间、训练时间、内存和显存峰值。当前没有速度或精度实测。'],
], [86, W-86])
p('Shannon熵反映组成是否均衡，不直接反映空间排列。两类各占50%时，无论左右分块还是交错分布，其熵相同，因此必须同时报告空间指标。区域更连续也可能是平滑过强，不能把边界越少直接视为越好。')
heading('首版预期可以得到的结果')
p('技术输出：每个ROI的4种参数热图、组织与超像素检查图、3种方法的区域分布图、各类特征概况，以及面积占比和空间指标表。跨ROI采用一致处理与类别定义，可比较同一来源不同视野的组成差异。')
p('研究结论：可检验邻接关系、相似度边权和图表示学习分别带来的变化，并找出对随机初始化及参数设置较敏感的环节。')
p('生物学解释：能否将某类区域对应到细胞、纤维或其他结构，要以参考图像和专家核对为依据。二维切片指标不直接代表完整类器官三维结构，也暂不支持诊断准确率、癌变分级或临床效能的结论。')
p('本轮将7个ROI共同用于探索，属于当前数据内的分析。若后续评估对新样本的适用性，应按真实的患者或类器官来源划分，并只用开发数据拟合标准化、编码器和聚类中心。', 'small')

new_page('六 计算平台与后续实操安排')
table(['平台', '使用方式与选择理由'], [
    ['本地Python 主选', 'NumPy/SciPy读取数值阵元，scikit-image完成掩膜与SLIC，scikit-learn完成标准化和聚类，PyTorch完成GAE。VS Code或JupyterLab用于查看过程。'],
    ['实验室工作站', '可迁移相同Python环境和配置，适合后续增加样本；将图像预处理和图模型训练分开执行。'],
    ['Google Colab 备选', '适合教学、共享Notebook或补充GPU实验。资源与运行时不保证固定，是否使用以实际访问和资源条件为准。[6]'],
], [98, W-98])
p('本机已检测到约31.7 GiB内存、RTX 4060 Laptop GPU约8 GB显存，现有Python 3.11与支持CUDA的PyTorch。按每ROI约1000个超像素设计，初步判断可以在本地开展该规模研究；实际耗时和峰值占用仍需后续测量。完整图像处理以CPU逐ROI完成，GPU主要接收节点特征和稀疏图。')
p('后续按项目建立独立环境并记录依赖版本，避免直接修改全局环境。本轮已停止未完成的环境安装，没有执行参数计算、分割、模型训练或效果评测。')
table(['阶段', '完成内容', '进入下一阶段的条件'], [
    ['1 数据与基础图像', '确认归一化、坐标和像素尺度，生成参数图和经查看的组织掩膜。', '公式和变量对应明确，掩膜及通道无明显异常。'],
    ['2 基础区域划分', '完成SLIC、8维区域特征及M0聚类，输出7个ROI检查图。', '区域划分可解释，特征不被背景或异常值主导。'],
    ['3 图方法对照', '完成M1和M2，固定共同条件并重复运行；再检验边权。', '三组结果和日志齐全，评价口径相同。'],
    ['4 解释与整理', '汇总空间指标、参考图核对和限制，决定是否补算退偏/延迟或增加DGI。', '每项研究结论均能找到图表或实验记录支持。'],
], [77, 228, W-305])
p('开始实操前优先确认：FinalMij是否已归一化；7个ROI的真实样本关系；组织与腔隙的统计范围；是否有对应参考图像和像素标定。这些信息影响解释和比较方式，数据目录核对及方案准备可以先完成。')

new_page('七 方法模块与资料依据')
p('下面是后续实操的模块划分。本版定义输入与输出，尚未实现或运行这些模块。实验配置文件中的数值都是待验证的起始设置。')
table(['模块', '输入', '输出'], [
    ['数据与参数', 'FinalMM.mat、m11.mat及采集说明', '参数数组、有效像素与质量记录'],
    ['掩膜与超像素', '光强图、参数图、人工检查的组织范围', '掩膜、超像素标签及检查图'],
    ['特征与空间图', '参数数组、超像素标签', '节点特征、节点面积、空间边及权重'],
    ['表示与聚类', '统一处理的特征、图和实验配置', '三组类别标签、表示、聚类中心及日志'],
    ['评价与解释', '类别图、特征、参考图和样本关系', '稳定性、面积与空间指标、可核对图表'],
], [95, 200, W-295])
heading('资料与证据说明')
p('项目范围来自用户提供的4张说明图；文件结构和样本代号来自当前数据目录及info.txt。引用的另一聊天因当前会话未提供read_thread工具而未读取，本版没有采用其未核实结论。', 'small')
references = [
    '[1] 姚悦等 基于全偏振显微成像的数字病理技术 中国激光 2024 51(9) 0907008。重点参考2.3、3.2与3.3节。https://doi.org/10.3788/CJL231462',
    '[2] Yao et al. Polarization imaging-based radiomics approach for the staging of liver fibrosis: supplement. Table S1. https://doi.org/10.6084/m9.figshare.19160606',
    '[3] scikit-image 官方文档 slic 的多通道输入 mask compactness及convert2lab说明。https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic',
    '[4] Wu et al. Simplifying Graph Convolutional Networks. ICML 2019. https://proceedings.mlr.press/v97/wu19e.html',
    '[5] Kipf and Welling. Variational Graph Auto-Encoders. 2016. 文中同时给出GAE。https://arxiv.org/abs/1611.07308',
    '[6] Google Colab 官方常见问题。https://research.google.com/colaboratory/faq.html',
    '[7] SciPy 官方文档 loadmat 的variable_names参数和MAT格式支持。https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html',
]
for ref in references:
    p(ref, 'small')
p('版本状态：研究设计已形成；文件头和运行环境已核对；参数图、组织掩膜、聚类结果、数值质量、训练性能及病理对应关系均待实操验证。', 'code')


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#D4DCE6'))
    canvas.line(46, 38, A4[0]-46, 38)
    canvas.setFont('Song', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(46, 25, '乳腺癌类器官偏振图像研究路径  第一版  方法论设计')
    canvas.drawRightString(A4[0]-46, 25, str(doc.page))


doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=46, rightMargin=46,
    topMargin=38, bottomMargin=50, title='乳腺癌类器官偏振图像研究路径', author='研究方案')
doc.build(story, onFirstPage=footer, onLaterPages=footer)

config = {
    'version': '1.0', 'date': '2026-10-30', 'status': 'methodology_only_not_executed',
    'data_root': str(ROOT/'类器官'), 'roi_ids': list(range(1,8)),
    'source_id_in_acquisition_logs': 'BR-029',
    'independent_patient_count': None, 'independent_organoid_count': None,
    'input_shape_from_variable_headers': [2008,2408],
    'pixel_size_um': None, 'mueller_normalization': 'requires_confirmation',
    'parameter_formulas_on_normalized_elements': {
        'PL':'sqrt(m21**2 + m31**2)', 'DL':'sqrt(m12**2 + m13**2)',
        'qL':'sqrt(m42**2 + m43**2)', 'rL':'sqrt(m24**2 + m34**2)'},
    'mask': 'human_reviewed_tissue_mask_excluding_background_and_lumens',
    'slic': {'n_segments_primary':1000,'n_segments_sensitivity':[500,1500],
        'compactness_primary':1.0,'compactness_sensitivity':[0.1,10.0],
        'convert2lab':False,'channel_axis':-1,'enforce_connectivity':True},
    'node_features': {'channels':['PL','DL','qL','rL'],'statistics':['mean','std'],'dimension':8},
    'graphs': ['spatial_adjacency_unweighted','spatial_adjacency_feature_weighted'],
    'cross_roi_edges': False, 'shared_encoder_across_rois': True,
    'clustering': {'primary_k':4,'candidate_k':[3,4,5,6],'n_init':20,
        'centers_shared_across_rois_per_method':True},
    'seeds':[0,1,2,3,4],
    'methods': {
        'M0':'standardized_node_features_then_kmeans',
        'M1':{'name':'sgc_style_propagation_then_kmeans','primary_hops':2,'candidate_hops':[1,2,3]},
        'M2':{'name':'gae_then_kmeans','encoder_dims':[8,32,16],'epochs':150,
            'learning_rate':0.001,'reconstruction_target':'binary_spatial_adjacency',
            'negative_sampling':'within_each_roi_graph_only','sparse_propagation':True}},
    'optional_later':['MMPD_depolarization_and_retardance','DGI','VGAE'],
    'evaluation':['silhouette_in_common_base_space','native_embedding_silhouette',
        'pairwise_seed_ARI','pixel_area_fraction','shannon_composition_entropy',
        'connected_component_distribution','boundary_length_and_class_contacts'],
    'inference_scope':'exploratory_analysis_on_current_7_rois_not_unseen_patient_validation',
    'required_before_practice':['normalization_confirmation','sample_relationships',
        'tissue_mask_scope','pixel_scale_and_reference_images'],
}
(OUT/'实验设计_v1.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'数据目录核对.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['roi_id','source_id_from_log','acquisition_datetime','matrix_path',
        'matrix_file_bytes','array_shape_from_header','numeric_elements','pixel_values_validated'])
    for i in range(1,8):
        d=ROOT/'类器官'/str(i)
        info=(d/'info.txt').read_text(encoding='utf-8-sig')
        writer.writerow([i,'BR-029',re.search(r'DateTime:(.+)',info)[1].strip(),
            str(d/'FinalMM.mat'),(d/'FinalMM.mat').stat().st_size,'2008x2408',
            'FinalM11..FinalM44 + CalibratedM11','not_validated'])
print(json.dumps({'pdf':str(PDF),'config':str(OUT/'实验设计_v1.json'),
    'inventory':str(OUT/'数据目录核对.csv')},ensure_ascii=False))
