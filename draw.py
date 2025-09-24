from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import get_parse_args
from mixgate.single_parser import NpzParser
import mixgate.top_model_rexmg as top_model
import torch.nn as nn
import pandas as pd
import seaborn as sns

DATA_DIR = './data/'
CKPT_PATH = './dense.pth'
CKT_NAME = 'ADD'

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-8)  # 加 1e-8 防止除以 0

def normalize_rows_to_minus1_1(matrix):
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    
    # 防止除以0
    denom = row_max - row_min
    denom[denom == 0] = 1

    normalized = 2 * (matrix - row_min) / denom - 1
    return normalized

def plot_aig_attention_heatmap(token_names, attn_scores, target_token="xmg_45", 
                              modal_prefix="aig_", top_n=10, figsize=(40,3)):
    """
    绘制目标token与指定模态（默认AIG）token的注意力热力图
    :param token_names: 全部token名称列表
    :param attn_scores: 注意力分数矩阵 [num_heads, seq_len, seq_len]
    :param target_token: 目标token名称（如"xmg_45"）
    :param modal_prefix: 要筛选的模态前缀（如"aig_"）
    :param top_n: 显示前n个token
    :param figsize: 图像尺寸
    """
    # 异常检测
    if target_token not in token_names:
        print(f"[ERROR] Target token {target_token} not found")
        return
    
    # 使用第一个注意力头
    attn_head = attn_scores[0].detach().cpu().numpy()  # [seq_len, seq_len]
    
    # 获取目标token索引
    target_idx = token_names.index(target_token)
    
    # 筛选AIG tokens
    aig_indices = [i for i, name in enumerate(token_names) 
                  if name.startswith(modal_prefix)]
    if not aig_indices:
        print(f"[ERROR] No tokens found with prefix {modal_prefix}")
        return
    
    # 提取目标到AIG的注意力分数
    target_to_aig = attn_head[target_idx, aig_indices]
    
    # 获取前top_n索引（处理不足top_n的情况）
    valid_top_n = min(top_n, len(aig_indices))
    sorted_indices = np.argsort(target_to_aig)[::-1][:valid_top_n]
    
    # 生成标签和分数
    top_labels = [token_names[aig_indices[i]] for i in sorted_indices]
    top_scores = target_to_aig[sorted_indices]
    
    # 创建热力图数据
    df = pd.DataFrame([top_scores], 
                     columns=top_labels,
                     index=[target_token])
    
    # 绘图设置
    plt.figure(figsize=figsize)
    sns.heatmap(df, 
               annot=True, 
               fmt=".2f",
               cmap="YlGnBu",
               cbar_kws={'label': 'Attention Score'},
               annot_kws={"size": 8})
    
    # 美化显示
    plt.title(f"Attention from {target_token} to Top-{valid_top_n} {modal_prefix.upper()}Tokens", 
             fontsize=14, pad=20)
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{target_token}_to_{modal_prefix}heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {target_token}_to_{modal_prefix}heatmap.png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_multi_modal_heatmap(token_names, attn_scores, target_token="xmg_45", 
                            top_n=10, figsize=(25, 8), save_path="./multi_modal_heatmap.png"):
    """
    修复版多模态热力图绘制函数
    每个单元格显示token名称，颜色反映注意力分数
    """
    # 检查目标token是否存在
    if target_token not in token_names:
        print(f"[ERROR] Target token {target_token} not found")
        return
    
    # 使用第一个注意力头
    attn_head = attn_scores[0].detach().cpu().numpy()
    target_idx = token_names.index(target_token)
    
    # 定义模态配置
    modal_config = {
        'AIG': {'prefix': 'aig_', 'color': 'Blues'},
        'MIG': {'prefix': 'mig_', 'color': 'Greens'},
        'XAG': {'prefix': 'xag_', 'color': 'Reds'}
    }
    
    # 收集各模态数据
    data_dict = {}
    max_length = 0
    for modal, config in modal_config.items():
        # 筛选当前模态token
        indices = [i for i, name in enumerate(token_names) 
                  if name.startswith(config['prefix'])]
        if not indices:
            print(f"[WARNING] No {modal} tokens found")
            continue
        
        # 提取注意力分数并排序
        scores = attn_head[target_idx, indices]
        sorted_idx = np.argsort(scores)[::-1][:top_n]
        
        # 保存token名称和对应分数
        tokens = [token_names[indices[i]] for i in sorted_idx]
        scores = scores[sorted_idx]
        
        # 对齐到最大长度
        padded_tokens = tokens + [''] * (top_n - len(tokens))
        padded_scores = np.concatenate([scores, np.full(top_n - len(scores), np.nan)])
        
        data_dict[modal] = {
            'tokens': padded_tokens,
            'scores': padded_scores
        }
        max_length = max(max_length, len(padded_tokens))
    
    # 创建数据矩阵
    score_matrix = []
    annot_matrix = []
    for modal in ['AIG', 'MIG', 'XAG']:
        if modal in data_dict:
            score_matrix.append(data_dict[modal]['scores'])
            annot_matrix.append(data_dict[modal]['tokens'])
        else:
            score_matrix.append(np.full(top_n, np.nan))
            annot_matrix.append([''] * top_n)
    
    # 转置矩阵以适应热力图格式
    score_matrix = np.array(score_matrix).T
    annot_matrix = np.array(annot_matrix).T
    
    # 创建DataFrame
    df_scores = pd.DataFrame(score_matrix, columns=modal_config.keys())
    df_annot = pd.DataFrame(annot_matrix, columns=modal_config.keys())
    
    # 绘图设置
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        df_scores,
        annot=df_annot,  # 使用token名称作为注释
        fmt="",          # 禁用数值格式化
        cmap="YlGnBu",
        # cbar_kws={'label': 'Attention Score'},
        linewidths=0.5,
        annot_kws={"size": 10, "color": "black"}
    )
    
    # 坐标轴设置
    ax.set_xticklabels([f"{k} Tokens" for k in modal_config.keys()], 
                      rotation=0, ha='center', fontsize=12)
    ax.set_yticklabels([])  # 隐藏默认y轴标签
    # ax.set_ylabel("Top Tokens Ranking", fontsize=12)
    
    # 添加右侧排名标注
    for y in range(top_n):
        ax.text(3.2, y+0.5, f"Top {y+1}", 
               ha='left', va='center', fontsize=10)
    
    # 标题设置
    plt.title(f"Attention from {target_token} to Top-{top_n} Tokens", 
             fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Heatmap saved to {save_path}")


def plot_full_cross_modal_heatmap(token_names, attn_scores, figsize=(40, 30), 
                                 save_path="./full_cross_modal_heatmap.png"):
    """
    绘制全量跨模态注意力热力图
    :param token_names: 全部token名称列表
    :param attn_scores: 注意力分数矩阵 [num_heads, seq_len, seq_len]
    :param figsize: 图像尺寸 (宽度, 高度)
    :param save_path: 图片保存路径
    """
    # 使用第一个注意力头
    attn_head = attn_scores[0].detach().cpu().numpy()
    # attn_head = normalize_rows_to_minus1_1(attn_head)
    
    # Stone: 对 mean-std : mean+std 进行截断
    for i in range(attn_head.shape[0]):
        mean = np.mean(attn_head[i])
        std = np.std(attn_head[i])
        attn_head[i] = np.clip(attn_head[i], mean - std, mean + std)
    
    print()
    
    # 筛选行索引（所有xmg节点）
    row_indices = [i for i, name in enumerate(token_names) if name.startswith("xmg_")]
    if not row_indices:
        print("[ERROR] No XMG nodes found")
        return
    
    # 筛选列索引（所有aig/mig/xag节点）
    col_indices = []
    col_labels = []
    for prefix in ["xag_"]:
        indices = [i for i, name in enumerate(token_names) if name.startswith(prefix)]
        col_indices.extend(indices)
        col_labels.extend([token_names[i] for i in indices])
    
    if not col_indices:
        print("[ERROR] No target modality nodes found")
        return
    
    # 提取注意力子矩阵
    sub_matrix = attn_head[np.ix_(row_indices, col_indices)]
    
    # 创建标签
    row_labels = [token_names[i] for i in row_indices]
    
    # 创建DataFrame
    df = pd.DataFrame(sub_matrix, 
                     index=row_labels,
                     columns=col_labels)
    
    # 绘图设置
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        df,
        cmap="seismic",
        cbar_kws={'label': 'Attention Score'},
        annot=False,  # 关闭数值标注（数据量大时影响性能）
        linewidths=0.1,
        linecolor='grey', 
        square=True, 
    )
    
    # 坐标轴优化
    # ax.set_xticks(np.arange(len(col_labels))) # 显示所有列标签
    # ax.set_xticklabels(col_labels, 
    #                   rotation=90, 
    #                   fontsize=6, 
    #                   ha='center')
    ax.set_xticks([])  # 不显示 x 轴刻度
    ax.set_xticklabels([])  # 不显示 x 轴标签

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, 
                      rotation=0, 
                      fontsize=6)
    
    # 添加分隔线
    aig_len = len([n for n in col_labels if n.startswith("aig_")])
    mig_len = len([n for n in col_labels if n.startswith("mig_")])
    
    # 绘制模态分隔竖线
    ax.axvline(aig_len, color='white', linewidth=1)
    ax.axvline(aig_len + mig_len, color='white', linewidth=1)
    
    # 添加模态标签
    ax.text(aig_len/2, -0.05, 'AIG', 
           ha='center', va='top', fontsize=10, color='white')
    ax.text(aig_len + mig_len/2, -0.05, 'MIG', 
           ha='center', va='top', fontsize=10, color='white')
    ax.text(aig_len + mig_len + len(col_labels[aig_len+mig_len:])/2, -0.05, 'XAG', 
           ha='center', va='top', fontsize=10, color='white')
    
    # 标题设置
    plt.title("Cross-Modal Attention: XMG (rows) vs AIG/MIG/XAG (columns)", 
             fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Full cross-modal heatmap saved to {save_path}")


def plot_heatmap(token_name, attn_score, target_token, top_n=10, save_path='heatmap.png'):
    """
    修复后的热力图绘制函数
    :param token_name: 所有token名称列表
    :param attn_score: 注意力分数矩阵 [num_heads, seq_len, seq_len]
    :param target_token: 目标token名称（如'xmg_45'）
    :param top_n: 显示前n个相关token
    :param save_path: 图片保存路径
    """
    # 定位目标token索引
    try:
        target_idx = token_name.index(target_token)
    except ValueError:
        print(f"[ERROR] Token {target_token} not found")
        return

    # 使用第一个注意力头 [seq_len, seq_len]
    attn_head = attn_score[0].detach().cpu().numpy()
    
    # 提取目标token对其他token的注意力分数
    target_scores = attn_head[target_idx, :]
    
    # 获取前top_n索引（确保一维）
    sorted_indices = np.argsort(target_scores)[::-1][:top_n]
    
    # 生成标签和分数
    top_tokens = [token_name[i] for i in sorted_indices]
    top_scores = target_scores[sorted_indices]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 2))
    cax = ax.matshow([top_scores], cmap='viridis', aspect='auto')
    
    # 设置坐标标签
    ax.set_xticks(range(len(top_tokens)))
    ax.set_xticklabels(top_tokens, rotation=90)
    # ax.set_yticks([0])
    ax.set_yticklabels([target_token])
    
    # 添加颜色条
    plt.colorbar(cax)
    plt.title(f"Attention from {target_token} to Top-{top_n} Tokens")
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.close(fig)



if __name__ == '__main__':
    args = get_parse_args()
    args.hier_tf = False
    
    # 加载模型
    print('[INFO] Creating Model...')
    model = top_model.TopModel(
        args,
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )
    model.load(CKPT_PATH)
    model.eval()
    print(f'[INFO] Loaded weights from {CKPT_PATH}')
    
    # 处理电路数据
    circuit_name = 'ADD'
    circuit_path = f'./dataset/npz_dataset/{circuit_name}.npz'
    print(f'[INFO] Parsing {circuit_name} dataset')
    dataset = NpzParser(DATA_DIR, circuit_path)
    _, val_dataset = dataset.get_dataset()
    g = val_dataset[0]  # 取第一个图
    
    # 模型推理
    with torch.no_grad():
        *_, attention_info = model(g)
    
    # 提取注意力信息
    token_names = attention_info['token_names'][0]
    attn_scores = attention_info['attentions'][0]  # [num_heads, seq_len, seq_len]
    
    # Stone: Random Sample 500 tokens
    # sample_idx = np.array(list(range(len(token_names))))
    # np.random.shuffle(sample_idx)
    # sample_idx = sample_idx[:500]
    # sample_idx = np.sort(sample_idx)
    sample_idx = [0, 4, 7, 8, 10, 14, 16, 19, 22, 23, 26, 30, 31, 32, 34, 35, 39, 41, 44, 49, 50, 60, 67, 78, 80, 83, 88, 90, 92, 97, 98, 99, 100, 103, 105, 107, 112, 116, 119, 121, 129, 130, 132, 140, 143, 146, 152, 153, 157, 158, 159, 161, 164, 167, 168, 169, 173, 178, 179, 183, 184, 185, 188, 190, 192, 193, 195, 198, 200, 202, 203, 205, 206, 207, 212, 215, 216, 222, 225, 234, 235, 237, 239, 241, 246, 247, 254, 257, 264, 265, 268, 269, 276, 277, 278, 280, 281, 285, 290, 295, 299, 301, 303, 308, 312, 315, 316, 317, 318, 322, 334, 337, 339, 340, 341, 342, 346, 348, 349, 352, 353, 355, 356, 359, 362, 363, 364, 366, 372, 373, 376, 378, 386, 387, 388, 389, 395, 396, 397, 400, 401, 402, 406, 407, 408, 409, 410, 414, 416, 417, 424, 432, 433, 435, 437, 444, 448, 450, 453, 461, 462, 465, 466, 469, 474, 480, 489, 505, 507, 508, 510, 511, 517, 519, 520, 521, 523, 526, 527, 531, 533, 534, 535, 536, 537, 539, 540, 545, 548, 549, 550, 554, 557, 559, 564, 568, 569, 570, 572, 575, 578, 584, 588, 590, 593, 595, 598, 601, 613, 619, 626, 629, 631, 633, 634, 637, 638, 639, 644, 648, 649, 651, 666, 668, 672, 674, 678, 683, 684, 686, 687, 690, 693, 695, 707, 708, 709, 710, 712, 713, 718, 719, 720, 723, 724, 725, 727, 728, 729, 732, 737, 738, 739, 743, 745, 747, 750, 751, 753, 757, 758, 759, 768, 772, 779, 784, 786, 791, 795, 796, 797, 798, 807, 825, 828, 829, 831, 841, 846, 847, 849, 853, 857, 858, 860, 861, 865, 871, 872, 881, 886, 887, 888, 892, 893, 897, 900, 902, 911, 912, 920, 921, 922, 923, 927, 928, 932, 940, 941, 947, 948, 953, 954, 960, 964, 965, 970, 975, 977, 979, 984, 986, 987, 993, 994, 997, 998, 1000, 1002, 1003, 1005, 1008, 1010, 1012, 1017, 1020, 1021, 1025, 1029, 1030, 1032, 1040, 1046, 1050, 1051, 1053, 1054, 1055, 1057, 1063, 1064, 1066, 1068, 1069, 1070, 1073, 1074, 1075, 1076, 1078, 1079, 1080, 1083, 1088, 1094, 1097, 1102, 1105, 1109, 1114, 1116, 1118, 1119, 1123, 1127, 1130, 1139, 1142, 1143, 1145, 1148, 1151, 1155, 1162, 1172, 1175, 1176, 1177, 1179, 1180, 1182, 1185, 1189, 1190, 1196, 1201, 1205, 1206, 1208, 1210, 1211, 1212, 1217, 1219, 1223, 1233, 1237, 1240, 1242, 1252, 1255, 1263, 1264, 1266, 1269, 1272, 1282, 1284, 1285, 1286, 1292, 1294, 1299, 1302, 1303, 1304, 1315, 1317, 1318, 1322, 1324, 1325, 1330, 1334, 1338, 1340, 1341, 1344, 1350, 1352, 1354, 1355, 1358, 1364, 1366, 1369, 1370, 1372, 1376, 1377, 1379, 1380, 1382, 1384, 1385, 1387, 1392, 1396, 1397, 1398, 1401, 1402, 1405, 1407, 1415, 1416, 1421, 1424, 1426, 1428, 1430, 1433, 1435, 1445, 1446, 1452, 1453, 1454, 1458, 1461, 1469, 1476, 1477, 1478, 1479, 1480, 1483, 1484, 1485, 1488, 1490, 1491, 1493, 1499, 1501, 1503, 1505, 1506, 1515, 1517]

    sample_token_names = [token_names[i] for i in sample_idx]
    sample_attn_scores = attn_scores[:, sample_idx, :][:, :, sample_idx]
    sample_attn_scores = torch.sum(sample_attn_scores, dim=0).unsqueeze(0)
        
    print()
    # 绘制热力图
    # plot_heatmap(token_names, attn_scores, 'xmg_45', top_n=10)
    # 绘制aig节点到xmg_45节点的前50个相关token的注意力热力图
    # 在主函数中调用（接在获取attention_info之后）
    # plot_aig_attention_heatmap(
    #     token_names=token_names,
    #     attn_scores=attn_scores,
    #     target_token="xmg_45",
    #     modal_prefix="aig_",
    #     top_n=10
    # )

    # 在主函数中调用（接在获取attention_info之后）
# 调用示例
    # plot_multi_modal_heatmap(
    #     token_names=token_names,
    #     attn_scores=attn_scores,
    #     target_token="xmg_45",
    #     top_n=10,
    #     figsize=(18, 12),
    #     save_path="./fixed_heatmap.png"
    # )

    # 调用示例
    plot_full_cross_modal_heatmap(
        token_names=sample_token_names,
        attn_scores=sample_attn_scores, 
        
        # token_names=token_names,
        # attn_scores=attn_scores, 
        figsize=(150, 150),  # 根据节点数量调整尺寸
        save_path="./xag_attention_matrix.pdf"
    )
    
    # select_file = open('./select.txt', 'w')
    # select_file.write('[')
    # for i in range(len(sample_idx)):
    #     if i != len(sample_idx) - 1:
    #         select_file.write('{}, '.format(sample_idx[i]))
    #     else:
    #         select_file.write('{}]\n'.format(sample_idx[i]))
    # select_file.close()
