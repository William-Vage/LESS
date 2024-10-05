from queue import Queue
import numpy as np
import argparse
import json
from time import *
import torch
import leonard.config as config
import zipfile
import os

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-dataset_flag', action='store', dest='dataset_flag')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-gpu', action='store', dest='gpu',
                    help='gpu')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-data_params', action='store', dest='params_file',
                    help='params file')
parser.add_argument('-table_file', action='store', dest='table_file',
                    help='table_file')
parser.add_argument('-model_lite', action='store', dest='model_lite',
                    help='table_file')
parser.add_argument('-edge_file', action='store', dest='edge_file',
                    help='table_file')
args = parser.parse_args()
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

extract_path = os.path.join(config.project_root, 'data/compress/')
args.model_lite = os.path.join(extract_path, 'leonard_lstm.pt')  # "../../data/model/leonard_lstm.pt"
args.model_name = "LSTM_multi"
args.params_file = os.path.join(extract_path, 'params.json')  # "../../data/encode/params.json"
args.table_file = os.path.join(extract_path,
                               'my_calibration_table.json')  # "../../data/correct/my_calibration_table.json"
args.dataset_flag = 1
args.edge_file = os.path.join(extract_path, 'edges.txt')  # "../../data/encode/edges.txt"
args.gpu = 0
device = torch.device('cpu')
print(torch.cuda.is_available())


# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def translate_edge(path):  # 从文件中读取边
    f = open(path, 'r', encoding="unicode_escape")
    strr = f.read()
    strr = strr.split('\n')
    edges = []
    edges.append(eval(strr[0]))
    edges.append(eval(strr[1]))
    edges.append(eval(strr[2]))
    edges.append(eval(strr[3]))
    return edges


# 在NumPy数组上创建滑动窗口视图，可以帮助实现数据的分块处理
def strided_app(a, L, S):  # a:数组，L:滑动窗口的长度，即每个窗口中包含的元素数量，S:滑动窗口的步幅，即窗口之间相邻的元素之间的距离。
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)
    # shape: 输出数组的形状，这里是 (nrows, L)，表示行数为 nrows，列数为 L。
    # strides: 输出数组中每个轴上的步幅，这里是 (S * n, n)，表示在第一个轴上的步幅为 S * n，在第二个轴上的步幅为 n。


global key_pattern
global id2char_dict
global char2id_dict
global re_values
global mins
global key_template_dict
global key_template
global alphabet_size
global table


# 从输入的查询数据中提取最近的一些时间步骤，以用于预测任务。
def get_for_presict(queries, timesteps):  # queries：包含多个查询的列表，timesteps: 要提取的时间步骤的数量，即从每个查询的末尾向前提取的步骤数目。
    tmp_query = []
    for i in range(len(queries)):
        tmp_query.append(np.array(queries[i][len(queries[i]) - timesteps:]))  # 从查询的末尾向前数 timesteps 步
    return tmp_query


def generate_query_counters(query_sequence, counter, flag):
    # 函数接受三个参数：
    # query_sequence: 一个包含查询序列的列表
    # counter: 计数器
    # flag: 标志，用于确定要在查询中添加的特定信息

    # 初始化一些空列表用于存储结果
    sentences = []  # 用于存储生成的查询token
    counters = []  # 用于存储query_sequece编号的字符串格式
    strr = []  # strr中存储的类型要么是'vertxid:'，要么是‘eventid:’

    # 根据标志设置不同的字符串，分别用于“verteid”和“eventid”
    if flag == -1:
        for i in range(len(query_sequence)):
            strr.append('verteid:')
    if flag == -2:
        for i in range(len(query_sequence)):
            strr.append('eventid:')
    if flag > -1:
        # 如果标志大于-1，则将"verteid"和"eventid"相继出现在查询中
        for i in range(flag + 1):
            strr.append('verteid:')
        for i in range(flag + 1, len(query_sequence)):
            strr.append('eventid:')
    # 遍历查询序列
    for i in range(len(query_sequence)):
        tmp_sentence = []  # 用于存储当前查询的token
        tmp_strr = query_sequence[i]  # 获取当前查询的节点/边的编号
        counters.append(str(query_sequence[i]))  # 将当前查询添加到计数器列表中
        # 遍历当前查询的数据类型，strr中存储的类型要么是'vertxid:'，要么是‘eventid:’
        for j in strr[i]:
            # 将'verteid:'/'eventid:'转换为相应的编码，并将其添加到token中
            tmp_sentence.append(int(char2id_dict[j]))
        # 遍历当前查询的边号/节点号的每个字符
        for j in str(tmp_strr):
            # 将字符转换为相应的编码，并将其添加到token中
            tmp_sentence.append(int(char2id_dict[j]))
        tmp_sentence.append(0)  # 在token末尾添加一个零，表示此'verteid:xxx'的编码结束
        sentences.append(tmp_sentence)  # 将token添加到token列表中

    return sentences, counters  # 返回生成的token列表和counters列表，token和counters涉及到的编号均为o1和o2输入数据的原始节点编号


def predict_lstm(queries, counters_, timesteps, alphabet_size, start, overlen, overflag, cut_num=-1):
    flag = []
    queries_len = len(queries)  # token的数目
    # 初始化标志位列表，用于标记哪些查询已经完成预测
    for i in range(len(queries)):
        flag.append(False)

    strr = []  # 构建字符串列表 strr，用于存储查询的类型信息（'e:' 或 'v:'）

    # 根据 overflag 设置标志位，跳过已经预测的查询
    if overflag != -1:
        for i in range(overflag, len(queries)):
            flag[i] = True

    cnt_v = 0
    # 构建字符串列表 strr，存储查询的类型信息
    for i in range(len(queries)):
        if flag[i]:
            continue
        else:
            if queries[i][0] == 3:  # 如果这一位是3，则是边，否则是节点
                strr.append('e:')
            else:
                strr.append('v:')
                cnt_v += 1
    begin = 0

    # 在有未完成预测的查询的情况下循环预测
    while False in flag:
        lenn = 0
        record = []
        tmp_query = get_for_presict(queries, timesteps)  # 获取用于预测的查询窗口数据
        begin_ = time()
        X = np.array(tmp_query).astype(np.float32)  # 将查询窗口数据转换为 NumPy 数组
        X = torch.tensor(X, device=device, dtype=torch.long)

        # 初始化 TensorFlow Lite 解释器
        interpreter = torch.load(args.model_lite).to(device)
        prob = interpreter(X).detach().cpu().numpy()
        # interpreter.allocate_tensors()
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # interpreter.set_tensor(input_details[0]["index"], X)
        # interpreter.invoke()
        #
        # prob = interpreter.get_tensor(output_details[0]["index"]) # 获取模型预测结果

        # 对每个查询进行处理
        for i in range(len(queries)):
            if flag[i]:
                continue
            # 纠错
            if strr[i] + counters_[i] in table.keys():  # 如果在纠错表中，则用纠错表的数据，否则用预测数据
                if str(begin) in table[strr[i] + counters_[i]][0]:
                    queries[i].append(
                        int(table[strr[i] + counters_[i]][1][table[strr[i] + counters_[i]][0].index(str(begin))]))
                else:
                    queries[i].append(np.argmax(prob[i]))
            else:
                queries[i].append(np.argmax(prob[i]))
            # 如果遇到为数字1的值，则表明此边/节点的详细信息已恢复完毕
            if queries[i][-1] == 1:
                flag[i] = True
        begin = begin + 1
    if cut_num == -1:
        return queries[:cnt_v], queries[cnt_v:]
    else:
        return queries[:cnt_v], queries[cnt_v:cut_num]


def get_slice(data, flag):
    add = 0
    if flag == 1:
        add = 1
    ind = np.where(np.array(data) == flag)[0]
    finaldata = []
    for i in range(len(ind)):
        if i == 0:
            finaldata.append(data[:ind[i] + add])
        else:
            finaldata.append(data[ind[i - 1] + 1:ind[i] + add])
    return finaldata


def get_slice_index(data, flag):
    ind = np.where(np.array(data) == flag)[0]
    return ind


def translate_str(data, key=''):
    process = ''
    if isinstance(data, list):
        for i in data:
            process = process + id2char_dict[str(i)]
    else:
        process = process + id2char_dict[str(data)]
    return process


def translate(data, key='', flag=0):
    data = data[0:len(data) - 1]
    if data[0] == 3:
        flag = 1
    else:
        flag = 0
    data = get_slice(data, 0)
    counter_tmp = data[0][8:]
    key_index = data[1]
    strrr = ''
    key_list = {}
    for i in key_index:
        strrr = strrr + id2char_dict[str(i)]
    key_index = int(strrr)
    if key_index > 14:
        return []
    key_pattern = key_template[key_index].split(',')
    if flag == 0:
        key_list['hash'] = translate_str(counter_tmp)
    begin = 2
    for index in range(len(key_pattern)):
        key = key_pattern[index]
        if key != 'hash':
            if key == 'parentVertexHash':
                key_list[key] = str(o1[int(translate_str(counter_tmp))])
                continue
            elif key == 'childVertexHash':
                key_list[key] = str(o2[int(translate_str(counter_tmp))])
                continue
            elif key == 'timestampNanos':
                key_list[key] = str(int(translate_str(data[begin], key)) + mins[0])
            elif key == 'startTimestampNanos':
                key_list[key] = str(int(translate_str(data[begin], key)) + mins[1])
            elif key == 'sequence':
                key_list[key] = str(int(re_values[key][int(translate_str(data[begin], key))]) + mins[2])
            elif key == 'time':
                key_list[key] = str(translate_str(data[begin], key).replace(",", "."))
            else:
                key_list[key] = re_values[key][int(translate_str(data[begin], key))]
            begin = begin + 1
    return key_list


def process_table(table):
    table_tmp = {}
    for key, values in table.items():
        table_tmp[key] = [[], []]
        for value in values:
            table_tmp[key][0].append(value[0])
            table_tmp[key][1].append(value[1])
    return table_tmp


def iter_bfs_score(now, sett_edges, sett_points, G, mapp):  # 在一个图中搜索节点
    searched = set()
    count = 0
    # now=sorted(now, key=lambda x: x[3])
    # 在待搜索队列中存在元素的情况下循环
    while now:
        large = -1.0
        tmp_edge = 0
        # 从当前待搜索队列中选择权重最大的边
        for j in now:
            if large < float(j[3]):
                tmp_edge = j
                large = float(j[3])
            elif large == float(j[3]):
                # 如果权重相同，按照节点的数据进行比较
                # if data[int(tmp_edge[0])+1][0]<=data[int(j[0])+1][0]:
                tmp_edge = j
            else:
                continue
        now.remove(tmp_edge)  # 从待搜索队列中移除选择的边
        i = tmp_edge
        # 将边的两个节点添加到结果集合
        sett_points.add(i[2])
        sett_points.add(i[1])
        sett_edges.add(i[0])
        # 如果结果集合的大小超过了4096，结束搜索
        if (len(sett_points) + len(sett_edges)) > 4096:
            break
        # 如果目标节点没有被搜索过，将其相邻的边加入待搜索队列
        if i[1] in searched:
            continue
        # 获取目标节点在图中的索引
        idx = mapp[i[1]]
        # 遍历目标节点相邻的边，将未搜索过的边加入待搜索队列
        for target in G[idx]:
            if target[1] not in sett_edges:
                now.append((target[1], target[0], i[1], target[2]))
        # 将目标节点标记为已搜索
        searched.add(i[1])


def query_bfs(start_nodes, sett_edges, sett_points, G, mapp):
    q = Queue()
    q.put(start_nodes[0])
    stop_flag = False
    while (not q.empty() and not stop_flag):
        cur_node = q.get()
        for item in G[mapp[cur_node]]:
            q.put(item[0])
            sett_points.add(mapp[item[0]])  # item[0] debug
            sett_edges.add(item[1])
            if (len(sett_points) + len(sett_edges) >= 20):
                stop_flag = True
                break


# nodes_for_dataset=[[1099876,672640,1683629,514060,468934,1787378,495914,1673209,1854115,499163,1662376,1058801,692629,1339861,87517,423047,351501,1624091,695449,75339,1634507,791085,1616974,1612807,1639534,718370,1600687,702072,1807501,1304266,1360273,478092,284379,44238,569243,1132880,922447,2019306,140756,1009252,1506758,1191750,318206,674460,684007,1530136,684763,254500,1178172,364387,133153,1116597,673242,360253,1640513,852603,272086,1485387,592571,1230191,1829617,1189368,2015994,1578470,997119,1113838,1870472,1924379,1082974,792123,1736645,1554034,456956,1449898,172122,1822603,2085431,664535,324110,518400,291993,2027534,2069026,2026525,1819327,1435847,197891,146763,672947,1728040,1582844,1470589,979480,1034243,1642994,1369251,1506015,1506378,1064841,1527819],[386380,1419319,489746,1208589,1351369,1322994,1441152,1457591,831647,1019033,1000324,1619080,1536855,1650251,1235636,1219451,412740,1502708,1775852,184512,403005,1098833,45920,1659365,1340094,574275,217646,1773816,1463021,1497852,2026509,499354,1612677,739469,1509590,1806135,433571,1851742,989442,933631,1563975,805993,1113467,1475433,1687174,66988,1500081,374387,1549762,479145,119387,38651,157823,643752,1366003,799440,270628,118770,464675,1650723,119396,2016400,2019577,1499278,159201,1435211,1298889,1788413,1123153,1327216,1649102,1991089,185710,1758935,1650063,1128066,1628617,1371920,1875922,321859,1466785,2043339,1329074,476021,1029566,2077727,1298458,619837,1120426,1424348,518053,389895,1058097,215756,1217221,859212,1497905,687354,810243,2085209],[2054723,1172632,1485986,1549171,1058520,2176463,1212923,1455768,478597,836553,2285074,2223302,2051109,1898061,1206031,146034,757153,1409430,1184120,1277486,1055475,1381211,354411,1542675,547630,503181,977851,863289,602619,139201,626089,2029164,523411,980114,363525,2279184,1859295,951357,1763616,893035,1187182,1728043,2477699,2070710,254372,2121434,407311,1491560,315540,922657,248777,1799208,1127736,103205,437186,1657621,744882,1536181,1475247,2144395,1175033,1677416,1984768,1719981,112217,1954096,694702,1704130,1564996,725601,580634,2097648,1498115,1342585,554636,1358709,1397009,334768,2025087,2132241,2254621,122438,2369066,1920374,1003936,1563641,2303017,2112770,108813,1342015,2451765,277774,463982,2320394,662901,1372130,1873686,1327713,11605,1493703],[2028413,1537437,2307752,750488,12411,1457486,67400,919108,1706665,1721297,1355245,1710035,981014,2061880,786164,1296562,1888900,1408198,1689322,1271718,2256451,2152657,386670,440839,1457064,311472,371297,1063072,1089877,2228589,749696,1570158,647429,306897,2328721,1572136,114364,945255,1354622,1688894,863135,799935,770571,1241480,2093111,1606483,187486,2257951,689875,615366,1418675,1529873,1405346,169461,659628,795627,2212837,2381089,1481331,1229565,427972,1562650,2304459,16150,736761,592786,1673849,2324251,2368891,430015,343653,1309356,1564634,621348,1348223,1500490,1550379,2043612,1298832,35512,1464280,496359,1176916,288298,641753,2045693,2012950,357783,722411,1017849,779582,1533708,449980,2074509,116983,88608,795783,643371,1107616,745769],[1661382,1650724,1269905,526014,114253,1693696,1629943,1169519,1321876,1726021,1693831,1768043,1157268,1332937,847821,300570,624866,1628081,1228962,825931,113235,1455527,1584311,911150,770295,734539,296826,1310022,1697164,1218926,1410931,1762553,1117447,1412238,1144590,270948,1190897,1311171,1366439,470059,747390,705080,1736835,1288862,959894,1193369,1156745,1747874,324553,1365218,25873,950816,419235,1103630,1713625,1715518,192929,111477,1377550,1069384,765409,942147,1208167,604624,1033381,1635313,656383,1453264,975508,861160,1357715,1783658,562905,879057,98371,1402400,672634,231872,155267,1223763,1230825,752588,898699,1278801,1366493,1328788,229544,291064,1351109,1681764,127579,1293580,1551511,1294815,458605,1133961,784528,747790,393502,172665]]


def query(dataset):
    # 从数据集中选择起始节点，args.dataset_flag 是一个命令行参数，表示选择的数据集标志
    # start_nodes=nodes_for_dataset[int(args.dataset_flag)-1]#[672640,1683629,514060,468934,1787378,495914,1673209,1854115,499163,692629]
    edges = translate_edge(args.edge_file)
    start_nodes = []
    if dataset == 'toy':
        start_nodes = [edges[2][0]]  # [44238]
    elif dataset == 'darpa_tc':
        start_nodes = [edges[2][0]]  # [14283]

    global o0  # 节点某id
    o0 = edges[0]
    global o1
    o1 = edges[2]  # 边的parent id
    global o2
    o2 = edges[3]  # 边的children id

    mapp = {}
    G = []  # 溯源图
    for i in range(len(o1)):
        a, b = o1[i], o2[i]
        if a not in mapp:
            mapp[a] = len(G)  # 将a这条边映射成G的大小这一数字
            G.append([])  # 这个新的list也许是G为上一行的边预留的存储边具体信息的空间
        if b not in mapp:
            mapp[b] = len(G)
            G.append([])
        G[mapp[a]].append((b, i, 1))  ########## parent连接的：【child节点，计数下标，score值】

    sett_edges = set()  # 边集
    sett_points = set()  # 节点集
    final_results_v = []
    final_results_e = []
    # 遍历起始节点进行搜索
    for start in start_nodes:
        # now=[]
        # indexx=[] #存储以当前起始节点为 parent id 的边在边集的索引
        # # 获取所有以当前起始节点为 parent id 的边在边集的索引
        # for i in range(len(o1)):
        #     if int(o1[i])==start:
        #         indexx.append(i)
        # for i in indexx:
        #     now.append((i,o2[i],o1[i],1)) ###########【parent节点的索引，child节点号，parent边，权重值】
        # iter_bfs_score(now,sett_edges,sett_points,G,mapp)  #寻找关联的边集
        query_bfs(start_nodes, sett_edges, sett_points, G, mapp)
        ver_len = len(sett_points)  # 深搜搜到的节点的数目

        query = []
        for ver in sett_points:
            query.append(o0[ver])  # 待查询还原详细信息的节点
        # for item in o0:
        #     if(item == 9116):
        #         print()
        # query[0] = 9116
        # query[1] = 1883
        for ed in sett_edges:
            query.append(ed)  # 待查询还原详细信息的边
        end_time1 = time()
        np.random.seed(0)
        timesteps = 10  # 某个超参数
        total_counter = len(query)  # 待查询的边和节点数总和
        batch_size = 2048
        counter = []
        addd = 0
        if total_counter > int(total_counter / batch_size) * batch_size:  # 是否有尾部的小batch
            addd = 1
        for j in range(int(total_counter / batch_size) + addd):  # 对于每一个batch
            if addd == 1 and j == int(total_counter / batch_size):  # 最后小batch的情况
                flag = -1
                if batch_size * j < ver_len:
                    flag = ver_len - 1 - batch_size * j
                    begin1 = flag
                else:
                    flag = -2
                queries, counters_ = generate_query_counters(query[batch_size * j:], counter, flag)  # 生成要query的序列
                tmp_query_len = len(queries)  # 要query序列的大小
                tmp_query, tmp_counter = generate_query_counters(query[:2], counter, -1)
                for i in range(batch_size - tmp_query_len):
                    queries.append(tmp_query[0])
                    counters_.append(tmp_counter[0])
                final_sentences_v, final_sentences_e = predict_lstm(queries, counters_, timesteps, alphabet_size,
                                                                    batch_size * j, ver_len, tmp_query_len,
                                                                    cut_num=tmp_query_len)
                # final_sentences=final_sentences[:tmp_query_len]
            else:
                flag = -1  # 表示verteid:
                # 改写flag：意思是若要查询的节点数不足一个一个batchsize，则将flag改写成节点数的大小，从而在generate_query_counters中走相应的if分支
                # 从而正确的将batch内的节点打上vertex:，边打上edges:
                if batch_size * j <= (ver_len - 1) <= batch_size * (j + 1):
                    flag = ver_len - 1 - (batch_size * j)
                    begin1 = flag
                if (ver_len - 1) < batch_size * j:
                    flag = -2  # 表示全是边，全部打上eventid:
                queries, counters_ = generate_query_counters(query[batch_size * j:batch_size * (j + 1)], counter, flag)

                final_sentences_v, final_sentences_e = predict_lstm(queries, counters_, timesteps, alphabet_size,
                                                                    batch_size * j, ver_len,
                                                                    -1)  # 完成模型预测详细信息恢复[2, 3, 4, 5, 3, 6, 7, 8, 10, 11, 12, 15, 18, 18, 12, 0, 11, 0, 11, 0, 10, 16, 12, 18, 0, 15, 18, 0, 18, 13, 15, 15, 0, 9, 0, 9, 0, 10, 11, 12, 15, 18, 18, 12, 0, 1]
            # 完成详细信息数字编码转换为最终字符串：{'hash': '1236993', 'cdmtype': 'SrcSinkObject', 'epoch': '258', 'fileDescriptor': '51', 'pid': '25254', 'source': 'SOURCE_LINUX_SYSCALL_TRACE', 'type': 'Object', 'uuid': '27b62dc819c0ebcf4131baf4fa52b004'}
            for i in range(len(final_sentences_v)):
                ret = translate(final_sentences_v[i], flag=0)
                if len(ret) != 0:
                    final_results_v.append(ret)
            for i in range(len(final_sentences_e)):
                ret = translate(final_sentences_e[i], flag=0)
                if len(ret) != 0:
                    final_results_e.append(ret)
        # final_result=[] #单纯的json转字符串，不知道想做啥。。。
        # for i in range(len(final_results)):
        #     final_result.append(str(final_results[i]))
    # 指定保存的文件名
    output_file_v = os.path.join(config.project_root, f"data/query_result/{dataset}_query_result_vertex.json")
    output_file_e = os.path.join(config.project_root, f"data/query_result/{dataset}_query_result_edge.json")

    # 将final_results列表保存为JSON文件
    with open(output_file_v, 'w') as f:
        json.dump(final_results_v, f)
    with open(output_file_e, 'w') as f:
        json.dump(final_results_e, f)
        # edges_output=[]
        # vertex_output=[]
        # pidd={}
        # count_for=[]
        # print(start)
        # print('searched number')
        # print(len(sett_edges),len(sett_points))
        # print(end_time1-begin_time)


def leonard_query_run(dataset='toy'):
    begin_time = time()
    out_zip_file = ''
    if dataset == 'toy':
        out_zip_file = os.path.join(config.project_root, 'data/compress/leonard_toy_output.zip')
    elif dataset == 'darpa_tc':
        out_zip_file = os.path.join(config.project_root, 'data/compress/leonard_darpa_tc_output.zip')
    # 指定解压缩目标路径
    extract_path = os.path.join(config.project_root, 'data/compress/')
    # 打开 zip 文件
    with zipfile.ZipFile(out_zip_file, "r") as zip_ref:
        # 解压缩所有文件到指定路径
        zip_ref.extractall(extract_path)
    with open(args.params_file, 'r') as f:
        params = json.load(f)
    global id2char_dict
    id2char_dict = params['id2char_dict']
    global char2id_dict
    char2id_dict = params['char2id_dict']
    global re_values
    re_values = params['re_values_dict']
    global mins
    mins = params['mins']
    global key_template_dict
    key_template_dict = params['key_template_dict']
    global key_template
    key_template = {}
    for i in key_template_dict.keys():
        tmp = i.replace('id', 'hash', 1)
        tmp = tmp.replace('predicate_id', 'parentVertexHash', 1)
        tmp = tmp.replace('subject_id', 'childVertexHash', 1)
        key_template[key_template_dict[i]] = tmp

    global alphabet_size
    alphabet_size = len(params['id2char_dict']) + 2
    global table
    with open(args.table_file, 'r') as f:
        params = json.load(f)
    table = process_table(params)

    query(dataset)
    end_time = time()
    print('Query time: ', end_time - begin_time)


if __name__ == "__main__":
    leonard_query_run(dataset='toy')
