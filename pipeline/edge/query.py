from queue import Queue


def query_bfs(correct_edge_dict:dict, start_node_ids:int, query_len:int = 10):
    q = Queue()
    for start_node_id in start_node_ids:
        q.put(start_node_id)
    nodes_set = set()
    edges_list = [] #乱序
    stop_flag = False
    while(not q.empty() and not stop_flag):
        cur_node = q.get()
        value = correct_edge_dict.get(cur_node)
        if(not value):
            continue
        cur_edge = value[0] #边id的累积
        for node in value[1:]:
            q.put(node)
            nodes_set.add(node)
            edges_list.append(cur_edge)
            cur_edge += 1
            if(len(nodes_set) + len(edges_list) >= query_len):
                stop_flag = True
                break
    nodes_list = list(nodes_set) #乱序
    return nodes_list, edges_list


def query_bfs2(correct_edge_dict:dict, correct_edge_dict2:dict, start_node_ids:int, query_len:int = 10):
    q = Queue()
    for start_node_id in start_node_ids:
        q.put(start_node_id)
    nodes_set = set()
    edges_list = [] #乱序
    stop_flag = False
    while(not q.empty() and not stop_flag):
        cur_node = q.get()
        value2 = correct_edge_dict2.get(cur_node)
        if(not value2):
            continue
        for node in value2:
            q.put(node)
            nodes_set.add(node)
            #获取cur_node与node相连边的id
            value = correct_edge_dict.get(node)
            cur_edge = value[0] + value.index(cur_node) - 1
            edges_list.append(cur_edge)
            if(len(nodes_set) + len(edges_list) >= query_len):
                stop_flag = True
                break
    nodes_list = list(nodes_set) #乱序
    return nodes_list, edges_list