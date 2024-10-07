from pipeline.preprocess.preprocess_leonard import preprocess_toy
from pipeline.preprocess.preprocess_darpa_tc import preprocess_darpa_tc
from pipeline.preprocess.preprocess_darpa_optc import preprocess_darpa_optc
from pipeline.edge.encode import edge_encode
from pipeline.edge.decode import edge_decode
from pipeline.edge.train_deep import train_deep_model
from pipeline.edge.train_ml import train_ml_model
from pipeline.edge.correct import correct_edge, re_construct_edge
from pipeline.edge.query import query_bfs
from pipeline.property.encode import property_encode
from pipeline.property.decode import property_decode
from pipeline.compress.compress import compress
import time
import os
import config
import json
import shutil
import zipfile

start_node_ids = [0]


def property_optc_experiments():
    preprocess_darpa_optc()
    # 2. Similarity Matrix Computing (S6)
    window_sizes = [2, 4, 6, 8, 10]
    for i in window_sizes:
        print('exp2 wind_size:', i)
        window_size = i
        t_start = time.time()
        property_encode(False, 'word_bag', window_size, suffix=f'_{window_size}')
        t_end = time.time()
        print(f'window_size: {window_size}, Encode property cost: {t_end - t_start}')


def create_dir():
    d = os.path.join(config.project_root, 'data')
    if not os.path.exists(d):
        os.makedirs(d)
    dir_list = ['raw', 'preprocess', 'model', 'encode', 'decode', 'correct', 'compress_result', 'query_result']
    for dir_item in dir_list:
        dir_item = os.path.join(d, dir_item)
        if not os.path.exists(dir_item):
            os.makedirs(dir_item)


def clear_dir():
    dir_root = os.path.join(config.project_root, 'data')
    dir_list = os.listdir(dir_root)
    skip_dir = ['raw', 'compress_result', 'query_result']
    for i in dir_list:
        folder_path = os.path.join(dir_root, i)
        if i in skip_dir:
            continue
        # Ensure the path to be deleted is a folder
        if not os.path.isdir(folder_path):
            continue
        # Delete all contents under the folder
        try:
            for filename in os.listdir(folder_path):
                if filename == '.gitkeep':
                    continue
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error when deleting contents under {folder_path}: {e}")


def topology_train_model(method_name):
    if method_name in config.ml_model_list:
        train_ml_model(method_name)
    else:
        train_deep_model(method_name)


def encode(dataset='darpa_optc', edge_file='', vertex_file='', out_zip_file_path='data/compress_result',
           out_zip_file_name='output.zip', topology_model_name='xgboost'):
    program_start = time.time()
    # 1. Preprocess data
    t_start = time.time()
    if dataset == 'toy':
        preprocess_toy()
    elif dataset == 'darpa_tc':
        preprocess_darpa_tc(edge_file, vertex_file)
    elif dataset == 'darpa_optc':
        preprocess_darpa_optc(edge_file)
    t_end = time.time()
    t_cost_preprocess = t_end - t_start
    print(f"\033[33m Preprocessing Time: {t_cost_preprocess:.1f}s \033[0m")
    # 2. Encode edge
    t_start = time.time()
    edge_encode(dataset) #边id顺序
    t_end = time.time()
    print(f'Encode edge cost: {t_end - t_start}')
    # 3. Train edge prediction model
    t_start = time.time()
    topology_train_model(topology_model_name)
    t_end = time.time()
    print(f'Train edge model cost: {t_end - t_start}')
    # 4. Get edge correction table
    t_start = time.time()
    correct_edge(topology_model_name)
    t_end = time.time()
    print(f'Calibration table cost: {t_end - t_start}')
    # 5. Train property prediction model
    t_start = time.time()
    if dataset == 'toy' or dataset == 'darpa_tc':
        property_encode(encode_vertex=True, method='word_bag', window_size=4)
    elif dataset == 'darpa_optc':
        property_encode(encode_vertex=False, method='word_bag', window_size=6)
    t_end = time.time()
    print(f'Encode property cost: {t_end - t_start}')
    # 6. Pack into zip
    t_start = time.time()
    if dataset == 'toy' or dataset == 'darpa_tc':
        compress(topology_model_name, encode_vertex=True, out_zip_file_path=out_zip_file_path,
                 out_file_name=out_zip_file_name)
    elif dataset == 'darpa_optc':
        compress(topology_model_name, encode_vertex=False, out_zip_file_path=out_zip_file_path,
                 out_file_name=out_zip_file_name)
    t_end = time.time()
    print(f'Make zip file cost: {t_end - t_start}')
    # 7. Output total time
    program_end = time.time()
    total_cost = program_end - program_start
    compression_time = total_cost - t_cost_preprocess
    # print(f'Total cost: {program_end - program_start}')
    print(f"\033[33m Compression Time: {compression_time:.1f}s \033[0m")
    return total_cost



def decode(dataset='darpa_optc', topology_model_name='xgboost',
           compressed_filename='output.zip'):
    compressed_filepathname = os.path.join(config.project_root, 'data/compress_result', compressed_filename)
    unzip_dir = os.path.dirname(compressed_filepathname)
    program_start = time.time()
    # Step 0: Unzip the zip file to the specified path
    try:
        with zipfile.ZipFile(compressed_filepathname, 'r') as zip_ref:
            print(f"Extracting {compressed_filepathname} ...")
            zip_ref.extractall(unzip_dir)
    except FileNotFoundError:
        print(f"Error: The zip file '{compressed_filepathname}' does not exist.")
        exit(1)
    except zipfile.BadZipFile:
        print(f"Error: The zip file '{compressed_filepathname}' is corrupted.")
        exit(1)
    try:
        # Move files to target directory
        found_file_path = os.path.join(unzip_dir, config.model_name_appendex[topology_model_name])
        destination = os.path.join(config.project_root, 'data/model', config.model_name_appendex[topology_model_name])
        shutil.move(found_file_path, destination)

        found_file_path = os.path.join(unzip_dir, 'calibration_table.txt')
        destination = os.path.join(config.project_root, 'data/correct/calibration_table.txt')
        shutil.move(found_file_path, destination)

        found_file_path = os.path.join(unzip_dir, 'edge_property.txt')
        destination = os.path.join(config.project_root, 'data/encode/edge_property.txt')
        shutil.move(found_file_path, destination)

        if dataset == 'darpa_tc' or dataset == 'toy':
            found_file_path = os.path.join(unzip_dir, 'vertex_property.txt')
            destination = os.path.join(config.project_root, 'data/encode/vertex_property.txt')
            shutil.move(found_file_path, destination)

        print(f"unzip {compressed_filepathname} successfully.")
    except Exception as e:
        print(f"Error while moving file: {e}")
    # 1. Read correction table
    t_start = time.time()
    re_construct_edge(topology_model_name)
    t_end = time.time()
    print(f'Reload calibration table cost: {t_end - t_start}')
    # 2. Decode
    t_start = time.time()
    correct_edge_dict, correct_edge_dict2 = edge_decode()
    t_end = time.time()
    print(f'Regenerate origin parent-children edge dict cost: {t_end - t_start}')
    # 3. Query BFS(search children)
    t_start = time.time()
    global start_node_ids
    if dataset == 'darpa_tc':
        start_node_ids = [39985]
    nodes_ids, edges_dict_list = query_bfs(correct_edge_dict, start_node_ids, 20)
    print('node id + edge id = ', len(nodes_ids) + len(edges_dict_list))
    print(f'query cost: {t_end - t_start}')
    # # Query BFS2(search parents)
    # t_start = time.time()
    # start_node_ids2 = [0] list(correct_edge_dict2.keys())[-100:]
    # nodes_ids, edges_dict_list = query_bfs(correct_edge_dict2, start_node_ids2, 4096)
    # nodes_ids, edges_dict_list = query_bfs2(correct_edge_dict, correct_edge_dict2, start_node_ids2, 4096)
    # t_end = time.time()
    # print(f'query2 cost: {t_end - t_start}')
    # print('node ids:', nodes_ids)
    # print('edge ids:', edges_dict_list)
    print('node id + edge id = ', len(nodes_ids) + len(edges_dict_list))
    # 4. Property decode
    t_start = time.time()
    # nodes_ids_sorted = sorted(nodes_ids)
    # edges_ids_sorted = sorted(edges_dict_list)
    nodes_json = json.dumps([])
    edges_json = property_decode('edge', edges_dict_list)
    if dataset == 'darpa_tc' or dataset == 'toy':
        nodes_json = property_decode('vertex', nodes_ids)
    t_end = time.time()
    print(f'property decode cost: {t_end - t_start}')
    # 5. Composite output
    nodes_json_dir = os.path.join(config.project_root, os.path.join('data/query_result', dataset + '_nodes_query.json'))
    edges_json_dir = os.path.join(config.project_root, os.path.join('data/query_result', dataset + '_edges_query.json'))
    with open(nodes_json_dir, 'w') as f:
        f.write(nodes_json)
    with open(edges_json_dir, 'w') as f:
        f.write(edges_json)
    # 6. Output total time
    program_end = time.time()
    total_cost = program_end - program_start
    # print(f'Total cost: {program_end - program_start}')
    print(f'\033[33m Query time: {total_cost:.1f}s \033[0m')
    return total_cost



if __name__ == '__main__':
    create_dir()
    # encode(dataset='leonard', edge_file='data/raw/edge200m.csv', vertex_file='data/raw/vertex200m.csv', out_zip_file_name='leonard_demo.zip')
    # encode(dataset='darpa_optc', edge_file='data/raw/AIA-101-125-top-300000.csv')
    # encode(dataset='darpa_tc', edge_file='data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv',
    #        vertex_file='data/raw/ta1-trace-3-e5-official-1.bin.1_concatenated_vertices_top_300000.csv', out_zip_file_name='darpa_tc_demo.zip')
    # print('***********************************************')
    # decode(dataset='leonard', compressed_filepathname='data/compress_result/leonard_demo.zip')
    # decode(dataset='darpa_optc', compressed_filepathname='data/compress_result/darpa_optc_demo.zip')
    decode(dataset='darpa_tc', compressed_filename='compress_result/darpa_tc_demo.zip')