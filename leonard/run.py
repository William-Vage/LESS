from preprocess.leonard_preprocess import leonard_preprocess_func
from train.leonard_train import leonard_train_func
from correct.leonard_correct import leonard_correct_func
import zipfile
import os
import leonard.config as config
import shutil
from query_withedge_vertex_removeedge_darpa_batch import leonard_query_run

def create_dir():
    d = os.path.join(config.project_root, 'data')
    if not os.path.exists(d):
        os.makedirs(d)
    dir_list = ['raw', 'preprocess', 'model', 'encode', 'correct', 'compress', 'query_result']
    for dir_item in dir_list:
        dir_item = os.path.join(d, dir_item)
        if not os.path.exists(dir_item):
            os.makedirs(dir_item)


def clear_dir():
    dir_root = os.path.join(config.project_root, 'data')
    dir_list = os.listdir(dir_root)
    skip_dir = ['raw', 'compress', 'query_result']
    for i in dir_list:
        folder_path = os.path.join(dir_root, i)
        if i in skip_dir:
            continue
        # 确保要删除的路径是一个文件夹
        if not os.path.isdir(folder_path):
            continue
        # 删除文件夹下的所有内容
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # 判断路径是文件还是文件夹
                # 删除文件
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # 递归删除子文件夹及其内容
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除 {folder_path} 下的内容时发生错误：{e}")


def leonard_run(dataset='toy'):
    if dataset == 'darpa_tc':
        edges_list = ['ta1-trace-3-e5-official-1.bin.1_concatenated_edges_top_300000.csv']
        vertices_list = ['ta1-trace-3-e5-official-1.bin.1_concatenated_vertices_top_300000.csv']
    elif dataset == 'toy':
        edges_list = ['edge200m.csv']
        vertices_list = ['vertex200m.csv']

    edge_file = os.path.join(config.project_root, 'data/raw/' + edges_list[0])
    vertex_file = os.path.join(config.project_root, 'data/raw/' + vertices_list[0])
    leonard_preprocess_func(leonard_edge_file=edge_file, leonard_vertex_file=vertex_file, dataset=dataset)
    leonard_train_func()
    leonard_correct_func()
    out_zip_file = os.path.join(config.project_root, 'data/compress/leonard_' + dataset + '_output.zip')
    leonard_zip_output_files(out_zip_file)
    leonard_query_run(dataset)


def leonard_zip_output_files(out_zip_file):
    model_lite = os.path.join(config.project_root, "data/model/leonard_lstm.pt")
    params_file = os.path.join(config.project_root, "data/encode/params.json")
    table_file = os.path.join(config.project_root, "data/correct/my_calibration_table.json")
    edge_file = os.path.join(config.project_root, "data/encode/edges.txt")
    files_to_compress = [model_lite, params_file, table_file, edge_file]

    with zipfile.ZipFile(out_zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_compress:
            file_path = os.path.abspath(file)
            zipf.write(file_path, os.path.basename(file_path))


if __name__ == '__main__':
    # create_dir()
    leonard_run()
