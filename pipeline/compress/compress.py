import config
import zipfile
import os


def compress(model_name, encode_vertex: bool, out_zip_file_path, out_file_name):
    # 文件路径
    out_zip_file_path = os.path.join(config.project_root, "data/compress_result")
    encode_dir = os.path.join(config.project_root, "data/encode")
    model_dir = os.path.join(config.project_root, "data/model")
    correct_dir = os.path.join(config.project_root, "data/correct")
    # 输出压缩文件
    # out_file_name = "output.zip"
    f = zipfile.ZipFile(os.path.join(out_zip_file_path, out_file_name), "w", zipfile.ZIP_DEFLATED)
    # 添加压缩文件
    f.write(os.path.join(encode_dir, 'edge_property.txt'), arcname='edge_property.txt')
    if encode_vertex:
        f.write(os.path.join(encode_dir, 'vertex_property.txt'), arcname='vertex_property.txt')
    f.write(os.path.join(correct_dir, 'calibration_table.txt'), arcname='calibration_table.txt')
    f.write(os.path.join(model_dir, config.model_name_appendex[model_name]), arcname=config.model_name_appendex[model_name])
    # 输出
    f.close()


def compress_only_topology(model_name):
    # 文件路径
    target_dir = os.path.join(config.project_root, "data/topology_test")
    model_dir = os.path.join(config.project_root, "data/model")
    correct_dir = os.path.join(config.project_root, "data/correct")
    # 输出压缩文件
    out_file = model_name + "_only_topology.zip"
    f = zipfile.ZipFile(os.path.join(target_dir, out_file), "w", zipfile.ZIP_DEFLATED)
    # 添加压缩文件
    f.write(os.path.join(correct_dir, 'calibration_table.txt'), arcname='calibration_table.txt')
    f.write(os.path.join(model_dir, config.model_name_appendex[model_name]), arcname=config.model_name_appendex[model_name])
    # 输出
    f.close()
    zip_file_size = os.path.getsize(os.path.join(target_dir, out_file))
    print(f'zip file size: {zip_file_size}')


if __name__ == '__main__':
    compress()
