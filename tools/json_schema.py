import os
import json
from tqdm import tqdm


def update_json_tree(json_data: dict, tree=None):
    """
    输入json，维护json结构树。
    每当遇到新结构时，修改树中的字段。
    :param json_data: 输入的json数据
    :param tree: 已有json结构树
    :return: 更新后的json结构树
    """
    if tree is None:
        tree = {}
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            type_name = type(value).__name__
            # 1.如果是字典，递归处理
            if isinstance(value, dict):
                # 新增分支
                if key not in tree:
                    tree[key] = {}
                # 如果原始分支是一个值，扩展成字典
                if tree[key] is None:
                    tree[key] = {'NoneType': None}
                elif isinstance(tree[key], str):
                    tree[key] = {tree[key]: None}
                # 递归更新
                update_json_tree(value, tree[key])
            # 2.如果是列表，列表下必须是同一结构（先验）
            elif isinstance(value, list) and value:
                if key not in tree:
                    tree[key] = []
                if len(tree[key]) == 0:
                    if len(value) > 0:
                        if isinstance(value[0], dict):
                            tree[key] = [update_json_tree(value[0])]
                        else:
                            tree[key] = [type(value[0]).__name__]
            # 3.是元素
            else:
                # 原本的树为空
                if key not in tree:
                    # None特殊处理
                    if value is None:
                        tree[key] = None
                    else:
                        tree[key] = type_name
                # 树中至少一个元素，向树中添加字段
                else:
                    existing_type = tree[key] if tree[key] is not None else 'NoneType'
                    if isinstance(existing_type, dict):
                        tree[key][type_name] = None
                    elif existing_type != type_name:
                        tree[key] = {existing_type: None, type_name: None}
    return tree


def update_tree_with_file(file_path: str, tree: dict = None, is_jsonl=True):
    """
    读取json文件，更新json结构树
    :param file_path: json文件路径
    :param tree: 已有json结构树
    :param is_jsonl: 是否为jsonl
    :return: json结构树
    """
    if tree is None:
        tree = {}
    with open(file_path, "r") as json_file:
        if is_jsonl:
            for line in tqdm(json_file, desc='Processing json lines'):
                json_object = json.loads(line)
                tree = update_json_tree(json_object, tree)
        else:
            json_objects = json.load(json_file)
            for i in json_objects:
                tree = update_json_tree(i, tree)
    return tree


if __name__ == '__main__':
    # JSON 数据
    jsonl_file_paths = ['../data/raw/audit.json']
    # JSON 结构树
    json_tree = None
    # 更新树结构
    for i in jsonl_file_paths:
        json_tree = update_tree_with_file(i, json_tree, False)
    # 打印更新后的树结构
    s = json.dumps(json_tree, indent=2)
    print(s)
    # 保存到文件
    with open('tree_structure.json', 'w') as f:
        f.write(s)
