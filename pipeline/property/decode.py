import json
from io import StringIO
import pandas as pd
import config
import struct
import os


def transform(op: str, s1: str, pos: int, s2):
    if op == 'insert':
        s = s1[:pos] + s2 + s1[pos:]
    elif op == 'replace':
        s = s1[:pos] + s2 + s1[pos + len(s2):]
    elif op == 'delete':
        s = s1[:pos] + s1[pos + s2:]
    else:
        raise Exception(f'{op} is not exist.')
    return s


def decode_line(lines: list[bytes], loc) -> str:
    line = lines[loc]
    # 从第0位开始读字符
    p = 0
    v = line[p] - 127
    # 递归取得锚点字符串
    if v != 0:
        res = decode_line(lines, loc + v)
    else:
        res = ''
    p += 1
    # 修正偏移量
    offset = 0
    # 遍历操作
    while p < len(line):
        # 读取操作
        op = line[p]
        p += 1
        # 操作位置
        pos = line[p]
        p += 1
        if pos == 0xfe:
            pos = struct.unpack('<H', line[p: p+2])[0]
            p += 2
        # delete
        if op == 0xfd:
            # 如果是删除，且删除的是一个区间
            s2 = line[p]
            p += 1
            res = transform('delete', res, pos + offset, s2)
            offset -= s2
            # print('delete', pos, s2)
        else:
            s2 = ''
            while p < len(line) and line[p] <= 0xfa:
                s2 += chr(line[p])
                p += 1
            # insert
            if op == 0xfb:
                res = transform('insert', res, pos + offset, s2)
                offset += len(s2)
                # print('insert', pos, s2)
            # replace
            elif op == 0xfc:
                res = transform('replace', res, pos + offset, s2)
                # print('replace', pos, s2)
        # print(res)
    # print('---------------------------------------------')
    return res


def transform_json(headers: list[str], csv_str: str, dtype: str, extra_info: list):
    df = pd.read_csv(StringIO(csv_str), names=headers)
    records = df.to_dict(orient='records')
    clean_records = []
    for i, record in enumerate(records):
        value = {}
        for k, v in record.items():
            if pd.notnull(v):
                value[k] = v
        if dtype == 'edge':
            value['parentVertexHash'] = str(extra_info[i][0])
            value['childVertexHash'] = str(extra_info[i][1])
        elif dtype == 'vertex':
            value['hash'] = str(extra_info[i])
        clean_records.append(value)
    for i in range(len(records), len(extra_info)):
        clean_records.append({'hash': str(extra_info[i])})
    json_data = json.dumps(clean_records, ensure_ascii=False, indent=4)
    return json_data


def property_decode(dtype: str, ids: list):
    decode_vertex_properties, decode_edge_properties = '', ''
    extra_info = []
    # 读取编码结果
    if dtype == 'edge':
        edge_encode_file = os.path.join(config.project_root, 'data/encode/edge_property.txt')
        with open(edge_encode_file, 'rb') as f:
            encoded_edge_properties = f.read().split(bytes([0xff]))
        # 解码
        headers = decode_line(encoded_edge_properties, 0).split(',')
        for i in ids:
            for k, v in i.items():
                if len(encoded_edge_properties[k]) > 0:
                    decode_edge_properties += decode_line(encoded_edge_properties, k + 1) + '\n'
                    extra_info.append(v)
        # 输出解码结果
        edge_decode_file = os.path.join(config.project_root, 'data/decode/edge_property.json')
        decode_json = transform_json(headers, decode_edge_properties, 'edge', extra_info)
        with open(edge_decode_file, 'w') as f:
            f.writelines(decode_json)
        return decode_json
    elif dtype == 'vertex':
        vertex_encode_file = os.path.join(config.project_root, 'data/encode/vertex_property.txt')
        with open(vertex_encode_file, 'rb') as f:
            encoded_vertex_properties = f.read().split(bytes([0xff]))
        headers = decode_line(encoded_vertex_properties, 0).split(',')
        for i in range(len(ids)):
            if len(encoded_vertex_properties[i]) > 0 and ids[i] + 1 < len(encoded_vertex_properties):
                decode_vertex_properties += decode_line(encoded_vertex_properties, ids[i] + 1) + '\n'
        vertex_decode_file = os.path.join(config.project_root, 'data/decode/vertex_property.json')
        decode_json = transform_json(headers, decode_vertex_properties, 'vertex', ids)
        with open(vertex_decode_file, 'w') as f:
            f.writelines(decode_json)
        return decode_json


if __name__ == '__main__':
    property_decode()

