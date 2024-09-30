import json

import Levenshtein
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


def property_decode(dtype: str, ids: list[int]):
    decode_vertex_properties, decode_edge_properties = {}, {}
    # 读取编码结果
    if dtype == 'edge':
        edge_encode_file = os.path.join(config.project_root, 'data/encode/edge_property.txt')
        with open(edge_encode_file, 'rb') as f:
            encoded_edge_properties = f.read().split(bytes([0xff]))
        # 解码
        for i in range(len(ids)):
            if len(encoded_edge_properties[i]) > 0:
                decode_edge_properties[ids[i]] = decode_line(encoded_edge_properties, ids[i]) + '\n'
        # 输出解码结果
        edge_decode_file = os.path.join(config.project_root, 'data/decode/edge_property.csv')
        with open(edge_decode_file, 'w') as f:
            f.writelines(json.dumps(decode_edge_properties))
        return decode_edge_properties
    elif dtype == 'vertex':
        vertex_encode_file = os.path.join(config.project_root, 'data/encode/vertex_property.txt')
        with open(vertex_encode_file, 'rb') as f:
            encoded_vertex_properties = f.read().split(bytes([0xff]))
        for i in range(len(ids)):
            if len(encoded_vertex_properties[i]) > 0:
                decode_vertex_properties[ids[i]] = decode_line(encoded_vertex_properties, ids[i]) + '\n'
        vertex_decode_file = os.path.join(config.project_root, 'data/decode/vertex_property.csv')
        with open(vertex_decode_file, 'w') as f:
            f.writelines(json.dumps(decode_vertex_properties))
        return decode_vertex_properties


if __name__ == '__main__':
    property_decode()

