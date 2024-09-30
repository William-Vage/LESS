import json
import asyncio
from tqdm import tqdm


def get_n(data, *keys, default=None):
    """
    递归获取嵌套字典中的值。
    :param data: 要检索的字典
    :param keys: 以层次结构提供的键
    :param default: 缺失值处理
    :return: 获取的值或None（如果未找到）
    """
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


class DarpaParser:
    def __init__(self):
        """
        Darpa日志解析器
        """

    @staticmethod
    async def parse_event(data) -> tuple[dict, dict]:
        """
        解析 com.bbn.tc.schema.avro.cdm20.Event 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid event_data format"

        # 解析事件信息
        event = {
            "id": get_n(data, "uuid"),
            "subject_id": get_n(data, "subject", "com.bbn.tc.schema.avro.cdm20.UUID", default=""),
            "predicate_id": get_n(data, "predicateObject", "com.bbn.tc.schema.avro.cdm20.UUID", default=""),
            "sequence": get_n(data, "sequence", "long", default=0),
            "type": get_n(data, "type", default=""),
            "threadId": get_n(data, "threadId", "int", default=0),
            "timestampNanos": get_n(data, "timestampNanos", default=0),
            "size": get_n(data, "size", "long", default=None),
        }
        # 提取属性信息
        properties_data = get_n(data, "properties", "map", default={})
        properties = {
            "flags": get_n(properties_data, "flags"),
            "opm": get_n(properties_data, "opm"),
            "protection": get_n(properties_data, "protection"),
            "signal": get_n(properties_data, "signal"),
            "mode": get_n(properties_data, "mode"),
            "operation": get_n(properties_data, "operation")
        }
        # 将 properties 信息作为属性附加到 event 节点上
        event.update(properties)
        # 检查是否存在predicateObject2
        predicate_object2_uuid = get_n(data, "predicateObject2", "com.bbn.tc.schema.avro.cdm20.UUID", default="")
        if len(predicate_object2_uuid) == 0:
            return event, dict()
        else:
            event2 = event.copy()
            event2['predicate_id'] = predicate_object2_uuid
            return event, event2

    @staticmethod
    async def parse_src_sink_object(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.SrcSinkObject 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid src_sink_object_data format"

        # 解析 SrcSinkObject 信息
        src_sink_object = {
            "id": get_n(data, "uuid"),
            "type": "SrcSinkObject",
            "internalType": get_n(data, "type"),
            "fileDescriptor": get_n(data, "fileDescriptor", "int", default=0),
        }

        # 解析 baseObject 信息
        base_object_data = get_n(data, "baseObject", default={})
        base_object = {
            "epoch": get_n(base_object_data, "epoch", "int", default=0),
            "pid": get_n(base_object_data, "properties", "map", "pid"),
        }

        # 将 baseObject 信息作为属性附加到 src_sink_object 节点上
        src_sink_object.update(base_object)

        return src_sink_object

    @staticmethod
    async def parse_net_flow_object(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.NetFlowObject 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid net_flow_object_data format"
        # 解析 NetFlowObject 信息
        net_flow_object = {
            "id": data.get("uuid", ""),
            "type": "NetFlowObject",
            "localAddress": get_n(data, "localAddress", "string", default=""),
            "localPort": get_n(data, "localPort", "int", default=0),
            "remoteAddress": get_n(data, "remoteAddress", "string", default=""),
            "remotePort": get_n(data, "remotePort", "int", default=0),
            "ipProtocol": get_n(data, "ipProtocol", "int", default=0),
        }
        # 解析 baseObject 信息
        base_object_data = data.get("baseObject", {})
        base_object = {
            "epoch": get_n(base_object_data, "epoch", "int", default=0),
        }
        # 将 baseObject 信息作为属性附加到 net_flow_object 节点上
        net_flow_object.update(base_object)
        return net_flow_object

    @staticmethod
    async def parse_file_object(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.FileObject 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid file_object_data format"

        # 解析 FileObject 信息
        file_object = {
            "id": data.get("uuid", ""),
            "type": "FileObject",
            "internalType": data.get("type", ""),
        }
        # 解析 baseObject 信息
        base_object_data = data.get("baseObject", {})
        base_object = {
            "permission": get_n(base_object_data, "permission", "com.bbn.tc.schema.avro.cdm20.SHORT", default=None),
            "epoch": get_n(base_object_data, "epoch", "int", default=0),
            "path": get_n(base_object_data, "properties", "map", "path", default="")
        }
        # 将 baseObject 信息作为属性附加到 file_object 节点上
        file_object.update(base_object)
        return file_object

    @staticmethod
    async def parse_subject(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.Subject 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid subject_data format"

        # 解析 Subject 信息
        subject = {
            "id": get_n(data, "uuid"),
            "type": "Subject",
            "internalType": get_n(data, "type"),
            "cid": get_n(data, "cid", default=0),
            "parentSubject": get_n(data, "parentSubject", "com.bbn.tc.schema.avro.cdm20.UUID", default=None),
            "localPrincipal": get_n(data, "localPrincipal", "com.bbn.tc.schema.avro.cdm20.UUID", default=""),
            "startTimestampNanos": get_n(data, "startTimestampNanos", "long", default=0),
            "unitId": get_n(data, "unitId", "int", default=0),
            "iteration": get_n(data, "iteration", "int", default=None),
            "count": get_n(data, "count", "int", default=None),
            "cmdLine": get_n(data, "cmdLine", "string", default=None),
        }

        # 解析 properties 信息
        properties_data = get_n(data, "properties", "map", default={})
        properties = {
            "name": get_n(properties_data, "name", default=""),
            "cwd": get_n(properties_data, "cwd", default=""),
            "ppid": get_n(properties_data, "ppid", default=""),
            "seen_time": get_n(properties_data, "seen time", default=""),
        }

        # 将 properties 信息作为属性附加到 subject 节点上
        subject.update(properties)

        return subject

    @staticmethod
    async def parse_memory_object(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.Subject 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid memory_object format"

        # 解析 MemoryObject 信息
        memory_object = {
            "id": get_n(data, "uuid"),
            "type": "MemoryObject",
            "memoryAddress": get_n(data, "memoryAddress", default=0),
            "size": get_n(data, "memoryAddress", default=0),
            "tgid": get_n(data, "baseObject", "properties", "map", "tgid", default="")
        }

        return memory_object

    @staticmethod
    async def parse_ipc_object(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.IpcObject 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid ipc_object_data format"
        # 解析 IpcObject 信息
        ipc_object = {
            "id": data.get("uuid", ""),
            "type": "IpcObject",
            "internalType": data.get("type", ""),
            "fd1": get_n(data, "fd1", "int", default=None),
            "fd2": get_n(data, "fd2", "int", default=None)
        }
        # 解析 baseObject 信息
        base_object_data = data.get("baseObject", {})
        base_object = {
            "permission": get_n(base_object_data, "permission", "com.bbn.tc.schema.avro.cdm20.SHORT", default=""),
            "epoch": get_n(base_object_data, "epoch", "int", default=0),
        }
        # 解析 properties 信息
        properties_data = get_n(base_object_data, "properties", "map", default={})
        properties = {
            "pid": properties_data.get("pid", ""),
            "subtype": properties_data.get("subtype", ""),
            "path": properties_data.get("path", ""),
        }
        # 将 properties 信息作为属性附加到 base_object 字典上
        ipc_object.update(properties)
        # 将 baseObject 信息作为属性附加到 ipc_object 节点上
        ipc_object.update(base_object)
        return ipc_object

    @staticmethod
    async def parse_principal(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.Principal 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid principal_data format"

        # 解析 Principal 信息
        principal = {
            "id": get_n(data, "uuid"),
            "type": "Principal",
            "internalType": get_n(data, "type"),
            "userId": get_n(data, "userId"),
        }

        # 解析 groupIds 信息
        group_ids_data = get_n(data, "groupIds", "array", default=[])
        group_ids = [str(group_id) for group_id in group_ids_data]

        # 解析 properties 信息
        properties_data = get_n(data, "properties", "map", default={})
        euid = get_n(properties_data, "euid")

        # 将 groupIds 和 properties 信息作为属性附加到 principal 节点上
        principal.update({"groupIds": group_ids, "euid": euid})

        return principal

    @staticmethod
    async def parse_host(data) -> dict:
        """
        解析 com.bbn.tc.schema.avro.cdm20.Host 类型的数据
        :param data: JSON 数据
        """
        assert isinstance(data, dict), "Invalid host_data format"

        # 解析 Host 信息
        host = {
            "id": get_n(data, "uuid"),
            "type": "Host",
            "hostName": get_n(data, "hostName"),
            "ta1Version": get_n(data, "ta1Version"),
            "hostType": get_n(data, "hostType"),
            "osDetails": get_n(data, "osDetails", "string")
        }
        # 解析 hostIdentifiers 信息
        host_identifiers_data = get_n(data, "hostIdentifiers", "array") or []
        host_identifiers_id_type = [get_n(identifier, "idType") for identifier in host_identifiers_data]
        host_identifiers_id_value = [get_n(identifier, "idValue") for identifier in host_identifiers_data]
        # 解析 interfaces 信息
        interfaces_data = get_n(data, "interfaces", "array") or []
        interfaces_name = [get_n(interface, "name") for interface in interfaces_data]
        interfaces_mac_address = [get_n(interface, "macAddress") for interface in interfaces_data]
        interfaces_ip_addresses = [
            ",".join(get_n(interface, "ipAddresses", "array") or []) for interface in interfaces_data
        ]
        # 将 host_identifiers、os_details 和 interfaces 信息作为属性附加到 host 节点上
        host.update({"hostIdentifiersIdType": host_identifiers_id_type,
                     "hostIdentifiersIdValue": host_identifiers_id_value, "interfacesName": interfaces_name,
                     "interfacesMacAddress": interfaces_mac_address, "interfacesIpAddresses": interfaces_ip_addresses})
        return host

    async def parse_time_marker(self, data):
        """
        解析 com.bbn.tc.schema.avro.cdm20.TimeMarker
        :param data: JSON数据
        :return:
        """
        pass

    async def parse_vertex(self, line, bar: tqdm = None) -> dict | None:
        """
        解析节点，单行 JSON 数据
        :param line: JSON 数据
        :param bar: 进度条
        """
        try:
            data = json.loads(line)
        except Exception as e:
            print(line)
            print(e)
            return
        assert isinstance(data["datum"], dict), "Invalid data format"

        cdm_type = next(iter(data["datum"]))
        assert cdm_type.startswith("com.bbn.tc.schema.avro.cdm20."), "Invalid CDM type"
        cdm_subtype = cdm_type[29:]  # 提取 xxx 部分

        if cdm_subtype == "SrcSinkObject":
            res = await self.parse_src_sink_object(data["datum"][cdm_type])
        elif cdm_subtype == "NetFlowObject":
            res = await self.parse_net_flow_object(data["datum"][cdm_type])
        elif cdm_subtype == "FileObject":
            res = await self.parse_file_object(data["datum"][cdm_type])
        elif cdm_subtype == "Subject":
            res = await self.parse_subject(data["datum"][cdm_type])
        elif cdm_subtype == "MemoryObject":
            res = await self.parse_memory_object(data["datum"][cdm_type])
        elif cdm_subtype == "IpcObject":
            res = await self.parse_ipc_object(data["datum"][cdm_type])
        elif cdm_subtype == "Principal":
            res = await self.parse_principal(data["datum"][cdm_type])
        elif cdm_subtype == "Host":
            res = await self.parse_host(data["datum"][cdm_type])
        else:
            res = None
        # 更新进度条
        if bar is not None:
            bar.update(1)
        return res

    async def parse_edge(self, line, bar: tqdm = None) -> tuple[dict, dict] | None:
        """
        解析边，单行 JSON 数据
        :param line: JSON 数据
        :param bar: 进度条
        """
        data = json.loads(line)

        assert isinstance(data["datum"], dict), "Invalid data format"

        cdm_type = next(iter(data["datum"]))
        assert cdm_type.startswith("com.bbn.tc.schema.avro.cdm20."), "Invalid CDM type"
        cdm_subtype = cdm_type[29:]  # 提取 xxx 部分

        if cdm_subtype == "Event":
            res = await self.parse_event(data["datum"][cdm_type])
        else:
            res = None

        # 更新进度条
        if bar is not None:
            bar.update(1)
        return res

    async def parse_json_file(self, file_path, batch_size: int = 50000):
        """
        从文件中读取 JSON 数据并解析插入到 Neo4j 数据库
        :param file_path: JSONL 文件路径
        :param batch_size: 同时最大工作数量
        """
        # 先解析点，再解析边，保证拓扑有序
        with open(file_path, "r") as file:
            bar = tqdm(desc='Processing vertex')
            task = []
            line = file.readline()
            while line:
                task.append(self.parse_vertex(line, bar))
                if len(task) >= batch_size:
                    await asyncio.gather(*task)
                    task.clear()
                line = file.readline()
            await asyncio.gather(*task)
        with open(file_path, "r") as file:
            bar = tqdm(desc='Processing edges')
            task = []
            line = file.readline()
            while line:
                task.append(self.parse_edge(line, bar))
                if len(task) >= batch_size:
                    await asyncio.gather(*task)
                    task.clear()
                line = file.readline()
            await asyncio.gather(*task)


class DarpaoptcParser:
    def __init__(self):
        """
        Darpaoptc日志解析器
        """

    async def parse_json_file(self, file_path, batch_size: int = 50000):
        """
        从文件中读取 JSON 数据并解析插入到 Neo4j 数据库
        :param file_path: JSONL 文件路径
        :param batch_size: 同时最大工作数量
        """
        # 先解析点，再解析边，保证拓扑有序
        with open(file_path, "r") as file:
            bar = tqdm(desc='Processing...')
            task = []
            line = file.readline()
            while line:
                task.append(self.parse_line(line, bar))
                if len(task) >= batch_size:
                    await asyncio.gather(*task)
                    task.clear()
                line = file.readline()
            await asyncio.gather(*task)

    async def parse_line(self, line, bar):
        """
        解析节点，单行 JSON 数据
        :param line: JSON 数据
        :param bar: 进度条
        """

        def reorder_keys(dictionary):
            keys = ['id', 'subject_id', 'predicate_id']
            return {k: dictionary[k] for k in keys} | {k: dictionary[k] for k in dictionary if k not in keys}

        data: dict = json.loads(line)
        data['subject_id'] = data.pop('actorID')
        data['predicate_id'] = data.pop('objectID')
        data.update(data['properties'])
        data.pop('properties')
        data = reorder_keys(data)
        # 更新进度条
        if bar is not None:
            bar.update(1)

        return data
