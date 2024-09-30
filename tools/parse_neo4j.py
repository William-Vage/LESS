from database.async_database import AsyncNeo4jDatabase
from darpa.parser.parser import DarpaParser, get_n
from tqdm import tqdm
import asyncio


class DarpaParserNeo4j(DarpaParser):
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password):
        """
        Darpa日志解析器，输出结果到neo4j
        :param neo4j_uri: neo4j数据库地址
        :param neo4j_username: 用户名
        :param neo4j_password: 密码
        """
        super().__init__()
        self.neo4j_db = AsyncNeo4jDatabase(neo4j_uri, neo4j_username, neo4j_password)

    async def parse_vertex(self, line, bar: tqdm = None):
        """
        解析节点
        解析单行 JSON 数据并插入到 Neo4j 数据库
        :param line: JSON 数据
        :param bar: 进度条
        """
        # 获得解析结果
        vertex = await super().parse_vertex(line)
        if vertex:
            existing_node = await self.neo4j_db.get_node_by_id(vertex)
            if existing_node:
                # 节点存在，更新属性
                await self.neo4j_db.update_node_properties(vertex['id'], vertex)
            else:
                # 节点不存在，创建新节点
                await self.neo4j_db.create_node(vertex['id'], vertex)
        # 更新进度条
        if bar is not None:
            bar.update(1)

    async def parse_event(self, data):
        """
        解析 com.bbn.tc.schema.avro.cdm20.Event 类型的数据并插入到 Neo4j 数据库
        :param data: JSON 数据
        """
        # 解析节点
        event, event2 = await super().parse_event(data)

        # 创建或更新节点：subject
        subject_uuid = get_n(data, "subject", "com.bbn.tc.schema.avro.cdm20.UUID", default="")
        subject_node = await self.neo4j_db.get_node_by_id(subject_uuid)

        if subject_node is None:
            await self.neo4j_db.create_node(subject_uuid, {"id": subject_uuid, "type": "Unknown"})

        # 创建或更新节点：predicateObject
        predicate_object_uuid = get_n(data, "predicateObject", "com.bbn.tc.schema.avro.cdm20.UUID", default="")
        predicate_object_node = await self.neo4j_db.get_node_by_id(predicate_object_uuid)

        if predicate_object_node is None:
            path = get_n(data, "predicateObjectPath", "string", default=None)
            await self.neo4j_db.create_node(predicate_object_uuid, {"id": predicate_object_uuid, "path": path,
                                                                    "type": "Unknown"})

        # 创建或更新节点：predicateObject2
        predicate_object2_uuid = get_n(data, "predicateObject2", "com.bbn.tc.schema.avro.cdm20.UUID", default="")
        predicate_object2_node = await self.neo4j_db.get_node_by_id(predicate_object2_uuid)

        if predicate_object2_node is None and predicate_object2_uuid:
            path = get_n(data, "predicateObject2Path", "string", default=None)
            await self.neo4j_db.create_node(predicate_object2_uuid, {"id": predicate_object2_uuid, "path": path,
                                                                     "type": "Unknown"})

        # 创建边连接：subject 到 predicateObject
        await self.neo4j_db.create_relationship(subject_uuid, predicate_object_uuid, event)

        # 创建边连接：subject 到 predicateObject2
        if event2:
            await self.neo4j_db.create_relationship(subject_uuid, predicate_object2_uuid, event2)


async def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "zzz20001119"
    parser = DarpaParserNeo4j(uri, user, password)
    await parser.neo4j_db.drop_all()
    path = '../data/example.json'
    await parser.parse_json_file(path)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
