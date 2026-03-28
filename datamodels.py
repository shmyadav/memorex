




from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any
from time import time
from embedder import EmbedderClient
from typing_extensions import LiteralString
from neo4j import time as neo4j_time
from embedder import OpenAIEmbedder
import asyncio
from enum import Enum
from llm import *
from graph_driver import GraphDriver
from dataclasses import dataclass, field
from collections import defaultdict
from typing import final
from reranker import CrossEncoderClient


from typing_extensions import Buffer

####################nodes related stuff#######################
# +++
####################llm related stuff#######################
# +++
####################driver related stuff#######################



MAX_SUMMARY_CHARS = 500
class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'

class Message(BaseModel):
    role: str
    content: str


def utc_now() -> datetime:
    """Returns the current UTC datetime with timezone information."""
    return datetime.now(timezone.utc)


class EntitySummary(BaseModel):
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )

class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EpisodeType(Enum):
    """
    Enumeration of different types of episodes that can be processed.

    This enum defines the various sources or formats of episodes that the system
    can handle. It's used to categorize and potentially handle different types
    of input data differently.

    Attributes:
    -----------
    message : str
        Represents a standard message-type episode. The content for this type
        should be formatted as "actor: content". For example, "user: Hello, how are you?"
        or "assistant: I'm doing well, thank you for asking."
        Represents an episode containing a JSON string object with structured data.
    json : str
    text : str
        Represents a plain text episode.
    """

    message = 'message'
    json = 'json'
    text = 'text'

    @staticmethod
    def from_str(episode_type: str):
        if episode_type == 'message':
            return EpisodeType.message
        if episode_type == 'json':
            return EpisodeType.json
        if episode_type == 'text':
            return EpisodeType.text
        # logger.error(f'Episode type: {episode_type} not implemented')
        raise NotImplementedError


class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())


    async def delete(self, driver: GraphDriver):        
        records, _, _ = await driver.execute_query(
            """
            MATCH (n {uuid: $uuid})
            WHERE n:Entity OR n:Episodic OR n:Community
            OPTIONAL MATCH (n)-[r]-()
            WITH collect(r.uuid) AS edge_uuids, n
            DETACH DELETE n
            RETURN edge_uuids
            """,
            uuid=self.uuid,
        )


    @classmethod
    async def delete_by_group_id(cls, driver: GraphDriver, group_id: str, batch_size: int = 100):
        async with driver.session() as session:
            await session.run(
                """
                MATCH (n:Entity|Episodic|Community {group_id: $group_id})
                CALL (n) {
                    DETACH DELETE n
                } IN TRANSACTIONS OF $batch_size ROWS
                """,
                group_id=group_id,
                batch_size=batch_size,
            )


    @classmethod
    async def delete_by_uuids(cls, driver: GraphDriver, uuids: list[str], batch_size: int = 100):
        async with driver.session() as session:
            # Collect all edge UUIDs before deleting nodes
            await session.run(
                """
                MATCH (n:Entity|Episodic|Community)
                WHERE n.uuid IN $uuids
                MATCH (n)-[r]-()
                RETURN collect(r.uuid) AS edge_uuids
                """,
                uuids=uuids,
            )

            # Now delete the nodes in batches
            await session.run(
                """
                MATCH (n:Entity|Episodic|Community)
                WHERE n.uuid IN $uuids
                CALL (n) {
                    DETACH DELETE n
                } IN TRANSACTIONS OF $batch_size ROWS
                """,
                uuids=uuids,
                batch_size=batch_size,
            )




class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None, description='embedding of the name')
    summary: str = Field(description='regional summary of surrounding edges', default_factory=str)
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the node. Dependent on node labels'
    )

    async def generate_name_embedding(self, embedder: EmbedderClient):
        start = time()
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
        end = time()

        return self.name_embedding

    async def load_name_embedding(self, driver: GraphDriver):
        
        
        query: LiteralString = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN n.name_embedding AS name_embedding
        """
        
        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            print(f'Node not found: {self.uuid}')
            raise Exception(f'Node not found: {self.uuid}')

        self.name_embedding = records[0]['name_embedding']

    


    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity {uuid: $uuid})
            RETURN
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at,
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
            """,
            uuid=uuid,
            routing_='r',
        )

        nodes = [cls.get_entity_node_from_record(record) for record in records]

        if len(nodes) == 0:
            raise Exception("Nodes not found")

        return nodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at,
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
            """,
            uuids=uuids,
            routing_='r',
        )

        nodes = [cls.get_entity_node_from_record(record) for record in records]

        return nodes

    @classmethod
    def get_entity_node_from_record(cls, record: Any):
        attributes = record['attributes']
        attributes.pop('uuid', None)
        attributes.pop('name', None)
        attributes.pop('group_id', None)
        attributes.pop('name_embedding', None)
        attributes.pop('summary', None)
        attributes.pop('created_at', None)
        attributes.pop('labels', None)

        labels = record.get('labels', [])
        group_id = record.get('group_id')
        if 'Entity_' + group_id.replace('-', '') in labels:
            labels.remove('Entity_' + group_id.replace('-', ''))

        entity_node = cls(
            uuid=record['uuid'],
            name=record['name'],
            name_embedding=record.get('name_embedding'),
            group_id=group_id,
            labels=labels,
            created_at=parse_db_date(record['created_at']),  # type: ignore
            summary=record['summary'],
            attributes=attributes,
        )

        return entity_node



def parse_db_date(input_date: neo4j_time.DateTime | str | None) -> datetime | None:
    if isinstance(input_date, neo4j_time.DateTime):
        return input_date.to_native()

    if isinstance(input_date, str):
        return datetime.fromisoformat(input_date)

    return input_date


def get_episodic_node_from_record(record: Any):
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
    )




class EpisodicNode(Node):
    source: EpisodeType = Field(description='source type')
    source_description: str = Field(description='description of the data source')
    content: str = Field(description='raw episode data')
    valid_at: datetime = Field(
        description='datetime of when the original document was created',
    )
    entity_edges: list[str] = Field(
        description='list of entity edges referenced in this episode',
        default_factory=list,
    )

    async def save(self, driver: GraphDriver):
        episode_args = {
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'source_description': self.source_description,
            'content': self.content,
            'entity_edges': self.entity_edges,
            'created_at': self.created_at,
            'valid_at': self.valid_at,
            'source': self.source.value,
        }
        episode_node_save_query = """
                    MERGE (n:Episodic {uuid: $uuid})
                    SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                    entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
                    RETURN n.uuid AS uuid
                """
        result = await driver.execute_query(
            episode_node_save_query, **episode_args
        )
        print("Added to graph")

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.created_at AS created_at,
            e.source AS source,
            e.source_description AS source_description,
            e.content AS content,
            e.valid_at AS valid_at,
            e.entity_edges AS entity_edges
            """,
            uuid=uuid,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        if len(episodes) == 0:
            print("****no nodes******")
            raise 

        return episodes[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.uuid IN $uuids
            RETURN DISTINCT
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.created_at AS created_at,
            e.source AS source,
            e.source_description AS source_description,
            e.content AS content,
            e.valid_at AS valid_at,
            e.entity_edges AS entity_edges
            """,
            uuids=uuids,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes



    @classmethod
    async def get_by_entity_node_uuid(cls, driver: GraphDriver, entity_node_uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN DISTINCT
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.created_at AS created_at,
            e.source AS source,
            e.source_description AS source_description,
            e.content AS content,
            e.valid_at AS valid_at,
            e.entity_edges AS entity_edges
            """,
            entity_node_uuid=entity_node_uuid,
            routing_='r',
        )

        episodes = [get_episodic_node_from_record(record) for record in records]

        return episodes


class MemorexClients(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Allow non-Pydantic runtime objects (drivers / clients) as fields
    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient




###################edge related stuff########################

class Edge(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str = Field(description='partition of the graph')
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

    @abstractmethod
    async def save(self, driver: GraphDriver): ...

    async def delete(self, driver: GraphDriver):
        
        await driver.execute_query(
            """
            MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
            DELETE e
            """,
            uuid=self.uuid,
        )

        print(f'Deleted Edge: {self.uuid}')

    @classmethod
    async def delete_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        
        await driver.execute_query(
            """
            MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(m)
            WHERE e.uuid IN $uuids
            DELETE e
            """,
            uuids=uuids,
        )

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False


class EpisodicEdge(Edge):

    EPISODIC_EDGE_SAVE : ClassVar[str] = """
        MATCH (episode:Episodic {uuid: $episode_uuid})
        MATCH (node:Entity {uuid: $entity_uuid})
        MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
        SET
            e.group_id = $group_id,
            e.created_at = $created_at
        RETURN e.uuid AS uuid"""
    
    EPISODIC_EDGE_RETURN: ClassVar[str] = """
        e.uuid AS uuid,
        e.group_id AS group_id,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.created_at AS created_at
        """


    async def save(self, driver: GraphDriver):
        result = await driver.execute_query(
            self.EPISODIC_EDGE_SAVE,
            episode_uuid=self.source_node_uuid,
            entity_uuid=self.target_node_uuid,
            uuid=self.uuid,
            group_id=self.group_id,
            created_at=self.created_at,
        )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Episodic)-[e:MENTIONS {uuid: $uuid}]->(m:Entity)
            RETURN
            """
            + cls.EPISODIC_EDGE_RETURN,
            uuid=uuid,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise ValueError(f"Edge with UUID {uuid} not found")
        return edges[0]

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
            WHERE e.uuid IN $uuids
            RETURN
            """
            + cls.EPISODIC_EDGE_RETURN,
            uuids=uuids,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise ValueError(f"Edge with UUID {uuids[0]} not found")
        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

        records, _, _ = await driver.execute_query(
            """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + cls.EPISODIC_EDGE_RETURN
            + """
            ORDER BY e.uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        edges = [get_episodic_edge_from_record(record) for record in records]

        if len(edges) == 0:
            raise ValueError(f"No edges found for group IDs {group_ids}")
        return edges


class EntityEdge(Edge):
    name: str = Field(description='name of the edge, relation name')
    fact: str = Field(description='fact representing the edge and nodes that it connects')
    fact_embedding: list[float] | None = Field(default=None, description='embedding of the fact')
    episodes: list[str] = Field(
        default=[],
        description='list of episode ids that reference these entity edges',
    )
    expired_at: datetime | None = Field(
        default=None, description='datetime of when the node was invalidated'
    )
    valid_at: datetime | None = Field(
        default=None, description='datetime of when the fact became true'
    )
    invalid_at: datetime | None = Field(
        default=None, description='datetime of when the fact stopped being true'
    )
    attributes: dict[str, Any] = Field(
        default={}, description='Additional attributes of the edge. Dependent on edge name'
    )

    async def generate_embedding(self, embedder: EmbedderClient):
        start = time()

        text = self.fact.replace('\n', ' ')
        self.fact_embedding = await embedder.create(input_data=[text])

        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')

        return self.fact_embedding

    async def load_fact_embedding(self, driver: GraphDriver):
        if driver.graph_operations_interface:
            return await driver.graph_operations_interface.edge_load_embeddings(self, driver)

        query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            RETURN e.fact_embedding AS fact_embedding
        """

        records, _, _ = await driver.execute_query(
            query,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise Exception(f"Edge with UUID {self.uuid} not found")

        self.fact_embedding = records[0]['fact_embedding']

    async def save(self, driver: GraphDriver):
        edge_data: dict[str, Any] = {
            'source_uuid': self.source_node_uuid,
            'target_uuid': self.target_node_uuid,
            'uuid': self.uuid,
            'name': self.name,
            'group_id': self.group_id,
            'fact': self.fact,
            'fact_embedding': self.fact_embedding,
            'episodes': self.episodes,
            'created_at': self.created_at,
            'expired_at': self.expired_at,
            'valid_at': self.valid_at,
            'invalid_at': self.invalid_at,
        }

        edge_data.update(self.attributes or {})
        result = await driver.execute_query(
            """
            MATCH (source:Entity {uuid: $edge_data.source_uuid})
            MATCH (target:Entity {uuid: $edge_data.target_uuid})
            MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
            SET e = $edge_data
            WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)
            RETURN e.uuid AS uuid
                """,
            edge_data=edge_data,
        )

        logger.debug(f'Saved edge to Graph: {self.uuid}')

        return result

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
        """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            RETURN
            """
            + """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        properties(e) AS attributes""",
            uuid=uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        if len(edges) == 0:
            raise Exception(f"Edge with UUID {uuid} not found")
        return edges[0]

    @classmethod
    async def get_between_nodes(
        cls, driver: GraphDriver, source_node_uuid: str, target_node_uuid: str
    ):
        match_query = """
            MATCH (n:Entity {uuid: $source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
        """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            RETURN
            """
            + """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        properties(e) AS attributes""",
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        return edges

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        if len(uuids) == 0:
            return []

        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            WHERE e.uuid IN $uuids
            RETURN
            """
            + """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        properties(e) AS attributes""",
            uuids=uuids,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        return edges

    @classmethod
    async def get_by_group_ids(
        cls,
        driver: GraphDriver,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
        with_embeddings: bool = False,
    ):
        cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''
        with_embeddings_query: LiteralString = (
            """,
                e.fact_embedding AS fact_embedding
                """
            if with_embeddings
            else ''
        )

        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            WHERE e.group_id IN $group_ids
            """
            + cursor_query
            + """
            RETURN
            """
            + """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        properties(e) AS attributes"""
            + with_embeddings_query
            + """
            ORDER BY e.uuid DESC
            """
            + limit_query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        if len(edges) == 0:
            raise Exception(f"No edges found for group IDs {group_ids}")
        return edges

    @classmethod
    async def get_by_node_uuid(cls, driver: GraphDriver, node_uuid: str):
        match_query = """
            MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(m:Entity)
        """

        records, _, _ = await driver.execute_query(
            match_query
            + """
            RETURN
            """
            + """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        properties(e) AS attributes""",
            node_uuid=node_uuid,
            routing_='r',
        )

        edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

        return edges


# class CommunityEdge(Edge):
#     async def save(self, driver: GraphDriver):
#         result = await driver.execute_query(
#             get_community_edge_save_query(driver.provider),
#             community_uuid=self.source_node_uuid,
#             entity_uuid=self.target_node_uuid,
#             uuid=self.uuid,
#             group_id=self.group_id,
#             created_at=self.created_at,
#         )

#         logger.debug(f'Saved edge to Graph: {self.uuid}')

#         return result

#     @classmethod
#     async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
#         records, _, _ = await driver.execute_query(
#             """
#             MATCH (n:Community)-[e:HAS_MEMBER {uuid: $uuid}]->(m)
#             RETURN
#             """
#             + COMMUNITY_EDGE_RETURN,
#             uuid=uuid,
#             routing_='r',
#         )

#         edges = [get_community_edge_from_record(record) for record in records]

#         return edges[0]

#     @classmethod
#     async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
#         records, _, _ = await driver.execute_query(
#             """
#             MATCH (n:Community)-[e:HAS_MEMBER]->(m)
#             WHERE e.uuid IN $uuids
#             RETURN
#             """
#             + COMMUNITY_EDGE_RETURN,
#             uuids=uuids,
#             routing_='r',
#         )

#         edges = [get_community_edge_from_record(record) for record in records]

#         return edges

#     @classmethod
#     async def get_by_group_ids(
#         cls,
#         driver: GraphDriver,
#         group_ids: list[str],
#         limit: int | None = None,
#         uuid_cursor: str | None = None,
#     ):
#         cursor_query: LiteralString = 'AND e.uuid < $uuid' if uuid_cursor else ''
#         limit_query: LiteralString = 'LIMIT $limit' if limit is not None else ''

#         records, _, _ = await driver.execute_query(
#             """
#             MATCH (n:Community)-[e:HAS_MEMBER]->(m)
#             WHERE e.group_id IN $group_ids
#             """
#             + cursor_query
#             + """
#             RETURN
#             """
#             + COMMUNITY_EDGE_RETURN
#             + """
#             ORDER BY e.uuid DESC
#             """
#             + limit_query,
#             group_ids=group_ids,
#             uuid=uuid_cursor,
#             limit=limit,
#             routing_='r',
#         )

#         edges = [get_community_edge_from_record(record) for record in records]

#         return edges


# Edge helpers
def get_episodic_edge_from_record(record: Any) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
    )


def get_entity_edge_from_record(record: Any, provider: GraphProvider) -> EntityEdge:
    episodes = record['episodes']
    if provider == GraphProvider.KUZU:
        attributes = json.loads(record['attributes']) if record['attributes'] else {}
    else:
        attributes = record['attributes']
        attributes.pop('uuid', None)
        attributes.pop('source_node_uuid', None)
        attributes.pop('target_node_uuid', None)
        attributes.pop('fact', None)
        attributes.pop('fact_embedding', None)
        attributes.pop('name', None)
        attributes.pop('group_id', None)
        attributes.pop('episodes', None)
        attributes.pop('created_at', None)
        attributes.pop('expired_at', None)
        attributes.pop('valid_at', None)
        attributes.pop('invalid_at', None)

    edge = EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=episodes,
        created_at=parse_db_date(record['created_at']),  # type: ignore
        expired_at=parse_db_date(record['expired_at']),
        valid_at=parse_db_date(record['valid_at']),
        invalid_at=parse_db_date(record['invalid_at']),
        attributes=attributes,
    )

    return edge


# def get_community_edge_from_record(record: Any):
#     return CommunityEdge(
#         uuid=record['uuid'],
#         group_id=record['group_id'],
#         source_node_uuid=record['source_node_uuid'],
#         target_node_uuid=record['target_node_uuid'],
#         created_at=parse_db_date(record['created_at']),  # type: ignore
#     )


async def create_entity_edge_embeddings(embedder: EmbedderClient, edges: list[EntityEdge]):
    # filter out falsey values from edges
    filtered_edges = [edge for edge in edges if edge.fact]

    if len(filtered_edges) == 0:
        return
    fact_embeddings = await embedder.create_batch([edge.fact for edge in filtered_edges])
    for edge, fact_embedding in zip(filtered_edges, fact_embeddings, strict=True):
        edge.fact_embedding = fact_embedding




##############search related stuff###########################
class SearchResults(BaseModel):
    ### somethings dont work
    edges: list[EntityEdge] = Field(default_factory=list)
    edge_reranker_scores: list[float] = Field(default_factory=list)
    nodes: list[EntityNode] = Field(default_factory=list)
    node_reranker_scores: list[float] = Field(default_factory=list)
    episodes: list[EpisodicNode] = Field(default_factory=list)
    episode_reranker_scores: list[float] = Field(default_factory=list)
    # communities: list[CommunityNode] = Field(default_factory=list)
    community_reranker_scores: list[float] = Field(default_factory=list)
    
    
    @classmethod
    def merge(cls, results_list: list['SearchResults']) -> 'SearchResults':
        """
        Merge multiple SearchResults objects into a single SearchResults object.

        Parameters
        ----------
        results_list : list[SearchResults]
            List of SearchResults objects to merge

        Returns
        -------
        SearchResults
            A single SearchResults object containing all results
        """
        if not results_list:
            return cls()

        merged = cls()
        for result in results_list:
            merged.edges.extend(result.edges)
            merged.edge_reranker_scores.extend(result.edge_reranker_scores)
            merged.nodes.extend(result.nodes)
            merged.node_reranker_scores.extend(result.node_reranker_scores)
            merged.episodes.extend(result.episodes)
            merged.episode_reranker_scores.extend(result.episode_reranker_scores)
            merged.communities.extend(result.communities)
            merged.community_reranker_scores.extend(result.community_reranker_scores)

        return merged


#####################dedupe related stuff##########################
@dataclass
class DedupCandidateIndexes:
    """Precomputed lookup structures that drive entity deduplication heuristics."""

    existing_nodes: list[EntityNode]
    nodes_by_uuid: dict[str, EntityNode]
    normalized_existing: defaultdict[str, list[EntityNode]]
    shingles_by_candidate: dict[str, set[str]]
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]]

@dataclass
class DedupResolutionState:
    """Mutable resolution bookkeeping shared across deterministic and LLM passes."""

    resolved_nodes: list[EntityNode | None]
    uuid_map: dict[str, str]
    unresolved_indices: list[int]
    duplicate_pairs: list[tuple[EntityNode, EntityNode]] = field(default_factory=list)



class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate entity. If no duplicate entities are found, default to -1.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Should be the most complete and descriptive name of the entity. Do not include any JSON formatting in the Entity name such as {}.',
    )
    duplicates: list[int] = Field(
        ...,
        description='idx of all entities that are a duplicate of the entity with the above id.',
    )


class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(..., description='List of resolved nodes')












if __name__ == "__main__":
    # Create an EntityNode with hardcoded values
    def create_random_entity_node():
        # Hardcoded values
        name = "Sample Entity"
        group_id = "group-12345"
        labels = ["Person", "Important"]
        summary = "This is a sample entity for testing purposes"
        attributes = {
            "age": 25,
            "location": "New York",
            "occupation": "Engineer"
        }
        name_embedding = None

        # Create the EntityNode
        entity_node = EntityNode(
            name=name,
            group_id=group_id,
            labels=labels,
            summary=summary,
            attributes=attributes,
            name_embedding=name_embedding
        )

        return entity_node

    # Create and print a random EntityNode
    random_entity = create_random_entity_node()
    embedder = OpenAIEmbedder()
    asyncio.run(random_entity.generate_name_embedding(embedder))
    print("Random EntityNode created:")
    print(f"UUID: {random_entity.uuid}")
    print(f"Name: {random_entity.name}")
    print(f"Group ID: {random_entity.group_id}")
    print(f"Labels: {random_entity.labels}")
    print(f"Created At: {random_entity.created_at}")
    print(f"Summary: {random_entity.summary}")
    print(f"Attributes: {random_entity.attributes}")
    print(f"Name Embedding: {'Present (' + str(len(random_entity.name_embedding)) + ' dimensions)' if random_entity.name_embedding else 'None'}")