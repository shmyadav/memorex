import logging
from collections.abc import Coroutine
from typing import Any

from neo4j import AsyncGraphDatabase, EagerResult
from typing_extensions import LiteralString

logger = logging.getLogger(__name__)
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from enum import Enum
from typing import Any
from dotenv import load_dotenv
import asyncio


class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'



SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))
async def semaphore_gather(
    *coroutines: Coroutine,
    max_coroutines: int | None = None,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max_coroutines or SEMAPHORE_LIMIT)

    async def _wrap_coroutine(coroutine):
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(_wrap_coroutine(coroutine) for coroutine in coroutines))




logger = logging.getLogger(__name__)

DEFAULT_SIZE = 10

load_dotenv()


class GraphDriverSession(ABC):
    provider: GraphProvider

    async def __aenter__(self):
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    @abstractmethod
    async def run(self, query: str, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def close(self):
        raise NotImplementedError()

    @abstractmethod
    async def execute_write(self, func, *args, **kwargs):
        raise NotImplementedError()


class GraphDriver(ABC):
    provider: GraphProvider
    fulltext_syntax: str = (
        ''  # Neo4j (default) syntax does not require a prefix for fulltext queries
    )
    _database: str
    default_group_id: str = ''
    search_interface: None = None
    graph_operations_interface:  None = None

    @abstractmethod
    def execute_query(self, cypher_query_: str, **kwargs: Any) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    def session(self, database: str | None = None) -> GraphDriverSession:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def delete_all_indexes(self) -> Coroutine:
        raise NotImplementedError()

    @abstractmethod
    async def build_indices_and_constraints(self, delete_existing: bool = False):
        raise NotImplementedError()




class Neo4jDriver(GraphDriver):
    provider = GraphProvider.NEO4J
    default_group_id: str = ''

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        super().__init__()
        self.uri = uri or os.getenv('NEO4J_URI')
        if not self.uri:
            raise ValueError("NEO4J_URI must be provided")
        self.user = user or os.getenv('NEO4J_USER')
        self.password = password or os.getenv('NEO4J_PASSWORD')
        self.database = database or os.getenv('NEO4J_DATABASE') or 'neo4j'
        self.client = AsyncGraphDatabase.driver(
            uri=self.uri,
            auth=(self.user or '', self.password or ''),
        )
        self._database = self.database

        self.aoss_client = None

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        # Check if database_ is provided in kwargs.
        # If not populated, set the value to retain backwards compatibility
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        params.setdefault('database_', self._database)

        try:
            result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
        except Exception as e:
            logger.error(f'Error executing Neo4j query: {e}\n{cypher_query_}\n{params}')
            raise

        return result

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return self.client.session(database=_database)  # type: ignore

    async def close(self) -> None:
        return await self.client.close()

    def delete_all_indexes(self) -> Coroutine:
        return self.client.execute_query(
            'CALL db.indexes() YIELD name DROP INDEX name',
        )

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()

        range_indices: list[LiteralString] = [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX community_group_id IF NOT EXISTS FOR (n:Community) ON (n.group_id)',
        'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
        'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
        'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
        'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
        'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
        'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
        'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
        'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
        'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
        'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
        'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
    ]

        fulltext_indices: list[LiteralString] = [
        """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS
        FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS
        FOR (n:Community) ON EACH [n.name, n.group_id]""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
        FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
    ]

        index_queries: list[LiteralString] = range_indices + fulltext_indices

        await semaphore_gather(
            *[
                self.execute_query(
                    query,
                )
                for query in index_queries
            ]
        )

