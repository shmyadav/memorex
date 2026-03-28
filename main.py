
from graph_driver import Neo4jDriver, GraphDriver
from embedder import OpenAIEmbedder, EmbedderClient
from llm import OpenAIClient, LLMClient
from reranker import OpenAIRerankerClient, CrossEncoderClient
from pydantic import BaseModel, ConfigDict
import os
from dotenv import load_dotenv
from helpers import GraphProvider
load_dotenv()
import asyncio
from datamodels import *
from node_operations import extract_nodes
from node_operations import *
from edge_operations import extract_edges, resolve_edge_pointers, resolve_extracted_edges
from bulk_utils import add_nodes_and_edges_bulk

class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    communities: list
    community_edges: list



class Memorex:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        store_raw_episode_content: bool = True,
        graph_driver: Neo4jDriver | None = None,
        max_coroutines: int | None = None
    ):
        """
        Initialize a Memorex instance.

        This constructor sets up a connection to a graph database and initializes
        the LLM client for natural language processing tasks.

        Parameters
        ----------
        uri : str
            The URI of the Neo4j database.
        user : str
            The username for authenticating with the Neo4j database.
        password : str
            The password for authenticating with the Neo4j database.
        llm_client : LLMClient | None, optional
            An instance of LLMClient for natural language processing tasks.
            If not provided, a default OpenAIClient will be initialized.
        embedder : EmbedderClient | None, optional
            An instance of EmbedderClient for embedding tasks.
            If not provided, a default OpenAIEmbedder will be initialized.
        cross_encoder : CrossEncoderClient | None, optional
            An instance of CrossEncoderClient for reranking tasks.
            If not provided, a default OpenAIRerankerClient will be initialized.
        store_raw_episode_content : bool, optional
            Whether to store the raw content of episodes. Defaults to True.
        graph_driver : GraphDriver | None, optional
            An instance of GraphDriver for database operations.
            If not provided, a default Neo4jDriver will be initialized.
        max_coroutines : int | None, optional
            The maximum number of concurrent operations allowed. Overrides SEMAPHORE_LIMIT set in the environment.
            If not set, the Memorex default is used.
        tracer : Tracer | None, optional
            An OpenTelemetry tracer instance for distributed tracing. If not provided, tracing is disabled (no-op).
        trace_span_prefix : str, optional
            Prefix to prepend to all span names. Defaults to 'memorex'.

        Returns
        -------
        None

        Notes
        -----
        This method establishes a connection to a graph database (Neo4j by default) using the provided
        credentials. It also sets up the LLM client, either using the provided client
        or by creating a default OpenAIClient.

        The default database name is defined during the driver’s construction. If a different database name
        is required, it should be specified in the URI or set separately after
        initialization.

        The OpenAI API key is expected to be set in the environment variables.
        Make sure to set the OPENAI_API_KEY environment variable before initializing
        Memorex if you're using the default OpenAIClient.
        """
        if uri is None or user is None or password is None:
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            user = os.getenv('NEO4J_USER', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'password')

        self.driver = Neo4jDriver(uri, user, password)

        self.store_raw_episode_content = store_raw_episode_content
        self.max_coroutines = max_coroutines
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder()
        if cross_encoder:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = OpenAIRerankerClient()

        # Initialize tracer


        self.clients = MemorexClients(
            driver=self.driver,
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
        )

    async def add_episode(
        self,
        uuid: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str | None = None,
        # update_communities: bool = False,
        entity_types: dict[str, type[BaseModel]] | None = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,):
        
        
        """
        Process an episode and update the graph.

        This method extracts information from the episode, creates nodes and edges,
        and updates the graph database accordingly.

        Parameters
        ----------
        name : str
            The name of the episode.
        episode_body : str
            The content of the episode.
        source_description : str
            A description of the episode's source.
        reference_time : datetime
            The reference time for the episode.
        source : EpisodeType, optional
            The type of the episode. Defaults to EpisodeType.message.
        group_id : str | None
            An id for the graph partition the episode is a part of.
        uuid : str | None
            Optional uuid of the episode.
        update_communities : bool
            Optional. Whether to update communities with new node information
        entity_types : dict[str, BaseModel] | None
            Optional. Dictionary mapping entity type names to their Pydantic model definitions.
        excluded_entity_types : list[str] | None
            Optional. List of entity type names to exclude from the graph. Entities classified
            into these types will not be added to the graph. Can include 'Entity' to exclude
            the default entity type.
        previous_episode_uuids : list[str] | None
            Optional.  list of episode uuids to use as the previous episodes. If this is not provided,
            the most recent episodes by created_at date will be used.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including node extraction, edge extraction,
        deduplication, and database updates. It also handles embedding generation
        and edge invalidation.

        It is recommended to run this method as a background process, such as in a queue.
        It's important that each episode is added sequentially and awaited before adding
        the next one. For web applications, consider using FastAPI's background tasks
        or a dedicated task queue like Celery for this purpose.

        Example using FastAPI background tasks:
            @app.post("/add_episode")
            async def add_episode_endpoint(episode_data: EpisodeData):
                background_tasks.add_task(memorex.add_episode, **episode_data.dict())
                return {"message": "Episode processing started"}
        """
        now = datetime.now()
        start = time()
        EPISODE_WINDOW_LEN =10
        previous_episodes = (
                    await self.retrieve_episodes(
                        reference_time,
                        last_n=EPISODE_WINDOW_LEN,
                        # group_ids=[group_id],
                        source=source,
                    )
                    if previous_episode_uuids is None
                    else await EpisodicNode.get_by_uuids(self.driver, previous_episode_uuids)
                )
        episode = EpisodicNode(name=name,
                        group_id=group_id,#remove if currently not using
                        labels=[],
                        source=source,
                        content=episode_body,
                        source_description=source_description,
                        created_at=now,
                        valid_at=reference_time)

        ### why to use at all
        edge_type_map_default = (
                {('Entity', 'Entity'): list(edge_types.keys())}
                if edge_types is not None
                else {('Entity', 'Entity'): []}
            )
        # Extract and resolve nodes
        extracted_nodes = await extract_nodes(
                self.clients.llm_client, episode, previous_episodes, entity_types, excluded_entity_types
            )
        
        nodes, uuid_map, _ = await resolve_extracted_nodes(
                self.clients,
                extracted_nodes,
                episode,
                previous_episodes,
                entity_types,
            )
            
        # Extract and resolve edges in parallel with attribute extraction
        resolved_edges, invalidated_edges = await self._extract_and_resolve_edges(
            episode,
            extracted_nodes,
            previous_episodes,
            edge_type_map or edge_type_map_default,
            group_id,
            edge_types,
            nodes,
            uuid_map,
        )

        # Extract node attributes
        hydrated_nodes = await extract_attributes_from_nodes(
            self.clients, nodes, episode, previous_episodes, entity_types
        )

        entity_edges = resolved_edges + invalidated_edges

        # Process and save episode data
        episodic_edges, episode = await self._process_episode_data(
            episode, hydrated_nodes, entity_edges, now
        )
        end = time()
        duration = end - start
        print(f"Time taken to add episode: {duration} seconds")
        communities = []
        community_edges = []

        return AddEpisodeResults(
            episode=episode,
            episodic_edges=episodic_edges,
            nodes=hydrated_nodes,
            edges=entity_edges,
            communities=communities,
            community_edges=community_edges,
        )

    async def _process_episode_data(
        self,
        episode: EpisodicNode,
        nodes: list[EntityNode],
        entity_edges: list[EntityEdge],
        now: datetime) -> tuple[list[EpisodicEdge], EpisodicNode]:
        """Process and save episode data to the graph."""
        episodic_edges = [
            EpisodicEdge(
                source_node_uuid=episode.uuid,
                target_node_uuid=node.uuid,
                created_at=now,
                group_id=node.group_id,
            )
            for node in nodes
            ]

        episode.entity_edges = [edge.uuid for edge in entity_edges]

        if not self.store_raw_episode_content:
            episode.content = ''

        await add_nodes_and_edges_bulk(
            self.driver,
            [episode],
            episodic_edges,
            nodes,
            entity_edges,
            self.embedder,
        )

        return episodic_edges, episode



    async def _extract_and_resolve_edges(
        self,
        episode: EpisodicNode,
        extracted_nodes: list[EntityNode],
        previous_episodes: list[EpisodicNode],
        edge_type_map: dict[tuple[str, str], list[str]],
        group_id: str,
        edge_types: dict[str, type[BaseModel]] | None,
        nodes: list[EntityNode],
        uuid_map: dict[str, str],
    ) -> tuple[list[EntityEdge], list[EntityEdge]]:
        """Extract edges from episode and resolve against existing graph."""
        extracted_edges = await extract_edges(
            self.clients,
            episode,
            extracted_nodes,
            previous_episodes,
            edge_type_map,
            group_id,
            edge_types,
        )

        edges = resolve_edge_pointers(extracted_edges, uuid_map)

        resolved_edges, invalidated_edges = await resolve_extracted_edges(
            self.clients,
            edges,
            episode,
            nodes,
            edge_types or {},
            edge_type_map,
        )

        return resolved_edges, invalidated_edges


    async def retrieve_episodes(
            self,
            reference_time: datetime,
            last_n: int,
            # group_ids: list[str] | None = None, #can see no scope of group id for now
            source: EpisodeType | None = None,) -> list[EpisodicNode]:
            
            """
            Retrieve the last n episodic nodes from the graph.

            Args:
                driver (Driver): The Neo4j driver instance.
                reference_time (datetime): The reference time to filter episodes. Only episodes with a valid_at timestamp
                                        less than or equal to this reference_time will be retrieved. This allows for
                                        querying the graph's state at a specific point in time.
                last_n (int, optional): The number of most recent episodes to retrieve, relative to the reference_time.
                group_ids (list[str], optional): The list of group ids to return data from.

            Returns:
                list[EpisodicNode]: A list of EpisodicNode objects representing the retrieved episodes.
            """

            query_params: dict = {}
            query_filter = ''

            # if group_ids and len(group_ids) > 0:
            #     query_filter += '\nAND e.group_id IN $group_ids'
            #     query_params['group_ids'] = group_ids

            # if source is not None:
            #     query_filter += '\nAND e.source = $source'
            #     query_params['source'] = source.name

            query: LiteralString = (
                """
                MATCH (e:Episodic)
                WHERE e.valid_at <= $reference_time                            
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
                ORDER BY e.valid_at DESC
                LIMIT $num_episodes
                """
            )
            # print("*************"*2,query)
            # print("*************"*5)
            # print("*************"*2,query_params)   
            result, _, _ = await self.driver.execute_query(
                query,
                reference_time=reference_time,
                num_episodes=last_n,
                **query_params,
            )

            episodes = [get_episodic_node_from_record(record) for record in result]
            return list(reversed(episodes))  # Return in chronological order


            

        



async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Memorex indices
    # This is required before using other Memorex
    # functionality
    #################################################

    # Initialize Memorex with Neo4j connection
    memorex = Memorex()

    # await memorex.driver.build_indices_and_constraints()
    await memorex.add_episode(
        uuid="123",
        name="Whatsupp",
        episode_body="Dosa is my comfort food and Idli is my favorite food",
        source_description="Test source",
        reference_time=datetime.now(),
        source=EpisodeType.message,
        group_id="test_group",
    )



if __name__ == '__main__':
    asyncio.run(main())