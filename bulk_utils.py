from datamodels import *
from graph_driver import GraphDriver, GraphDriverSession
from embedder import EmbedderClient
from typing import Any
from search_utils import get_episode_node_save_bulk_query,get_entity_node_save_bulk_query



def get_episodic_edge_save_bulk_query() -> str:
    return """
        UNWIND $episodic_edges AS edge
        MATCH (episode:Episodic {uuid: edge.source_node_uuid})
        MATCH (node:Entity {uuid: edge.target_node_uuid})
        MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
        SET
            e.group_id = edge.group_id,
            e.created_at = edge.created_at
        RETURN e.uuid AS uuid
    """


def get_entity_edge_save_bulk_query(has_aoss: bool = False) -> str:
    save_embedding_query = (
        'WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)'
        if not has_aoss
        else ''
    )
    return (
        """
            UNWIND $entity_edges AS edge
            MATCH (source:Entity {uuid: edge.source_node_uuid})
            MATCH (target:Entity {uuid: edge.target_node_uuid})
            MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
            SET e = edge
            """
        + save_embedding_query
        + """
        RETURN edge.uuid AS uuid
    """
    )

async def add_nodes_and_edges_bulk(
    driver: GraphDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
):
    session = driver.session()
    try:
        await session.execute_write(
            add_nodes_and_edges_bulk_tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver=driver,
        )
    finally:
        await session.close()


async def add_nodes_and_edges_bulk_tx(
    tx: GraphDriverSession,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    driver: GraphDriver):
    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
        episode.pop('labels', None)

    nodes = []

    for node in entity_nodes:
        if node.name_embedding is None:
            await node.generate_name_embedding(embedder)

        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
            'name_embedding': node.name_embedding,
            'labels': list(set(node.labels + ['Entity'])),
        }

        entity_data.update(node.attributes or {})

        nodes.append(entity_data)

    edges = []
    for edge in entity_edges:
        if edge.fact_embedding is None:
            await edge.generate_embedding(embedder)
        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'fact_embedding': edge.fact_embedding,
        }

        edge_data.update(edge.attributes or {})

        edges.append(edge_data)

    else:
        await tx.run(get_episode_node_save_bulk_query(), episodes=episodes)
        await tx.run(
            get_entity_node_save_bulk_query( nodes),
            nodes=nodes,
        )
        await tx.run(
            get_episodic_edge_save_bulk_query(),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )
        await tx.run(
            get_entity_edge_save_bulk_query(driver.provider),
            entity_edges=edges,
        )