from datamodels import *
import helpers
from helpers import semaphore_gather
from llm import LLMClient
import prompt_library
from helpers import *
# from helpers import _normalize_string_exact, _normalize_name_for_fuzzy, _has_high_entropy
from search_config_recipies import *
from search import *


MAX_SUMMARY_CHARS = 500

NodeSummaryFilter = Callable[[EntityNode], Awaitable[bool]]

async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
    group_id: str | None = None,
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.reflexion(context),
        MissedEntities,
        group_id=group_id,
        prompt_name='extract_nodes.reflexion',
    )
    missed_entities = llm_response.get('missed_entities', [])

    return missed_entities



async def extract_nodes(
    llm_client,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,) -> list[EntityNode]:

    start = time()
    entities_missed = True
    reflexion_iterations = 0
    MAX_REFLEXION_ITERATIONS = 2
    custom_prompt = ""
    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
        }
    ]
    ### play more with it, how to add custom entity types
    entity_types_context += (
        [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]
        if entity_types is not None
        else []
    )

    #just provide content of previous episodes
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
        'entity_types': entity_types_context,
        'source_description': episode.source_description,
    }


    while entities_missed and reflexion_iterations < MAX_REFLEXION_ITERATIONS:
        if episode.source == EpisodeType.message:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_message(context),
                response_model=ExtractedEntities,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_message',
            )
        elif episode.source == EpisodeType.text:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_text(context),
                response_model=ExtractedEntities,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_text',
            )
        elif episode.source == EpisodeType.json:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_json(context),
                response_model=ExtractedEntities,
                group_id=episode.group_id,
                prompt_name='extract_nodes.extract_json',
            )

        response_object = ExtractedEntities(**llm_response)

        extracted_entities: list[ExtractedEntity] = response_object.extracted_entities

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            missing_entities = await extract_nodes_reflexion(
                llm_client,
                episode,
                previous_episodes,
                [entity.name for entity in extracted_entities],
                episode.group_id,
            )

            entities_missed = len(missing_entities) != 0

            custom_prompt = 'Make sure that the following entities are extracted: '
            for entity in missing_entities:
                custom_prompt += f'\n{entity},'

    filtered_extracted_entities = [entity for entity in extracted_entities if entity.name.strip()]
    end = time()
    print(f'Extracted new nodes: {filtered_extracted_entities} in {(end - start) * 1000} ms')
    
    
    # Convert the extracted data into EntityNode objects
    extracted_nodes = []
    for extracted_entity in filtered_extracted_entities:
        type_id = extracted_entity.entity_type_id
        if 0 <= type_id < len(entity_types_context):
            entity_type_name = entity_types_context[extracted_entity.entity_type_id].get(
                'entity_type_name'
            )
        else:
            entity_type_name = 'Entity'

        # Check if this entity type should be excluded
        if excluded_entity_types and entity_type_name in excluded_entity_types:
            print(f'Excluding entity "{extracted_entity.name}" of type "{entity_type_name}"')
            continue

        labels: list[str] = list({'Entity', str(entity_type_name)})

        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)
        print(f'Created new node: {new_node.name} (UUID: {new_node.uuid})')

    print(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

    return extracted_nodes



def _build_candidate_indexes(existing_nodes: list[EntityNode]) -> DedupCandidateIndexes:
    """
    Precompute exact and fuzzy lookup structures for efficient entity deduplication.

    This function processes a list of existing entity nodes to create optimized data structures
    that enable fast exact and fuzzy matching during the deduplication process. It performs
    several key preprocessing steps:

    1. **Exact Matching Preparation**: Normalizes entity names for case-insensitive exact matching
    2. **Fuzzy Matching Preparation**: Creates MinHash signatures and Locality-Sensitive Hashing (LSH) buckets
       for approximate string matching using n-gram shingles
    3. **Index Construction**: Builds lookup tables for O(1) access by UUID and efficient candidate retrieval

    The function uses a combination of:
    - Text normalization (lowercasing, whitespace collapsing)
    - 3-gram shingling for fuzzy matching
    - MinHash signatures for dimensionality reduction
    - LSH banding for efficient similarity search

    Args:
        existing_nodes: List of EntityNode objects to index for deduplication.
                       Each node should have 'name' and 'uuid' attributes.

    Returns:
        DedupCandidateIndexes: A comprehensive index structure containing:
            - existing_nodes: Original list of nodes
            - nodes_by_uuid: Dict mapping UUIDs to node objects for O(1) lookup
            - normalized_existing: Dict grouping nodes by normalized exact names
            - shingles_by_candidate: Dict mapping UUIDs to their 3-gram shingle sets
            - lsh_buckets: LSH buckets for efficient fuzzy candidate retrieval
    """
    normalized_existing: defaultdict[str, list[EntityNode]] = defaultdict(list)
    nodes_by_uuid: dict[str, EntityNode] = {}
    shingles_by_candidate: dict[str, set[str]] = {}
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for candidate in existing_nodes:
        normalized = _normalize_string_exact(candidate.name)
        normalized_existing[normalized].append(candidate)
        nodes_by_uuid[candidate.uuid] = candidate
        
        """
        Create 3-gram shingles from the normalized name for MinHash calculations.
        ex: 3 char shingles for "Shubham Yadav"-"shubhamyadav" are 
        "shu", "hub", "ubh", "bha", "ham", "amy", "mya", "yad", "ada", "dav" 
        """
        shingles = _cached_shingles(_normalize_name_for_fuzzy(candidate.name))
        """store shingles for each candidate"""
        shingles_by_candidate[candidate.uuid] = shingles
        
        """compute MinHash signature for the shingles"""
        signature = _minhash_signature(shingles)
        
        """
        Split the MinHash signature into fixed-size bands for locality-sensitive hashing.
        store the signature in the LSH buckets
        """
        for band_index, band in enumerate(_lsh_bands(signature)):
            lsh_buckets[(band_index, band)].append(candidate.uuid)
    """return the candidate indexes"""
    return DedupCandidateIndexes(
        existing_nodes=existing_nodes,
        nodes_by_uuid=nodes_by_uuid,
        normalized_existing=normalized_existing,
        shingles_by_candidate=shingles_by_candidate,
        lsh_buckets=lsh_buckets,
    )

def _resolve_with_similarity(
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
    _FUZZY_JACCARD_THRESHOLD = 0.9
) -> None:
    """
    Resolve extracted entities against existing entities using exact-name
    normalization, MinHash-based fuzzy similarity, and lookup indexes.

    This function performs the *deterministic* stage of entity deduplication.
    For each extracted node, it attempts to map it to an existing node by:

    1. **Exact-name resolution**
       - Normalize the name (lowercase + whitespace collapse).
       - If exactly one existing entity has the same normalized name,
         resolve immediately and record duplicate pairs if needed.
       - If multiple exact matches exist, mark the node as unresolved
         (to be handled in the LLM fallback pass).

    2. **Entropy filtering**
       - Skip fuzzy matching for names that are too short or too
         low-entropy to be reliable for fuzzy similarity comparisons.

    3. **MinHash + LSH fuzzy resolution**
       - Produce a fuzzy-normalized form preserving alphanumerics for
         n-gram shingles.
       - Compute shingles and a MinHash signature.
       - Use LSH bands to retrieve candidate existing entities that
         share similar MinHash buckets.
       - Compute Jaccard similarity between shingles of the extracted node
         and each candidate.
       - If the best candidate exceeds the fuzzy similarity threshold,
         resolve to that entity and record duplicates if applicable.

    4. **Fallback handling**
       - If neither exact nor fuzzy resolution succeeds, add the index
         to `state.unresolved_indices` for later LLM-based resolution.

    Updates:
        - `state.resolved_nodes[idx]` is set when a deterministic match is found.
        - `state.uuid_map` tracks the chosen canonical UUID for each node.
        - `state.duplicate_pairs` collects all pairs identified as duplicates.
        - `state.unresolved_indices` accumulates indices requiring LLM resolution.

    Parameters:
        extracted_nodes (list[EntityNode]):
            Newly extracted entities that need to be deduplicated.
        indexes (DedupCandidateIndexes):
            Precomputed lookup tables (exact-name map, shingles, MinHash LSH buckets).
        state (DedupResolutionState):
            Mutable structures tracking resolutions, unresolved items,
            and detected duplicates.

    Returns:
        None. All results are stored in the provided `state`.
    """




    """Attempt deterministic resolution using exact name hits and fuzzy MinHash comparisons."""
    for idx, node in enumerate(extracted_nodes):
        """Lowercase text and collapse whitespace so equal names map to the same key."""
        normalized_exact = _normalize_string_exact(node.name)
        """Lowercase text and collapse whitespace so equal names map to the same key.
    Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles."""
        normalized_fuzzy = _normalize_name_for_fuzzy(node.name)

        """Filter out very short or low-entropy names that are unreliable for fuzzy matching."""
        if not _has_high_entropy(normalized_fuzzy):
            """Fuzzy matching works poorly on:
            - very short strings
            - repetitive or low-information strings
            - single-letter or meaningless tokens
            SO THIS FUNCTION FILTERS THOSE OUT.
            """
            state.unresolved_indices.append(idx)
            continue

        """Get all existing nodes that match the exact name."""
        existing_matches = indexes.normalized_existing.get(normalized_exact, [])
        """If there is only one exact match, we can resolve the node to that."""
        if len(existing_matches) == 1:
            match = existing_matches[0]
            state.resolved_nodes[idx] = match
            state.uuid_map[node.uuid] = match.uuid
            """If the match is not the same as the node, we add it to the duplicate pairs."""
            if match.uuid != node.uuid:
                state.duplicate_pairs.append((node, match))
            continue
        """If there are multiple exact matches, we need to use fuzzy matching to resolve the node."""
        if len(existing_matches) > 1:
            """We add the node to the unresolved indices to be resolved by the LLM."""
            state.unresolved_indices.append(idx)
            continue

        """Get the shingles for the fuzzy name."""
        shingles = _cached_shingles(normalized_fuzzy)
        """Compute the MinHash signature for the shingles."""
        signature = _minhash_signature(shingles)
        """Get all existing nodes that match the fuzzy name."""
        candidate_ids: set[str] = set()
        """Split the MinHash signature into fixed-size bands for locality-sensitive hashing."""
        for band_index, band in enumerate(_lsh_bands(signature)):
            candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

        best_candidate: EntityNode | None = None
        best_score = 0.0
        for candidate_id in candidate_ids:
            candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
            score = _jaccard_similarity(shingles, candidate_shingles)
            if score > best_score:
                best_score = score
                best_candidate = indexes.nodes_by_uuid.get(candidate_id)

        if best_candidate is not None and best_score >= _FUZZY_JACCARD_THRESHOLD:
            state.resolved_nodes[idx] = best_candidate
            state.uuid_map[node.uuid] = best_candidate.uuid
            if best_candidate.uuid != node.uuid:
                state.duplicate_pairs.append((node, best_candidate))
            continue

        state.unresolved_indices.append(idx)

def to_prompt_json(data: Any, ensure_ascii: bool = False, indent: int | None = None) -> str:
    """
    Serialize data to JSON for use in prompts.

    Args:
        data: The data to serialize
        ensure_ascii: If True, escape non-ASCII characters. If False (default), preserve them.
        indent: Number of spaces for indentation. Defaults to None (minified).

    Returns:
        JSON string representation of the data

    Notes:
        By default (ensure_ascii=False), non-ASCII characters (e.g., Korean, Japanese, Chinese)
        are preserved in their original form in the prompt, making them readable
        in LLM logs and improving model understanding.
    """
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)


async def _resolve_with_llm(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_types: dict[str, type[BaseModel]] | None,
) -> None:
    """Escalate unresolved nodes to the dedupe prompt so the LLM can select or reject duplicates."""
    if not state.unresolved_indices:
        return

    entity_types_dict: dict[str, type[BaseModel]] = entity_types if entity_types is not None else {}
    llm_extracted_nodes = [extracted_nodes[i] for i in state.unresolved_indices]

    extracted_nodes_context = [
        {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': entity_types_dict.get(
                next((item for item in node.labels if item != 'Entity'), '')
            ).__doc__ or 'Default Entity Type',
        }
        for i, node in enumerate(llm_extracted_nodes)
    ]

    existing_nodes_context = [
        {
            **{'idx': i, 'name': candidate.name, 'entity_types': candidate.labels},
            **candidate.attributes,
        }
        for i, candidate in enumerate(indexes.existing_nodes)
    ]

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

    llm_response = await llm_client.generate_response(
        prompt_library.nodes(context),
        response_model=NodeResolutions,
        prompt_name='dedupe_nodes.nodes',
    )

    node_resolutions: list[NodeDuplicate] = NodeResolutions(**llm_response).entity_resolutions
    valid_relative_range = range(len(state.unresolved_indices))
    processed_relative_ids: set[int] = set()

    for resolution in node_resolutions:
        relative_id: int = resolution.id
        duplicate_idx: int = resolution.duplicate_idx

        if relative_id not in valid_relative_range or relative_id in processed_relative_ids:
            continue
        processed_relative_ids.add(relative_id)

        original_index = state.unresolved_indices[relative_id]
        extracted_node = extracted_nodes[original_index]

        if duplicate_idx == -1:
            resolved_node = extracted_node
        elif 0 <= duplicate_idx < len(indexes.existing_nodes):
            resolved_node = indexes.existing_nodes[duplicate_idx]
        else:
            resolved_node = extracted_node

        state.resolved_nodes[original_index] = resolved_node
        state.uuid_map[extracted_node.uuid] = resolved_node.uuid
        if resolved_node.uuid != extracted_node.uuid:
            state.duplicate_pairs.append((extracted_node, resolved_node))


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    """Filter out duplicate node pairs that already have an IS_DUPLICATE_OF edge in the graph.
    
    Queries the graph database to find which duplicate pairs already have the
    IS_DUPLICATE_OF relationship, and returns only those pairs that don't have
    this edge yet (i.e., new duplicates that need to be created).
    
    Args:
        driver: GraphDriver instance for executing database queries.
        duplicates_node_tuples: List of (source, target) node tuples representing
            potential duplicate relationships.
    
    Returns:
        List of duplicate node pairs that don't already have an IS_DUPLICATE_OF edge.
        Returns empty list if input is empty.
    """
    print("*************",duplicates_node_tuples)
    if not duplicates_node_tuples:
        return []

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    query: LiteralString = """
        UNWIND $duplicate_node_uuids AS duplicate_tuple
        MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
        RETURN DISTINCT
            n.uuid AS source_uuid,
            m.uuid AS target_uuid
    """
    duplicate_node_uuids = list(duplicate_nodes_map.keys())

    records, _, _ = await driver.execute_query(
        query,
        duplicate_node_uuids=duplicate_node_uuids,
        routing_='r',
    )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())



async def resolve_extracted_nodes(
    clients: GraphitiClients,#needs both llm and driver client
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    existing_nodes_override: list[EntityNode] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Search for existing nodes, resolve deterministic matches, then escalate holdouts to the LLM dedupe prompt."""
    llm_client = clients.llm_client
    driver = clients.driver
    existing_nodes = await _collect_candidate_nodes(
        clients,
        extracted_nodes,
        existing_nodes_override,
    )

    indexes: DedupCandidateIndexes = _build_candidate_indexes(existing_nodes)

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=[],
    )
    ###check if something happens
    _resolve_with_similarity(extracted_nodes, indexes, state)

    await _resolve_with_llm(
        llm_client,
        extracted_nodes,
        indexes,
        state,
        episode,
        previous_episodes,
        entity_types,
    )

    for idx, node in enumerate(extracted_nodes):
        if state.resolved_nodes[idx] is None:
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    logger.debug(
        'Resolved nodes: %s',
        [(node.name, node.uuid) for node in state.resolved_nodes if node is not None],
    )
    print("*************"*2,state.duplicate_pairs)

    new_node_duplicates: list[
        tuple[EntityNode, EntityNode]
    ] = await filter_existing_duplicate_of_edges(driver, state.duplicate_pairs)

    return (
        [node for node in state.resolved_nodes if node is not None],
        state.uuid_map,
        new_node_duplicates,
    )

async def _collect_candidate_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Search per extracted name and return unique candidates with overrides honored in order."""
    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,
                group_ids=[node.group_id],
                search_filter=SearchFilters(),
                config=NODE_HYBRID_SEARCH_RRF,
            )
            for node in extracted_nodes
        ]
    )

    candidate_nodes: list[EntityNode] = [node for result in search_results for node in result.nodes]

    if existing_nodes_override is not None:
        candidate_nodes.extend(existing_nodes_override)

    seen_candidate_uuids: set[str] = set()
    ordered_candidates: list[EntityNode] = []
    for candidate in candidate_nodes:
        if candidate.uuid in seen_candidate_uuids:
            continue
        seen_candidate_uuids.add(candidate.uuid)
        ordered_candidates.append(candidate)

    return ordered_candidates


async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
) -> list[EntityNode]:
    llm_client = clients.llm_client
    embedder = clients.embedder
    updated_nodes: list[EntityNode] = await semaphore_gather(
        *[
            extract_attributes_from_node(
                llm_client,
                node,
                episode,
                previous_episodes,
                (
                    entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                    if entity_types is not None
                    else None
                ),
                should_summarize_node,
            )
            for node in nodes
        ]
    )

    await create_entity_node_embeddings(embedder, updated_nodes)

    return updated_nodes


async def extract_attributes_from_node(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_type: type[BaseModel] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
) -> EntityNode:
    # Extract attributes if entity type is defined and has attributes
    llm_response = await _extract_entity_attributes(
        llm_client, node, episode, previous_episodes, entity_type
    )

    # Extract summary if needed
    await _extract_entity_summary(
        llm_client, node, episode, previous_episodes, should_summarize_node
    )

    node.attributes.update(llm_response)

    return node


def _build_episode_context(
    node_data: dict[str, Any],
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
) -> dict[str, Any]:
    return {
        'node': node_data,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [ep.content for ep in previous_episodes] if previous_episodes is not None else []
        ),
    }

async def _extract_entity_attributes(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_type: type[BaseModel] | None,
) -> dict[str, Any]:
    if entity_type is None or len(entity_type.model_fields) == 0:
        return {}

    attributes_context = _build_episode_context(
        # should not include summary
        node_data={
            'name': node.name,
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    llm_response = await llm_client.generate_response(
        prompt_library.extract_attributes(attributes_context),
        response_model=entity_type,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_attributes',
    )
    # validate response
    entity_type(**llm_response)

    return llm_response


async def _extract_entity_summary(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
) -> None:
    if should_summarize_node is not None and not await should_summarize_node(node):
        return

    summary_context = _build_episode_context(
        node_data={
            'name': node.name,
            'summary': truncate_at_sentence(node.summary, MAX_SUMMARY_CHARS),
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    summary_response = await llm_client.generate_response(
        prompt_library.extract_summary(summary_context),
        response_model=EntitySummary,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_summary',
    )

    node.summary = truncate_at_sentence(summary_response.get('summary', ''), MAX_SUMMARY_CHARS)