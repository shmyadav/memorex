from enum import Enum

from pydantic import BaseModel, Field

RELEVANT_SCHEMA_LIMIT = 10
DEFAULT_MIN_SCORE = 0.6
DEFAULT_MMR_LAMBDA = 0.5
MAX_SEARCH_DEPTH = 3
MAX_QUERY_LENGTH = 128
DEFAULT_SEARCH_LIMIT = 10

class EdgeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'

class NodeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'

class EpisodeSearchMethod(Enum):
    bm25 = 'bm25'

class CommunitySearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'

class EdgeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'

class NodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'

class EpisodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    cross_encoder = 'cross_encoder'

class CommunityReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'

class EdgeSearchConfig(BaseModel):
    search_methods: list[EdgeSearchMethod]
    reranker: EdgeReranker = Field(default=EdgeReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)

class NodeSearchConfig(BaseModel):
    search_methods: list[NodeSearchMethod]
    reranker: NodeReranker = Field(default=NodeReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)

class EpisodeSearchConfig(BaseModel):
    search_methods: list[EpisodeSearchMethod]
    reranker: EpisodeReranker = Field(default=EpisodeReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)

class CommunitySearchConfig(BaseModel):
    search_methods: list[CommunitySearchMethod]
    reranker: CommunityReranker = Field(default=CommunityReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)

class SearchConfig(BaseModel):
    #should have just one config from edge, node, episode and community
    edge_config: EdgeSearchConfig | None = Field(default=None)
    node_config: NodeSearchConfig | None = Field(default=None)
    episode_config: EpisodeSearchConfig | None = Field(default=None)
    community_config: CommunitySearchConfig | None = Field(default=None)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT)
    reranker_min_score: float = Field(default=0)

