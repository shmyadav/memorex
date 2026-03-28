

import asyncio
import hashlib
import math
import os
import re
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, ClassVar
import numpy as np
from dotenv import load_dotenv
from neo4j import time as neo4j_time
from numpy._typing import NDArray
from pydantic import BaseModel
from enum import Enum
# from datamodels import *
from functools import lru_cache
from collections.abc import Iterable
from typing_extensions import Buffer, Self, final
from embedder import EmbedderClient

_NAME_ENTROPY_THRESHOLD = 1.5
_MIN_NAME_LENGTH = 6
_MIN_TOKEN_COUNT = 2

class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'


@final
class blake2b:
    """### i dont know wtf is this"""
    """Return a new BLAKE2b hash object."""
    MAX_DIGEST_SIZE: ClassVar[int] = 64
    MAX_KEY_SIZE: ClassVar[int] = 64
    PERSON_SIZE: ClassVar[int] = 16
    SALT_SIZE: ClassVar[int] = 16
    block_size: int
    digest_size: int
    name: str
    def __new__(
        cls,
        data: Buffer = b"",
        /,
        *,
        digest_size: int = 64,
        key: Buffer = b"",
        salt: Buffer = b"",
        person: Buffer = b"",
        fanout: int = 1,
        depth: int = 1,
        leaf_size: int = 0,
        node_offset: int = 0,
        node_depth: int = 0,
        inner_size: int = 0,
        last_node: bool = False,
        usedforsecurity: bool = True,
    ) -> Self: ...
    def copy(self) -> Self:
        """Return a copy of the hash object."""
        ...
    def digest(self) -> bytes:
        """Return the digest value as a bytes object."""
        ...
    def hexdigest(self) -> str:
        """Return the digest value as a string of hexadecimal digits."""
        ...
    def update(self, data: Buffer, /) -> None:
        """Update this hash object's state with the provided bytes-like object."""
        ...


load_dotenv()

USE_PARALLEL_RUNTIME = bool(os.getenv('USE_PARALLEL_RUNTIME', False))
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))
MAX_REFLEXION_ITERATIONS = int(os.getenv('MAX_REFLEXION_ITERATIONS', 0))
DEFAULT_PAGE_LIMIT = 20


def parse_db_date(input_date: neo4j_time.DateTime | str | None) -> datetime | None:
    if isinstance(input_date, neo4j_time.DateTime):
        return input_date.to_native()

    if isinstance(input_date, str):
        return datetime.fromisoformat(input_date)

    return input_date


def get_default_group_id(provider: GraphProvider) -> str:
    """
    This function differentiates the default group id based on the database type.
    For most databases, the default group id is an empty string, while there are database types that require a specific default group id.
    """
    if provider == GraphProvider.FALKORDB:
        return '\\_'
    else:
        return ''


def lucene_sanitize(query: str) -> str:
    # Escape special characters from a query before passing into Lucene
    # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
    escape_map = str.maketrans(
        {
            '+': r'\+',
            '-': r'\-',
            '&': r'\&',
            '|': r'\|',
            '!': r'\!',
            '(': r'\(',
            ')': r'\)',
            '{': r'\{',
            '}': r'\}',
            '[': r'\[',
            ']': r'\]',
            '^': r'\^',
            '"': r'\"',
            '~': r'\~',
            '*': r'\*',
            '?': r'\?',
            ':': r'\:',
            '\\': r'\\',
            '/': r'\/',
            'O': r'\O',
            'R': r'\R',
            'N': r'\N',
            'T': r'\T',
            'A': r'\A',
            'D': r'\D',
        }
    )

    sanitized = query.translate(escape_map)
    return sanitized


def normalize_l2(embedding: list[float]) -> NDArray:
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array, 2, axis=0, keepdims=True)
    return np.where(norm == 0, embedding_array, embedding_array / norm)


async def semaphore_gather(
    *coroutines: Coroutine,          # Accept any number of coroutine objects
    max_coroutines: int | None = None # Optional limit for concurrent execution
) -> list[Any]:
    
    # Create a semaphore that allows only N coroutines to run at the same time
    # If max_coroutines is None, fall back to a predefined SEMAPHORE_LIMIT
    semaphore = asyncio.Semaphore(max_coroutines or SEMAPHORE_LIMIT)

    # Wrapper coroutine that enforces the semaphore
    async def _wrap_coroutine(coroutine):
        # Wait until the semaphore allows entry (blocks if limit is reached)
        async with semaphore:
            # Run the original coroutine and return its result
            return await coroutine

    # Wrap every coroutine with semaphore control,
    # run them all via asyncio.gather,
    # and return the list of results (in input order)
    return await asyncio.gather(
        *(_wrap_coroutine(coroutine) for coroutine in coroutines)
    )



def validate_group_id(group_id: str | None) -> bool:
    """
    Validate that a group_id contains only ASCII alphanumeric characters, dashes, and underscores.

    Args:
        group_id: The group_id to validate

    Returns:
        True if valid, False otherwise

    Raises:
        GroupIdValidationError: If group_id contains invalid characters
    """

    # Allow empty string (default case)
    if not group_id:
        return True

    # Check if string contains only ASCII alphanumeric characters, dashes, or underscores
    # Pattern matches: letters (a-z, A-Z), digits (0-9), hyphens (-), and underscores (_)
    if not re.match(r'^[a-zA-Z0-9_-]+$', group_id):
        raise #GroupIdValidationError(group_id)

    return True


def validate_excluded_entity_types(
    excluded_entity_types: list[str] | None, entity_types: dict[str, type[BaseModel]] | None = None
) -> bool:
    """
    Validate that excluded entity types are valid type names.

    Args:
        excluded_entity_types: List of entity type names to exclude
        entity_types: Dictionary of available custom entity types

    Returns:
        True if valid

    Raises:
        ValueError: If any excluded type names are invalid
    """
    if not excluded_entity_types:
        return True

    # Build set of available type names
    available_types = {'Entity'}  # Default type is always available
    if entity_types:
        available_types.update(entity_types.keys())

    # Check for invalid type names
    invalid_types = set(excluded_entity_types) - available_types
    if invalid_types:
        raise ValueError(
            f'Invalid excluded entity types: {sorted(invalid_types)}. Available types: {sorted(available_types)}'
        )

    return True


############deduping################

def _normalize_string_exact(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key."""
    normalized = re.sub(r'[\s]+', ' ', name.lower())
    return normalized.strip()


def _normalize_name_for_fuzzy(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key.
    Keeps only lowercase alphanumerics, apostrophes, and spaces; all other
    characters are replaced with spaces and whitespace is re-collapsed."""
    normalized = re.sub(r"[^a-z0-9' ]", ' ', _normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r'[\s]+', ' ', normalized)

def _lsh_bands(signature: Iterable[int], _MINHASH_BAND_SIZE = 4) -> list[tuple[int, ...]]:
    """Split the MinHash signature into fixed-size bands for locality-sensitive hashing."""
    signature_list = list(signature)
    if not signature_list:
        return []

    bands: list[tuple[int, ...]] = []
    for start in range(0, len(signature_list), _MINHASH_BAND_SIZE):
        band = tuple(signature_list[start : start + _MINHASH_BAND_SIZE])
        if len(band) == _MINHASH_BAND_SIZE:
            bands.append(band)
    return bands

def _shingles(normalized_name: str) -> set[str]:
    """Create 3-gram shingles from the normalized name for MinHash calculations."""
    cleaned = normalized_name.replace(' ', '')
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()

    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}



@lru_cache(maxsize=512)
def _cached_shingles(name: str) -> set[str]:
    """Cache shingle sets per normalized name to avoid recomputation within a worker."""
    return _shingles(name)

def _hash_shingle(shingle: str, seed: int) -> int:
    """Generate a deterministic 64-bit hash for a shingle given the permutation seed."""
    digest = hashlib.blake2b(f'{seed}:{shingle}'.encode(), digest_size=8)
    return int.from_bytes(digest.digest(), 'big')

def _minhash_signature(shingles: Iterable[str],_MINHASH_PERMUTATIONS = 32) -> tuple[int, ...]:
    """Compute the MinHash signature for the shingle set across predefined permutations."""
    if not shingles:
        return tuple[int, ...]()

    seeds = range(_MINHASH_PERMUTATIONS)
    signature: list[int] = []
    for seed in seeds:
        min_hash = min(_hash_shingle(shingle, seed) for shingle in shingles)
        signature.append(min_hash)
    return tuple(signature)


def _name_entropy(normalized_name: str) -> float:
    """Approximate text specificity using Shannon entropy over characters.

    We strip spaces, count how often each character appears, and sum
    probability * -log2(probability). Short or repetitive names yield low
    entropy, which signals we should defer resolution to the LLM instead of
    trusting fuzzy similarity.
    """
    if not normalized_name:
        return 0.0

    counts: dict[str, int] = {}
    for char in normalized_name.replace(' ', ''):
        counts[char] = counts.get(char, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def _has_high_entropy(normalized_name: str) -> bool:
    """Filter out very short or low-entropy names that are unreliable for fuzzy matching."""
    token_count = len(normalized_name.split())
    if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
        return False

    return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD

def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity between two shingle sets, handling empty edge cases."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list):
    # filter out falsey values from nodes
    filtered_nodes = [node for node in nodes if node.name]

    if not filtered_nodes:
        return

    name_embeddings = await embedder.create_batch([node.name for node in filtered_nodes])
    for node, name_embedding in zip(filtered_nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at or about max_chars while respecting sentence boundaries.

    Attempts to truncate at the last complete sentence before max_chars.
    If no sentence boundary is found before max_chars, truncates at max_chars.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text

    # Find all sentence boundaries (., !, ?) up to max_chars
    truncated = text[:max_chars]

    # Look for sentence boundaries: period, exclamation, or question mark followed by space or end
    sentence_pattern = r'[.!?](?:\s|$)'
    matches = list(re.finditer(sentence_pattern, truncated))

    if matches:
        # Truncate at the last sentence boundary found
        last_match = matches[-1]
        return text[: last_match.end()].rstrip()

    # No sentence boundary found, truncate at max_chars
    return truncated.rstrip()

__all__ = [
    '_cached_shingles',
    '_hash_shingle',
    '_has_high_entropy',
    '_jaccard_similarity',
    '_lsh_bands',
    '_minhash_signature',
    '_name_entropy',
    '_normalize_name_for_fuzzy',
    '_normalize_string_exact',
    '_shingles',
    'get_default_group_id',
    'lucene_sanitize',
    'normalize_l2',
    'parse_db_date',
    'semaphore_gather',
    'validate_excluded_entity_types',
    'validate_group_id',
    'create_entity_node_embeddings',
    'truncate_at_sentence'
]

