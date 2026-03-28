from abc import ABC, abstractmethod
from collections.abc import Iterable

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from abc import ABC, abstractmethod
from collections.abc import Iterable
import os
from pydantic import BaseModel, Field

load_dotenv()

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'



class OpenAIEmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = os.getenv('OPENAI_API_KEY')


class EmbedderClient(ABC):
    @abstractmethod
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        pass

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        raise NotImplementedError()


class OpenAIEmbedder(EmbedderClient):
    """
    OpenAI Embedder Client

    This client supports both AsyncOpenAI and AsyncAzureOpenAI clients.
    """
    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        if config is None:
            config = OpenAIEmbedderConfig()
        self.config = config

        if client is not None:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=config.api_key)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]


async def main():
    """Initialize and test the OpenAI embedder."""
    # Create an embedder instance
    embedder = OpenAIEmbedder()

    # Test with a simple string
    test_text = "Hello, world!"
    embedding = await embedder.create(test_text)

    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch embedding
    test_texts = ["Hello", "World", "OpenAI", "Embeddings"]
    batch_embeddings = await embedder.create_batch(test_texts)

    print(f"Batch embeddings count: {len(batch_embeddings)}")
    print(f"Each embedding dimension: {len(batch_embeddings[0])}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
