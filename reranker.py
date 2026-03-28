import logging
from typing import Any
from pydantic import BaseModel
from abc import ABC, abstractmethod
import numpy as np
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from helpers import semaphore_gather
from llm import LLMConfig, OpenAIClient


class RateLimitError(Exception):
    """Exception raised when the rate limit is exceeded."""

    def __init__(self, message='Rate limit exceeded. Please try again later.'):
        self.message = message
        super().__init__(self.message)

class Message(BaseModel):
    role: str
    content: str


DEFAULT_MODEL = 'gpt-4.1-nano'

class CrossEncoderClient(ABC):
    """
    CrossEncoderClient is an abstract base class that defines the interface
    for cross-encoder models used for ranking passages based on their relevance to a query.
    It allows for different implementations of cross-encoder models to be used interchangeably.
    """

    @abstractmethod
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank the given passages based on their relevance to the query.

        Args:
            query (str): The query string.
            passages (list[str]): A list of passages to rank.

        Returns:
            list[tuple[str, float]]: A list of tuples containing the passage and its score,
                                     sorted in descending order of relevance.
        """
        pass



class OpenAIRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | OpenAIClient | None = None,
    ):
        """
        Initialize the OpenAIRerankerClient with the provided configuration and client.

        This reranker uses the OpenAI API to run a simple boolean classifier prompt concurrently
        for each passage. Log-probabilities are used to rank the passages.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            client (AsyncOpenAI | AsyncAzureOpenAI | OpenAIClient | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        elif isinstance(client, OpenAIClient):
            self.client = client.client
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        openai_messages_list: Any = [
            [
                Message(
                    role='system',
                    content='You are an expert tasked with determining whether the passage is relevant to the query',
                ),
                Message(
                    role='user',
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]
        try:
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=self.config.model or DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=0,
                        max_tokens=1,
                        logit_bias={'6432': 1, '7983': 1},
                        logprobs=True,
                        top_logprobs=2,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            responses_top_logprobs = [
                response.choices[0].logprobs.content[0].top_logprobs
                if response.choices[0].logprobs is not None
                and response.choices[0].logprobs.content is not None
                else []
                for response in responses
            ]
            scores: list[float] = []
            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    continue
                norm_logprobs = np.exp(top_logprobs[0].logprob)
                if top_logprobs[0].token.strip().split(' ')[0].lower() == 'true':
                    scores.append(norm_logprobs)
                else:
                    scores.append(1 - norm_logprobs)

            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
            results.sort(reverse=True, key=lambda x: x[1])
            return results
        except openai.RateLimitError as e:
            print("Rate limit error")
            raise
        except Exception as e:
            print("Exception")
            raise
