from abc import ABC, abstractmethod
from enum import Enum
import openai
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletionMessageParam
# ChatCompletionMessageParam :type definition (TypedDict / union type) provided by the OpenAI Python SDK to describe what a 
# single chat message is allowed to look like when you send it to a chat/completions-style API.
from pydantic import BaseModel
from typing import *
from typing import Any
import typing
import json
import logging
logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 1
DEFAULT_MODEL = 'gpt-5-mini'
DEFAULT_SMALL_MODEL = 'gpt-5-nano'
DEFAULT_REASONING = 'minimal'
DEFAULT_VERBOSITY = 'low'

class Message(BaseModel):
    role: str
    content: str

class ModelSize(Enum):
    small = 'small'
    medium = 'medium'


class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
                api_key (str): The authentication key for accessing the LLM API.
                                                This is required for making authorized requests.

                model (str, optional): The specific LLM model to use for generating responses.
                                                                Defaults to "gpt-4.1-mini".

                base_url (str, optional): The base URL of the LLM API service.
                                                                        Defaults to "https://api.openai.com", which is OpenAI's standard API endpoint.
                                                                        This can be changed if using a different provider or a custom endpoint.

                small_model (str, optional): The specific LLM model to use for generating responses of simpler prompts.
                                                                Defaults to "gpt-4.1-nano".
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.small_model = small_model
        self.temperature = temperature
        self.max_tokens = max_tokens


class LLMClient(ABC):
    def __init__(self, config: LLMConfig | None, cache: bool = False):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.model = config.model
        self.small_model = config.small_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.cache_enabled = cache
        self.cache_dir = None

    def _clean_input(self, input: str) -> str:
        """Clean input string of invalid unicode and control characters.
        Args:
            input: Raw input string to be cleaned
        Returns:
            Cleaned string safe for LLM processing
        """
        # Clean any invalid Unicode
        cleaned = input.encode('utf-8', errors='ignore').decode('utf-8')

        # Remove zero-width characters and other invisible unicode
        zero_width = '\u200b\u200c\u200d\ufeff\u2060'
        for char in zero_width:
            cleaned = cleaned.replace(char, '')

        # Remove control characters except newlines, returns, and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')

        return cleaned


    def _get_provider_type(self) -> str:
        """Get provider type from class name."""
        class_name = self.__class__.__name__.lower()
        if 'openai' in class_name:
            return 'openai'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        else:
            return 'unknown'
    

class BaseOpenAIClient(LLMClient):
    """
    Base client class for OpenAI-compatible APIs (OpenAI and Azure OpenAI).

    This class contains shared logic for both OpenAI and Azure OpenAI clients,
    reducing code duplication while allowing for implementation-specific differences.
    """

    # Class-level constants
    MAX_RETRIES = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str | None = DEFAULT_REASONING,
        verbosity: str | None = DEFAULT_VERBOSITY,
    ):
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI-based clients')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.verbosity = verbosity

    @abstractmethod
    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Create a completion using the specific client implementation."""
        pass

    @abstractmethod
    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None,
        verbosity: str | None,
    ) -> Any:
        """Create a structured completion using the specific client implementation."""
        pass


class OpenAIClient(BaseOpenAIClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the BaseOpenAIClient and provides OpenAI-specific implementation
    for creating completions.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a structured completion using OpenAI's beta parse API."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )

        response = await self.client.responses.parse(
            model=model,
            input=messages,  # type: ignore
            temperature=temperature if not is_reasoning_model else None,
            max_output_tokens=max_tokens,
            text_format=response_model,  # type: ignore
            reasoning={'effort': reasoning} if reasoning is not None else None,  # type: ignore
            text={'verbosity': verbosity} if verbosity is not None else None,  # type: ignore
        )

        return response
    
    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate model name based on the requested size."""
        if model_size == ModelSize.small:
            return self.small_model or DEFAULT_SMALL_MODEL
        else:
            return self.model or DEFAULT_MODEL


    def _convert_messages_to_openai_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal Message format to OpenAI ChatCompletionMessageParam format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        return openai_messages

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion with JSON format."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = (
            model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')
        )

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
        )
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Generate a response using the appropriate client implementation."""
        openai_messages = self._convert_messages_to_openai_format(messages)
        model = self._get_model_for_size(model_size)

        try:
            if response_model:
                response = await self._create_structured_completion(
                    model=model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    response_model=response_model,
                    reasoning=self.reasoning,
                    verbosity=self.verbosity,
                )
                return self._handle_structured_response(response)
            else:
                response = await self._create_completion(
                    model=model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                return self._handle_json_response(response)

        except Exception as e:
            error_msg = str(e)
            if isinstance(e, openai.LengthFinishReasonError):
                raise Exception(
                    f'Output length exceeded max tokens {self.max_tokens}: {e}'
                ) from e
            if isinstance(e, openai.AuthenticationError):
                logger.error(
                    f'OpenAI Authentication Error: {e}. Please verify your API key is correct.'
                )
                raise Exception(f'LLM authentication failed: {e}') from e
            if 'connection error' in error_msg.lower() or 'connection' in error_msg.lower():
                logger.error(
                    'Connection error communicating with OpenAI API. '
                    f'Please check your network connection and API key. Error: {e}'
                )
                raise Exception(f'LLM connection error: {e}') from e
            logger.error(f'Error in generating LLM response: {e}')
            raise Exception(f'LLM request failed: {e}') from e

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,) -> dict[str, typing.Any]:
        
        """Generate a response with retry logic and error handling."""
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Add multilingual extraction instructions
        messages[0].content += '\n\nAny extracted information should be returned in the same language as it was written in.'

        retry_count = 0
        last_error = None

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                return response
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise Exception(
                        f'LLM request failed after {self.MAX_RETRIES} retries: {e}'
                    ) from e

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'The previous response attempt was invalid. '
                    f'Error type: {e.__class__.__name__}. '
                    f'Error details: {str(e)}. '
                    f'Please try again with a valid response, ensuring the output matches '
                    f'the expected format and constraints.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                )

        # If we somehow get here, raise the last error
        raise Exception(
            f'LLM request failed after {self.MAX_RETRIES} retries: {last_error}'
        ) from last_error
    
    def _handle_structured_response(self, response: Any) -> dict[str, Any]:
        """Handle structured response parsing and validation."""
        response_object = response.output_text

        if response_object:
            return json.loads(response_object)
        elif response_object.refusal:
            raise Exception(f'LLM refusal: {response_object.refusal}')
        else:
            raise Exception(f'Invalid response from LLM: {response_object.model_dump()}')