import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls instead of using SDK.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, **config):
        """
        Initialize the Ollama provider with the given configuration.
        """
        self.url = config.get("api_url") or os.getenv(
            "OLLAMA_API_URL", "http://localhost:11434"
        )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the chat completions endpoint using httpx.
        """
        # Ensure that tools are correctly formatted for Ollama if present
        # Ollama expects tools at the top level of the request body
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        # Handle tools if present in kwargs
        if "tools" in kwargs:
            payload["tools"] = kwargs.pop("tools")

        # Pass remaining kwargs as options if they are common parameters
        # Ollama often expects these in an "options" dictionary
        options = {}
        for key in ["temperature", "top_p", "seed", "stop", "num_predict", "top_k"]:
            if key in kwargs:
                options[key] = kwargs.pop(key)
        
        if options:
            payload["options"] = options

        # Merge any remaining kwargs directly into the payload
        payload.update(kwargs)

        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                # Capture detailed error from Ollama if possible
                try:
                    error_detail = response.json().get("error", response.text)
                except Exception:
                    error_detail = response.text
                raise LLMError(f"Ollama request failed with status {response.status_code}: {error_detail}")
                
            response.raise_for_status()
        except httpx.ConnectError:  # Handle connection errors
            raise LLMError(f"Connection failed: {self._CONNECT_ERROR_MESSAGE}")
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Ollama request failed: {http_err}")
        except Exception as e:
            if isinstance(e, LLMError):
                raise e
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the API response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        message_data = response_data.get("message", {})
        
        # Normalize content
        normalized_response.choices[0].message.content = message_data.get("content", "")
        
        # Normalize tool_calls if present
        if "tool_calls" in message_data:
            normalized_response.choices[0].message.tool_calls = message_data["tool_calls"]
            
        return normalized_response
