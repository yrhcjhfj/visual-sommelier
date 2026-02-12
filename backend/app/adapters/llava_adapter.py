"""LLaVA adapter for vision-language model via Ollama."""

import logging
from typing import Optional, Dict, Any
import base64
import json

from .base import LLMProviderAdapter
from ..config import settings

logger = logging.getLogger(__name__)


class LLaVAAdapter(LLMProviderAdapter):
    """LLaVA implementation via Ollama."""
    
    def __init__(self):
        self._client = None
        self._model_name = settings.llava_model
        self._ollama_host = settings.ollama_host
        self._max_tokens = settings.llava_max_tokens
        self._temperature = settings.llava_temperature
        self._model_loaded = False
        
    def load_model(self) -> None:
        """Initialize Ollama client and ensure model is available."""
        if self._client is not None:
            logger.debug("LLaVA client already initialized")
            return
        
        try:
            import ollama
            
            # Initialize client
            self._client = ollama.Client(host=self._ollama_host)
            
            # Check if model exists
            try:
                models = self._client.list()
                model_names = [m['name'] for m in models.get('models', [])]
                
                if self._model_name not in model_names:
                    logger.warning(
                        f"Model {self._model_name} not found in Ollama. "
                        f"Available models: {model_names}"
                    )
                    logger.info(f"Attempting to pull {self._model_name}...")
                    # Note: Pulling is handled by Ollama automatically on first use
                else:
                    logger.info(f"Model {self._model_name} is available")
                    
            except Exception as e:
                logger.warning(f"Could not list Ollama models: {e}")
            
            self._model_loaded = True
            logger.info(f"LLaVA adapter initialized with model {self._model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing LLaVA adapter: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload model (Ollama manages this automatically)."""
        # Ollama manages model lifecycle automatically
        # We just clear our client reference
        if self._client is not None:
            self._client = None
            self._model_loaded = False
            logger.info("LLaVA adapter client cleared")
    
    def is_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._model_loaded and self._client is not None
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def generate_completion(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        language: str = "en",
        max_tokens: int = 512
    ) -> str:
        """Generate text completion with optional image."""
        if not self.is_loaded():
            self.load_model()
        
        try:
            # Add language instruction to prompt
            language_map = {
                "en": "English",
                "ru": "Russian",
                "zh": "Chinese"
            }
            lang_name = language_map.get(language, "English")
            
            enhanced_prompt = f"Please respond in {lang_name}.\n\n{prompt}"
            
            # Prepare request
            request_params = {
                "model": self._model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": self._temperature
                }
            }
            
            # Add image if provided
            if image:
                request_params["images"] = [self._encode_image(image)]
            
            # Generate completion
            response = self._client.generate(**request_params)
            
            # Extract response text
            result = response.get("response", "")
            logger.debug(f"Generated completion: {len(result)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate structured output matching a schema."""
        if not self.is_loaded():
            self.load_model()
        
        try:
            # Add schema to prompt
            schema_str = json.dumps(schema, indent=2)
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"Please respond with a JSON object matching this schema:\n"
                f"{schema_str}\n\n"
                f"Return ONLY the JSON object, no additional text."
            )
            
            # Generate completion
            response_text = self.generate_completion(
                prompt=enhanced_prompt,
                language=language,
                max_tokens=self._max_tokens
            )
            
            # Try to extract JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            logger.debug("Generated structured output successfully")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise
        except Exception as e:
            logger.error(f"Error generating structured output: {e}")
            raise
