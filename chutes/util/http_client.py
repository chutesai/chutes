"""
Reusable HTTP client for Chutes API interactions.

This module provides a centralized HTTP client that handles authentication,
request signing, and common error handling patterns.
"""

import aiohttp
from typing import Optional, Dict, Any, Callable
from loguru import logger


class ChuteHTTPClient:
    """
    Asynchronous HTTP client for Chutes API with built-in authentication.
    
    This client automatically handles:
    - Request signing with Bittensor hotkey
    - Session management
    - Error handling and logging
    - JSON serialization/deserialization
    
    Example:
        >>> from chutes.util.auth import sign_request
        >>> from chutes.config import get_config
        >>> 
        >>> config = get_config()
        >>> async with ChuteHTTPClient(
        ...     base_url=config.generic.api_base_url,
        ...     auth_provider=sign_request
        ... ) as client:
        ...     data = await client.post("/chutes/", {"name": "test"})
    """
    
    def __init__(
        self,
        base_url: str,
        auth_provider: Optional[Callable] = None,
        timeout: Optional[aiohttp.ClientTimeout] = None
    ):
        """
        Initialize the HTTP client.
        
        Args:
            base_url: Base URL for all requests (e.g., "https://api.chutes.ai").
            auth_provider: Function that signs requests, should match signature
                          of util.auth.sign_request.
            timeout: Optional custom timeout configuration.
        """
        self.base_url = base_url
        self.auth_provider = auth_provider
        self.timeout = timeout or aiohttp.ClientTimeout(total=300)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Enter async context manager."""
        self._session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and cleanup session."""
        if self._session:
            await self._session.close()
    
    def _prepare_request(
        self,
        data: Any = None,
        purpose: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> tuple[Dict[str, str], Any]:
        """
        Prepare request headers and payload with authentication.
        
        Args:
            data: Request payload (dict or string).
            purpose: Purpose string for signature (used when data is None).
            extra_headers: Additional headers to merge.
            
        Returns:
            Tuple of (headers dict, prepared payload).
        """
        headers = {}
        payload = data
        
        if self.auth_provider:
            if data is not None:
                auth_headers, payload = self.auth_provider(data)
            else:
                auth_headers, _ = self.auth_provider(purpose=purpose)
            headers.update(auth_headers)
        
        if extra_headers:
            headers.update(extra_headers)
        
        return headers, payload
    
    async def post(
        self,
        path: str,
        data: Any = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict:
        """
        Send a POST request.
        
        Args:
            path: Request path (e.g., "/chutes/").
            data: Request payload.
            extra_headers: Additional headers to include.
            **kwargs: Additional arguments to pass to aiohttp.
            
        Returns:
            Response data as dict.
            
        Raises:
            aiohttp.ClientError: On HTTP errors.
        """
        headers, payload = self._prepare_request(data, extra_headers=extra_headers)
        
        logger.debug(f"POST {path}")
        async with self._session.post(path, data=payload, headers=headers, **kwargs) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                logger.error(f"POST {path} failed with status {resp.status}: {error_text}")
            resp.raise_for_status()
            return await resp.json()
    
    async def post_raw(
        self,
        path: str,
        data: Any = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Send a POST request and return the raw response.
        
        Useful for streaming responses or custom response handling.
        
        Args:
            path: Request path.
            data: Request payload.
            extra_headers: Additional headers.
            **kwargs: Additional arguments to pass to aiohttp.
            
        Returns:
            Raw aiohttp ClientResponse object.
        """
        headers, payload = self._prepare_request(data, extra_headers=extra_headers)
        
        logger.debug(f"POST {path}")
        return await self._session.post(path, data=payload, headers=headers, **kwargs)
    
    async def get(
        self,
        path: str,
        purpose: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict:
        """
        Send a GET request.
        
        Args:
            path: Request path.
            purpose: Purpose string for authentication.
            params: Query parameters.
            extra_headers: Additional headers.
            **kwargs: Additional arguments to pass to aiohttp.
            
        Returns:
            Response data as dict.
            
        Raises:
            aiohttp.ClientError: On HTTP errors.
        """
        headers, _ = self._prepare_request(purpose=purpose or path, extra_headers=extra_headers)
        
        logger.debug(f"GET {path} with params={params}")
        async with self._session.get(path, headers=headers, params=params, **kwargs) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                logger.error(f"GET {path} failed with status {resp.status}: {error_text}")
            resp.raise_for_status()
            return await resp.json()
    
    async def delete(
        self,
        path: str,
        purpose: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict:
        """
        Send a DELETE request.
        
        Args:
            path: Request path.
            purpose: Purpose string for authentication.
            extra_headers: Additional headers.
            **kwargs: Additional arguments to pass to aiohttp.
            
        Returns:
            Response data as dict.
            
        Raises:
            aiohttp.ClientError: On HTTP errors.
        """
        headers, _ = self._prepare_request(purpose=purpose or path, extra_headers=extra_headers)
        
        logger.debug(f"DELETE {path}")
        async with self._session.delete(path, headers=headers, **kwargs) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                logger.error(f"DELETE {path} failed with status {resp.status}: {error_text}")
            resp.raise_for_status()
            return await resp.json()
