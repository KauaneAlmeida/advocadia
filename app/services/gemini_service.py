"""
Google Gemini API Service

This module provides integration with Google's Gemini API for generating
AI responses. It handles API authentication, request formatting, and
error handling for the Gemini generative AI service.
"""

import os
import httpx
import logging
from typing import Dict, Any
from fastapi import HTTPException, status

# Configure logging
logger = logging.getLogger(__name__)

# Gemini API configuration
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-1.5-flash"  # ✅ Modelo gratuito recomendado
REQUEST_TIMEOUT = 30.0


class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors."""
    pass


async def generate_gemini_response(user_message: str) -> str:
    """
    Generate a response using Google's Gemini API.
    
    Args:
        user_message (str): The user's input message
        
    Returns:
        str: The AI-generated response from Gemini
        
    Raises:
        HTTPException: If API key is missing, request fails, or response is invalid
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini API key not configured"
            )
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": user_message
                        }
                    ]
                }
            ]
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }
        
        # Construct the API endpoint URL
        url = f"{GEMINI_API_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
        
        logger.info(f"Sending request to Gemini API for message: {user_message[:50]}...")
        
        # Make the API request
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                url=url,
                json=payload,
                headers=headers
            )
            
            # Check if request was successful
            if response.status_code != 200:
                error_detail = f"Gemini API request failed with status {response.status_code}"
                try:
                    error_response = response.json()
                    if "error" in error_response:
                        error_detail += f": {error_response['error'].get('message', 'Unknown error')}"
                except Exception:
                    error_detail += f": {response.text}"
                
                logger.error(error_detail)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to get response from Gemini API"
                )
            
            # Parse the response
            response_data = response.json()
            
            # Extract the generated text from the response
            try:
                generated_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"Successfully generated Gemini response: {generated_text[:50]}...")
                return generated_text
                
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Invalid response structure from Gemini API: {e}")
                logger.error(f"Response data: {response_data}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Invalid response format from Gemini API"
                )
                
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except httpx.TimeoutException:
        logger.error("Gemini API request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Gemini API request timed out"
        )
    except httpx.RequestError as e:
        logger.error(f"Network error when calling Gemini API: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Network error when calling Gemini API"
        )
    except Exception as e:
        logger.error(f"Unexpected error in Gemini service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in AI service"
        )


async def get_gemini_service_status() -> Dict[str, Any]:
    """
    Get the current status of the Gemini service.
    
    Returns:
        Dict[str, Any]: Status information about the Gemini service
    """
    api_key_configured = bool(os.getenv("GEMINI_API_KEY"))
    
    return {
        "service": "gemini_service",
        "status": "active" if api_key_configured else "configuration_required",
        "implementation": "google_gemini_api",
        "model": GEMINI_MODEL,
        "api_key_configured": api_key_configured,
        "endpoint": f"{GEMINI_API_BASE_URL}/models/{GEMINI_MODEL}:generateContent",
        "supported_features": [
            "text_generation",
            "conversation",
            "error_handling",
            "timeout_handling"
        ],
        "configuration_notes": [
            "Set GEMINI_API_KEY environment variable",
            "Get API key from Google AI Studio: https://aistudio.google.com/app/apikey"
        ] if not api_key_configured else []
    }


async def test_gemini_connection() -> bool:
    """
    Test the connection to Gemini API with a simple request.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        test_response = await generate_gemini_response("Hello, this is a test message.")
        return bool(test_response)
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False
