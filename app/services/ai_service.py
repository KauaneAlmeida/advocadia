"""
AI Service Layer

Este módulo fornece funções de alto nível para interação com a camada de IA.
Ele atua como uma ponte entre os endpoints da API e a orquestração
LangChain + Gemini definida em `ai_chain.py`.
"""

import logging
from typing import Dict, Any, Optional
from app.services.ai_chain import (
    process_chat_message,
    clear_conversation_memory,
    get_conversation_summary,
    get_ai_service_status,
    ai_orchestrator,
)

# Configure logging
logger = logging.getLogger(__name__)


# Main service functions
async def process_chat_message_service(
    message: str, 
    session_id: str = "default",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Process user message using LangChain + Gemini with context support.
    """
    try:
        logger.info(f"📨 Processing message: {message[:50]}... (session={session_id})")

        # Process via LangChain with context
        response = await process_chat_message(message, session_id=session_id, context=context)

        logger.info(f"✅ Response generated: {response[:50]}...")
        return response

    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        return (
            "Desculpe, ocorreu um erro ao processar sua mensagem. "
            "Por favor, tente novamente mais tarde."
        )


async def get_ai_service_status_service() -> Dict[str, Any]:
    """
    Get current AI service status.
    """
    try:
        return await get_ai_service_status()
    except Exception as e:
        logger.error(f"❌ Error getting AI service status: {str(e)}")
        return {
            "service": "ai_service",
            "status": "error",
            "error": str(e),
            "configuration_required": True,
        }


# Aliases for compatibility
process_chat_message = process_chat_message_service
process_ai_message = process_chat_message_service
get_ai_service_status = get_ai_service_status_service

# Memory management
clear_memory = clear_conversation_memory
get_summary = get_conversation_summary