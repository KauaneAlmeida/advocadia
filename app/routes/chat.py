from fastapi import APIRouter, HTTPException, status
import logging

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.ai_service import process_chat_message, get_ai_service_status
from app.services.ai_chain import clear_conversation_memory

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Processa uma mensagem do usuário e retorna a resposta da IA.
    """
    try:
        # Log da mensagem recebida
        logger.info(f"Received chat message: {request.message}")

        # Validação da mensagem
        if not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )

        # Processa a mensagem pela IA (sem use_langchain)
        ai_reply = await process_chat_message(
            message=request.message,
            session_id=request.session_id
        )

        # Cria a resposta
        response = ChatResponse(reply=ai_reply)
        logger.info(f"Sending reply: {response.reply}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@router.get("/chat/status")
async def chat_status():
    """
    Retorna o status do serviço de chat e das integrações (LangChain + Gemini).
    """
    try:
        ai_status = await get_ai_service_status()

        return {
            "service": "chat",
            "status": "active" if ai_status["status"] in ["active", "degraded"] else "configuration_required",
            "message": "Chat service is running with LangChain + Gemini integration",
            "ai_integration": ai_status,
            "features": [
                "langchain_conversation_memory",
                "system_prompt_support",
                "gemini_api_integration",
                "automatic_fallback",
                "session_management",
                "brazilian_portuguese_responses"
            ],
            "endpoints": {
                "chat": "/api/v1/chat",
                "status": "/api/v1/chat/status",
                "clear_memory": "/api/v1/chat/clear-memory"
            }
        }

    except Exception as e:
        logger.error(f"Error getting chat status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat status"
        )


@router.post("/chat/clear-memory")
async def clear_memory(session_id: str):
    """
    Limpa a memória de conversa de uma sessão.
    """
    try:
        clear_conversation_memory(session_id)
        return {"message": f"Conversation memory cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing conversation memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear conversation memory"
        )
