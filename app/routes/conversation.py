"""
Conversation Flow Routes

Now handles intelligent conversation flow using AI orchestration instead of rigid Firebase flows.
The AI manages the entire conversation naturally while still collecting lead information.
"""

import uuid
import logging
import json
import os
from fastapi import APIRouter, HTTPException, status

from app.models.request import ConversationRequest
from app.models.response import ConversationResponse
from app.services.orchestration_service import intelligent_orchestrator
from app.services.firebase_service import get_firebase_service_status

# Logging
logger = logging.getLogger(__name__)

# FastAPI router
router = APIRouter()


@router.post("/conversation/start", response_model=ConversationResponse)
async def start_conversation():
    """
    Start a new intelligent conversation session.
    Uses AI orchestration instead of rigid Firebase flows.
    """
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"🚀 Starting new intelligent conversation | session={session_id}")

        # Start with a welcome message via AI
        result = await intelligent_orchestrator.process_message(
            "Olá", 
            session_id, 
            platform="web"
        )
        
        return ConversationResponse(
            session_id=session_id,
            response=result.get("response", "Olá! Como posso ajudá-lo hoje?"),
            ai_mode=True,
            flow_completed=False,
            phone_collected=False
        )

    except Exception as e:
        logger.error(f"❌ Error starting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start conversation"
        )


@router.post("/conversation/respond", response_model=ConversationResponse)
async def respond_to_conversation(request: ConversationRequest):
    """
    Process user response with intelligent AI orchestration.
    
    The AI handles everything:
    - Natural conversation flow
    - Lead information collection
    - Context awareness
    - Flexible response handling
    """
    try:
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
            logger.info(f"🆕 New session generated: {request.session_id}")

        logger.info(f"📝 Processing intelligent response | session={request.session_id} | msg={request.message[:50]}...")

        # Process via Intelligent Orchestrator
        result = await intelligent_orchestrator.process_message(
            request.message,
            request.session_id,
            platform="web"
        )
        
        return ConversationResponse(
            session_id=request.session_id,
            response=result.get("response", "Como posso ajudá-lo?"),
            ai_mode=True,
            flow_completed=True,  # AI manages flow dynamically
            phone_collected=False,  # Will be handled when needed
            lead_data=result.get("lead_data", {}),
            message_count=result.get("message_count", 1)
        )

    except Exception as e:
        logger.error(f"❌ Error processing response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process conversation response"
        )


@router.post("/conversation/submit-phone")
async def submit_phone_number(request: dict):
    """
    Submit phone number and trigger WhatsApp flow.
    """
    try:
        phone_number = request.get("phone_number")
        session_id = request.get("session_id")
        
        if not phone_number or not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing phone_number or session_id"
            )
        
        logger.info(f"📱 Phone submitted | session={session_id} | number={phone_number}")

        result = await intelligent_orchestrator.handle_phone_number_submission(phone_number, session_id)
        return result

    except Exception as e:
        logger.error(f"❌ Error submitting phone number: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process phone number submission"
        )


@router.get("/conversation/status/{session_id}")
async def get_conversation_status(session_id: str):
    """
    Get current conversation state for a session.
    """
    try:
        logger.info(f"📊 Fetching status | session={session_id}")
        status_info = await intelligent_orchestrator.get_session_context(session_id)
        return status_info

    except Exception as e:
        logger.error(f"❌ Error getting status for {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation status"
        )


@router.get("/conversation/ai-config")
async def get_ai_config():
    """
    Get AI system prompt + configuration (debug/admin use).
    """
    try:
        from app.services.ai_chain import ai_orchestrator

        config = {}
        if os.path.exists("ai_schema.json"):
            with open("ai_schema.json", "r", encoding="utf-8") as f:
                config = json.load(f)

        return {
            "current_system_prompt": ai_orchestrator.get_system_prompt(),
            "full_config": config,
            "config_source": "ai_schema.json" if config else "default",
            "editable_location": "ai_schema.json in project root or AI_SYSTEM_PROMPT in .env",
            "environment_prompt": bool(os.getenv("AI_SYSTEM_PROMPT")),
            "api_key_configured": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        }

    except Exception as e:
        logger.error(f"❌ Error getting AI config: {str(e)}")
        return {"error": str(e)}


@router.get("/conversation/flow")
async def get_conversation_flow():
    """
    Get current conversation approach info.
    Now shows AI-powered approach instead of rigid Firebase flows.
    """
    try:
        return {
            "approach": "ai_intelligent_orchestration",
            "description": "Conversation managed by AI (LangChain + Gemini) instead of rigid flows",
            "features": [
                "Natural language processing",
                "Context-aware responses", 
                "Flexible lead collection",
                "Conversation memory",
                "Brazilian Portuguese responses",
                "Empathetic and professional tone",
                "Automatic information extraction",
                "Smart phone collection"
            ],
            "lead_collection": {
                "method": "natural_extraction",
                "fields": ["name", "area_of_law", "situation", "consent"],
                "approach": "AI extracts information naturally from conversation"
            },
            "configuration": {
                "system_prompt": "Configurable via AI_SYSTEM_PROMPT in .env or ai_schema.json",
                "ai_model": "gemini-1.5-flash",
                "memory_window": "10 messages per session",
                "response_style": "Professional, empathetic, Brazilian Portuguese"
            }
        }

    except Exception as e:
        logger.error(f"❌ Error retrieving conversation flow info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation flow information"
        )


@router.get("/conversation/service-status")
async def conversation_service_status():
    """
    Check overall service health: Firebase + AI + intelligent orchestration.
    """
    try:
        # Firebase status
        firebase_status = await get_firebase_service_status()

        # AI service status
        try:
            from app.services.ai_chain import get_ai_service_status
            ai_status = await get_ai_service_status()
        except Exception as e:
            ai_status = {"status": "error", "error": str(e)}

        # Overall status
        overall_status = (
            "active" if firebase_status["status"] == "active" 
            and ai_status["status"] == "active" 
            else "degraded"
        )

        return {
            "service": "intelligent_conversation_service",
            "status": overall_status,
            "approach": "ai_powered_orchestration",
            "firebase_status": firebase_status,
            "ai_status": ai_status,
            "features": {
                "intelligent_responses": ai_status["status"] == "active",
                "lead_collection": True,
                "conversation_memory": True,
                "context_awareness": True,
                "flexible_dialogue": True,
                "whatsapp_integration": True
            },
            "endpoints": {
                "start": "/api/v1/conversation/start",
                "respond": "/api/v1/conversation/respond",
                "submit_phone": "/api/v1/conversation/submit-phone",
                "status": "/api/v1/conversation/status/{session_id}",
                "ai_config": "/api/v1/conversation/ai-config",
                "flow_info": "/api/v1/conversation/flow"
            }
        }

    except Exception as e:
        logger.error(f"❌ Error getting service status: {str(e)}")
        return {"service": "conversation_service", "status": "error", "error": str(e)}