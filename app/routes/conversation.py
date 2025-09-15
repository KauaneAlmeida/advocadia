"""
Conversation Flow Routes

Now handles intelligent conversation flow using AI orchestration instead of rigid Firebase flows.
The AI manages the entire conversation naturally while still collecting lead information.
"""

import uuid
import logging
import json
import os
from typing import Dict, Any
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
    Start a new conversation session for web platform.
    Uses Firebase flow for structured lead collection.
    """
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"🚀 Starting new web conversation | session={session_id}")

        # Start with Firebase flow for web platform
        result = await intelligent_orchestrator.process_message(
            "Olá", 
            session_id, 
            platform="web"
        )
        
        return ConversationResponse(
            session_id=session_id,
            response=result.get("response", "Olá! Para garantir que registramos corretamente suas informações, vamos começar do início. Tudo bem?"),
            ai_mode=False,
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
    Process user response with Firebase flow for web platform.
    
    The system handles:
    - Structured conversation flow
    - Lead information collection
    - Sequential question progression
    - Phone number collection
    """
    try:
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
            logger.info(f"🆕 New session generated: {request.session_id}")

        logger.info(f"📝 Processing web response | session={request.session_id} | msg={request.message[:50]}...")

        # Process via Intelligent Orchestrator (web platform)
        result = await intelligent_orchestrator.process_message(
            request.message,
            request.session_id,
            platform="web"
        )
        
        return ConversationResponse(
            session_id=request.session_id,
            response=result.get("response", "Como posso ajudá-lo?"),
            ai_mode=False,
            flow_completed=result.get("fallback_completed", False),
            phone_collected=result.get("phone_submitted", False),
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
    Shows platform-specific handling.
    """
    try:
        return {
            "approach": "platform_specific_handling",
            "description": "Web uses Firebase flow, WhatsApp uses AI responses",
            "features": [
                "Platform separation",
                "Web: Structured Firebase flow",
                "WhatsApp: AI-powered responses",
                "No flow repetition",
                "Single final WhatsApp message",
                "Manual lawyer handover"
            ],
            "lead_collection": {
                "method": "structured_web_flow",
                "fields": ["name", "area_of_law", "situation", "consent"],
                "approach": "Sequential questions on web, AI responses on WhatsApp"
            },
            "configuration": {
                "web_flow": "Firebase-based sequential questions",
                "whatsapp_flow": "AI responses only",
                "final_message": "Single formatted WhatsApp message",
                "handover": "Manual lawyer takeover after final message"
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
    Check overall service health with improved error handling.
    """
    try:
        # Get comprehensive status from orchestrator
        service_status = await intelligent_orchestrator.get_overall_service_status()

        return {
            "service": "platform_specific_conversation_service",
            "status": service_status["overall_status"],
            "approach": "platform_separation",
            "firebase_status": service_status["firebase_status"],
            "ai_status": service_status["ai_status"],
            "features": service_status["features"],
            "platforms": {
                "web": "Firebase structured flow",
                "whatsapp": "AI responses only"
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
        return {
            "service": "platform_specific_conversation_service", 
            "status": "error", 
            "approach": "platform_separation",
            "firebase_status": {"status": "unknown"},
            "ai_status": {"status": "unknown"},
            "features": {},
            "error": str(e)
        }