"""
Intelligent Orchestration Service

This service implements a clean separation between Firebase (structured flow) 
and Gemini AI (conversational responses) with proper fallback handling.

Orchestration Logic:
1. Firebase as main flow controller (source of truth for steps)
2. Gemini as secondary assistant (for off-topic/conversational responses)
3. Fallback handling (last resort for failures)

Flow: User message → Orchestrator → [Firebase OR Gemini OR Fallback]
"""

import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from app.services.firebase_service import (
    get_user_session,
    save_user_session,
    save_lead_data,
    get_conversation_flow,
    get_firebase_service_status
)
from app.services.gemini_service import generate_gemini_response, get_gemini_service_status
from app.services.baileys_service import baileys_service

logger = logging.getLogger(__name__)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC timezone aware."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class CleanOrchestrator:
    """
    Clean orchestration service with proper separation of concerns.
    Firebase handles structured flow, Gemini handles conversational responses.
    """

    def __init__(self):
        self.law_firm_number = "+5511918368812"
        self.flow_cache = None
        self.cache_timestamp = None
        
    async def get_overall_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        try:
            # Check Firebase status
            firebase_status = await get_firebase_service_status()
            
            # Check Gemini AI status
            gemini_status = await get_gemini_service_status()
            
            # Determine overall status
            firebase_healthy = firebase_status.get("status") == "active"
            gemini_healthy = gemini_status.get("status") == "active"
            
            if firebase_healthy and gemini_healthy:
                overall_status = "active"
            elif firebase_healthy:
                overall_status = "degraded"  # Firebase works, AI doesn't
            else:
                overall_status = "error"  # Firebase issues are critical
            
            return {
                "overall_status": overall_status,
                "firebase_status": firebase_status,
                "gemini_status": gemini_status,
                "features": {
                    "structured_flow": firebase_healthy,
                    "ai_responses": gemini_healthy,
                    "fallback_mode": firebase_healthy and not gemini_healthy,
                    "whatsapp_integration": True,
                    "lead_collection": firebase_healthy
                },
                "orchestration_mode": "firebase_primary_gemini_secondary"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting overall service status: {str(e)}")
            return {
                "overall_status": "error",
                "firebase_status": {"status": "error", "error": str(e)},
                "gemini_status": {"status": "error", "error": str(e)},
                "features": {},
                "error": str(e)
            }

    async def _get_conversation_flow(self) -> Dict[str, Any]:
        """Get conversation flow with 5-minute caching."""
        try:
            if (self.flow_cache is None or 
                self.cache_timestamp is None or
                (datetime.now(timezone.utc) - self.cache_timestamp).seconds > 300):
                
                self.flow_cache = await get_conversation_flow()
                self.cache_timestamp = datetime.now(timezone.utc)
                logger.info("📋 Conversation flow loaded from Firebase")
            
            return self.flow_cache
        except Exception as e:
            logger.error(f"❌ Error loading conversation flow: {str(e)}")
            # Return minimal default flow
            return {
                "steps": [
                    {"id": 1, "question": "Qual é o seu nome completo?"},
                    {"id": 2, "question": "Em qual área do direito você precisa de ajuda?"},
                    {"id": 3, "question": "Descreva brevemente sua situação."},
                    {"id": 4, "question": "Gostaria de agendar uma consulta?"}
                ],
                "completion_message": "Obrigado! Suas informações foram registradas."
            }

    async def _get_or_create_session(
        self,
        session_id: str,
        platform: str,
        phone_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get existing session or create new one."""
        session_data = await get_user_session(session_id)
        
        if not session_data:
            session_data = {
                "session_id": session_id,
                "platform": platform,
                "created_at": ensure_utc(datetime.now(timezone.utc)),
                "current_step": 1,
                "responses": {},
                "flow_completed": False,
                "phone_collected": False,
                "message_count": 0,
                "last_updated": ensure_utc(datetime.now(timezone.utc))
            }
            logger.info(f"🆕 Created new session {session_id} for platform {platform}")

        if phone_number:
            session_data["phone_number"] = phone_number

        return session_data

    def _is_phone_number(self, message: str) -> bool:
        """Check if message looks like a Brazilian phone number."""
        clean_message = ''.join(filter(str.isdigit, message))
        return len(clean_message) >= 10 and len(clean_message) <= 13

    def _is_step_response(self, message: str, step_id: int) -> bool:
        """
        Determine if user message is a valid response for the current Firebase step.
        This is the key function that decides Firebase vs Gemini routing.
        """
        message = message.strip().lower()
        
        if not message or len(message) < 1:
            return False
            
        # Step-specific validation
        if step_id == 1:  # Name step
            # Must have at least 2 words and reasonable length
            words = message.split()
            return (len(words) >= 2 and 
                    len(message) >= 4 and 
                    all(len(word) >= 2 for word in words) and
                    not any(word in message for word in ['olá', 'oi', 'hello', 'como', 'ajuda', 'preciso']))
                    
        elif step_id == 2:  # Area of law step
            # Check for legal area keywords
            legal_areas = ['penal', 'civil', 'trabalhista', 'família', 'familia', 'empresarial', 
                          'criminal', 'trabalho', 'divórcio', 'divorcio', 'comercial', 'contrato']
            return (len(message) >= 3 and 
                    any(area in message for area in legal_areas))
                    
        elif step_id == 3:  # Situation description step
            # Must be a meaningful description (not a greeting or question)
            return (len(message) >= 10 and 
                    not message.startswith(('olá', 'oi', 'como', 'você', 'qual', 'quando')))
                    
        elif step_id == 4:  # Meeting preference step
            # Check for yes/no type responses
            affirmative = ['sim', 'yes', 'quero', 'gostaria', 'pode', 'claro', 'ok']
            negative = ['não', 'nao', 'no', 'nope', 'talvez', 'depois']
            return any(word in message for word in affirmative + negative)
        
        # Default: if it's not clearly off-topic, consider it a step response
        off_topic_indicators = ['olá', 'oi', 'hello', 'como vai', 'tudo bem', 'ajuda', 
                               'o que', 'como funciona', 'preço', 'valor', 'quanto custa']
        return not any(indicator in message for indicator in off_topic_indicators)

    def _validate_and_normalize_answer(self, answer: str, step_id: int) -> str:
        """Validate and normalize answer for Firebase step."""
        answer = answer.strip()
        
        if step_id == 1:  # Name
            return " ".join(word.capitalize() for word in answer.split())
        elif step_id == 2:  # Area of law
            # Normalize common variations
            area_map = {
                'criminal': 'Penal',
                'penal': 'Penal', 
                'trabalho': 'Trabalhista',
                'trabalhista': 'Trabalhista',
                'família': 'Família',
                'familia': 'Família',
                'divórcio': 'Família',
                'divorcio': 'Família',
                'civil': 'Civil',
                'empresarial': 'Empresarial',
                'comercial': 'Empresarial'
            }
            answer_lower = answer.lower()
            for key, value in area_map.items():
                if key in answer_lower:
                    return value
            return answer.title()
        elif step_id == 3:  # Situation
            return answer  # Accept as-is
        elif step_id == 4:  # Meeting preference
            answer_lower = answer.lower()
            if any(word in answer_lower for word in ['sim', 'yes', 'quero', 'gostaria', 'pode', 'claro', 'ok']):
                return "Sim"
            else:
                return "Não"
        
        return answer

    async def _handle_firebase_step(
        self, 
        message: str, 
        session_data: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Handle Firebase structured flow step.
        Returns (response, step_advanced)
        """
        try:
            session_id = session_data["session_id"]
            current_step = session_data.get("current_step", 1)
            
            logger.info(f"🔥 Firebase handling step {current_step} for session {session_id}")
            
            flow = await self._get_conversation_flow()
            steps = flow.get("steps", [])
            
            # Find current step
            current_step_data = next((s for s in steps if s["id"] == current_step), None)
            if not current_step_data:
                logger.error(f"❌ Step {current_step} not found in flow")
                return "Como posso ajudá-lo?", False
            
            # Validate and store the answer
            normalized_answer = self._validate_and_normalize_answer(message, current_step)
            field_name = f"step_{current_step}"
            
            session_data["responses"][field_name] = normalized_answer
            session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
            
            logger.info(f"💾 Stored answer for step {current_step}: {normalized_answer[:30]}...")
            
            # Find next step
            next_step = current_step + 1
            next_step_data = next((s for s in steps if s["id"] == next_step), None)
            
            if next_step_data:
                # Advance to next step
                session_data["current_step"] = next_step
                await save_user_session(session_id, session_data)
                
                logger.info(f"➡️ Advanced to step {next_step} for session {session_id}")
                return next_step_data["question"], True
            else:
                # Flow completed - ask for phone
                session_data["flow_completed"] = True
                await save_user_session(session_id, session_data)
                
                logger.info(f"✅ Firebase flow completed for session {session_id}")
                return "Obrigado pelas informações! Para finalizar, preciso do seu número de WhatsApp com DDD (exemplo: 11999999999):", True
                
        except Exception as e:
            logger.error(f"❌ Error in Firebase step handling: {str(e)}")
            return "Ocorreu um erro. Pode repetir sua resposta?", False

    async def _handle_gemini_response(
        self, 
        message: str, 
        session_data: Dict[str, Any]
    ) -> str:
        """
        Handle conversational response via Gemini AI.
        Does NOT modify Firebase session state.
        """
        try:
            session_id = session_data["session_id"]
            logger.info(f"🤖 Gemini handling conversational message for session {session_id}")
            
            # Build context from current session
            current_step = session_data.get("current_step", 1)
            responses = session_data.get("responses", {})
            
            # Create context-aware prompt
            context_prompt = f"""Você é um assistente jurídico. O usuário está no meio de um processo de coleta de informações.

Passo atual: {current_step}
Informações já coletadas: {responses}

O usuário disse: "{message}"

Responda de forma conversacional e útil, mas lembre-o gentilmente de que precisa completar o processo de coleta de informações. Seja empático e profissional."""

            # Call Gemini with timeout
            ai_response = await asyncio.wait_for(
                generate_gemini_response(context_prompt),
                timeout=15.0
            )
            
            if ai_response and isinstance(ai_response, str) and ai_response.strip():
                logger.info(f"✅ Gemini response generated for session {session_id}")
                return ai_response
            else:
                logger.warning(f"⚠️ Gemini returned empty response for session {session_id}")
                return self._get_fallback_response()
                
        except asyncio.TimeoutError:
            logger.error(f"⏰ Gemini timeout for session {session_data['session_id']}")
            return self._get_fallback_response()
        except Exception as e:
            logger.error(f"❌ Gemini error for session {session_data['session_id']}: {str(e)}")
            return self._get_fallback_response()

    def _get_fallback_response(self) -> str:
        """Get fallback response when both Firebase and Gemini fail."""
        return ("Desculpe, não consegui processar sua mensagem no momento. "
                "Para continuar, por favor responda à pergunta anterior ou "
                "entre em contato conosco diretamente.")

    async def _handle_phone_collection(
        self, 
        phone_message: str, 
        session_id: str, 
        session_data: Dict[str, Any]
    ) -> str:
        """Handle phone number collection and send WhatsApp message."""
        try:
            # Clean and validate phone number
            phone_clean = ''.join(filter(str.isdigit, phone_message))
            
            if len(phone_clean) < 10 or len(phone_clean) > 13:
                return "Número inválido. Por favor, digite no formato com DDD (exemplo: 11999999999):"

            # Format phone number for WhatsApp
            if len(phone_clean) == 10:
                phone_formatted = f"55{phone_clean[:2]}9{phone_clean[2:]}"
            elif len(phone_clean) == 11:
                phone_formatted = f"55{phone_clean}"
            elif phone_clean.startswith("55"):
                phone_formatted = phone_clean
            else:
                phone_formatted = f"55{phone_clean}"

            whatsapp_number = f"{phone_formatted}@s.whatsapp.net"

            # Update session
            session_data.update({
                "phone_number": phone_clean,
                "phone_formatted": phone_formatted,
                "phone_collected": True,
                "last_updated": ensure_utc(datetime.now(timezone.utc))
            })
            await save_user_session(session_id, session_data)

            # Save lead data
            responses = session_data.get("responses", {})
            answers = []
            for i in range(1, 5):
                answer = responses.get(f"step_{i}", "")
                if answer:
                    answers.append({"id": i, "answer": answer})
            
            # Add phone as final answer
            answers.append({"id": 5, "answer": phone_clean})
            
            try:
                await save_lead_data({"answers": answers})
                logger.info(f"💾 Lead saved for session {session_id}")
            except Exception as save_error:
                logger.error(f"❌ Error saving lead: {str(save_error)}")

            # Prepare WhatsApp message
            user_name = responses.get("step_1", "Cliente")
            area = responses.get("step_2", "não informada")
            situation = responses.get("step_3", "não detalhada")[:150]

            whatsapp_message = f"""Olá {user_name}! 👋

Recebemos sua solicitação através do nosso site e estamos aqui para ajudá-lo com questões jurídicas.

Nossa equipe especializada está pronta para analisar seu caso.

📄 Resumo do caso:
- 👤 Nome: {user_name}
- 📌 Área: {area}
- 📝 Situação: {situation}

Nossa equipe entrará em contato em breve."""

            # Send WhatsApp message
            whatsapp_success = False
            try:
                await baileys_service.send_whatsapp_message(whatsapp_number, whatsapp_message)
                logger.info(f"📤 WhatsApp message sent to {phone_formatted}")
                whatsapp_success = True
            except Exception as whatsapp_error:
                logger.error(f"❌ Error sending WhatsApp: {str(whatsapp_error)}")

            # Return confirmation
            confirmation = f"""Número confirmado: {phone_clean} 📱

Perfeito! Suas informações foram registradas com sucesso. Nossa equipe entrará em contato em breve.

{'✅ Mensagem enviada para seu WhatsApp!' if whatsapp_success else '⚠️ Houve um problema ao enviar a mensagem do WhatsApp, mas suas informações foram salvas.'}"""

            return confirmation

        except Exception as e:
            logger.error(f"❌ Error handling phone collection: {str(e)}")
            return "Ocorreu um erro ao processar seu número. Por favor, tente novamente."

    async def process_message(
        self,
        message: str,
        session_id: str,
        phone_number: Optional[str] = None,
        platform: str = "web"
    ) -> Dict[str, Any]:
        """
        Main orchestration logic: Firebase → Gemini → Fallback
        """
        try:
            logger.info(f"🎯 Orchestrating message - Session: {session_id}, Platform: {platform}")
            logger.info(f"📝 Message: '{message[:100]}...'")

            session_data = await self._get_or_create_session(session_id, platform, phone_number)
            session_data["message_count"] = session_data.get("message_count", 0) + 1

            # Handle phone collection (after flow completion)
            if (session_data.get("flow_completed") and 
                not session_data.get("phone_collected") and 
                self._is_phone_number(message)):
                
                logger.info(f"📱 Processing phone number submission")
                phone_response = await self._handle_phone_collection(message, session_id, session_data)
                return {
                    "response_type": "phone_collected",
                    "platform": platform,
                    "session_id": session_id,
                    "response": phone_response,
                    "phone_collected": True,
                    "message_count": session_data["message_count"]
                }

            # Skip structured flow for WhatsApp platform - use Gemini only
            if platform == "whatsapp":
                logger.info(f"📱 WhatsApp platform - using Gemini only")
                try:
                    ai_response = await self._handle_gemini_response(message, session_data)
                    session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
                    await save_user_session(session_id, session_data)
                    
                    return {
                        "response_type": "gemini_whatsapp",
                        "platform": platform,
                        "session_id": session_id,
                        "response": ai_response,
                        "message_count": session_data["message_count"]
                    }
                except Exception as e:
                    logger.error(f"❌ WhatsApp Gemini error: {str(e)}")
                    return {
                        "response_type": "whatsapp_fallback",
                        "platform": platform,
                        "session_id": session_id,
                        "response": "Obrigado pela sua mensagem. Nossa equipe analisará e retornará em breve.",
                        "message_count": session_data["message_count"]
                    }

            # Web platform: Use orchestration logic
            current_step = session_data.get("current_step", 1)
            
            # STEP 1: Check if this is a Firebase step response
            if not session_data.get("flow_completed") and self._is_step_response(message, current_step):
                logger.info(f"🔥 Routing to Firebase - valid step {current_step} response")
                firebase_response, step_advanced = await self._handle_firebase_step(message, session_data)
                
                return {
                    "response_type": "firebase_step",
                    "platform": platform,
                    "session_id": session_id,
                    "response": firebase_response,
                    "current_step": session_data.get("current_step", current_step),
                    "step_advanced": step_advanced,
                    "flow_completed": session_data.get("flow_completed", False),
                    "message_count": session_data["message_count"]
                }
            
            # STEP 2: Try Gemini for conversational response
            else:
                logger.info(f"🤖 Routing to Gemini - conversational/off-topic message")
                try:
                    gemini_response = await self._handle_gemini_response(message, session_data)
                    session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
                    await save_user_session(session_id, session_data)
                    
                    return {
                        "response_type": "gemini_conversational",
                        "platform": platform,
                        "session_id": session_id,
                        "response": gemini_response,
                        "current_step": session_data.get("current_step", current_step),
                        "flow_completed": session_data.get("flow_completed", False),
                        "message_count": session_data["message_count"]
                    }
                
                # STEP 3: Fallback if Gemini fails
                except Exception as gemini_error:
                    logger.error(f"❌ Gemini failed, using fallback: {str(gemini_error)}")
                    fallback_response = self._get_fallback_response()
                    session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
                    await save_user_session(session_id, session_data)
                    
                    return {
                        "response_type": "fallback",
                        "platform": platform,
                        "session_id": session_id,
                        "response": fallback_response,
                        "current_step": session_data.get("current_step", current_step),
                        "flow_completed": session_data.get("flow_completed", False),
                        "message_count": session_data["message_count"],
                        "error": str(gemini_error)
                    }

        except Exception as e:
            logger.error(f"❌ Critical orchestration error: {str(e)}")
            return {
                "response_type": "critical_error",
                "platform": platform,
                "session_id": session_id,
                "response": "Desculpe, ocorreu um erro interno. Nossa equipe foi notificada.",
                "error": str(e)
            }

    async def handle_phone_number_submission(
        self,
        phone_number: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle phone number submission from web interface."""
        try:
            session_data = await get_user_session(session_id) or {}
            response = await self._handle_phone_collection(phone_number, session_id, session_data)
            return {
                "status": "success",
                "message": response,
                "phone_collected": True
            }
        except Exception as e:
            logger.error(f"❌ Error in phone submission: {str(e)}")
            return {
                "status": "error",
                "message": "Erro ao processar número de WhatsApp",
                "error": str(e)
            }

    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get current session context and status."""
        try:
            session_data = await get_user_session(session_id)
            if not session_data:
                return {"exists": False}

            return {
                "exists": True,
                "session_id": session_id,
                "platform": session_data.get("platform", "unknown"),
                "current_step": session_data.get("current_step", 1),
                "flow_completed": session_data.get("flow_completed", False),
                "phone_collected": session_data.get("phone_collected", False),
                "responses": session_data.get("responses", {}),
                "message_count": session_data.get("message_count", 0),
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated")
            }
        except Exception as e:
            logger.error(f"❌ Error getting session context: {str(e)}")
            return {"exists": False, "error": str(e)}


# Global instance
clean_orchestrator = CleanOrchestrator()

# Aliases for backward compatibility
intelligent_orchestrator = clean_orchestrator
hybrid_orchestrator = clean_orchestrator