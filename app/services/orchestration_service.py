import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.services.firebase_service import (
    get_user_session,
    save_user_session,
    save_lead_data,
    get_conversation_flow,
    get_firebase_service_status
)
from app.services.ai_chain import ai_orchestrator
from app.services.baileys_service import baileys_service

logger = logging.getLogger(__name__)


def ensure_utc(dt: datetime) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class IntelligentHybridOrchestrator:
    def __init__(self):
        self.gemini_available = True
        self.gemini_timeout = 15.0
        self.law_firm_number = "+5511918368812"
        self.schema_flow_cache = None
        self.cache_timestamp = None
        
    async def get_gemini_health_status(self) -> Dict[str, Any]:
        """
        Safe health check for Gemini AI service.
        Returns status without raising exceptions.
        """
        try:
            # Quick test of Gemini availability
            test_response = await asyncio.wait_for(
                ai_orchestrator.generate_response(
                    "test", 
                    session_id="__health_check__"
                ),
                timeout=5.0  # Short timeout for health checks
            )
            
            # Clean up test session
            ai_orchestrator.clear_session_memory("__health_check__")
            
            if test_response and isinstance(test_response, str) and test_response.strip():
                self.gemini_available = True
                return {
                    "service": "gemini_ai",
                    "status": "active",
                    "available": True,
                    "message": "Gemini AI is operational"
                }
            else:
                self.gemini_available = False
                return {
                    "service": "gemini_ai", 
                    "status": "inactive",
                    "available": False,
                    "message": "Gemini AI returned invalid response"
                }
                
        except asyncio.TimeoutError:
            self.gemini_available = False
            return {
                "service": "gemini_ai",
                "status": "inactive", 
                "available": False,
                "message": "Gemini AI timeout - likely quota exceeded"
            }
        except Exception as e:
            self.gemini_available = False
            error_str = str(e).lower()
            
            if self._is_quota_error(error_str):
                return {
                    "service": "gemini_ai",
                    "status": "quota_exceeded",
                    "available": False, 
                    "message": f"Gemini API quota exceeded: {str(e)}"
                }
            else:
                return {
                    "service": "gemini_ai",
                    "status": "error",
                    "available": False,
                    "message": f"Gemini AI error: {str(e)}"
                }
    
    async def get_overall_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status including Firebase, AI, and overall health.
        """
        try:
            # Check Firebase status
            firebase_status = await get_firebase_service_status()
            
            # Check Gemini AI status
            ai_status = await self.get_gemini_health_status()
            
            # Determine overall status
            firebase_healthy = firebase_status.get("status") == "active"
            ai_healthy = ai_status.get("status") == "active"
            
            if firebase_healthy and ai_healthy:
                overall_status = "active"
            elif firebase_healthy:
                overall_status = "degraded"  # Firebase works, AI doesn't
            else:
                overall_status = "error"  # Firebase issues are critical
            
            return {
                "overall_status": overall_status,
                "firebase_status": firebase_status,
                "ai_status": ai_status,
                "features": {
                    "conversation_flow": firebase_healthy,
                    "ai_responses": ai_healthy,
                    "fallback_mode": firebase_healthy and not ai_healthy,
                    "whatsapp_integration": True,  # Assumed available
                    "lead_collection": firebase_healthy
                },
                "gemini_available": self.gemini_available,
                "fallback_mode": not self.gemini_available
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting overall service status: {str(e)}")
            return {
                "overall_status": "error",
                "firebase_status": {"status": "error", "error": str(e)},
                "ai_status": {"status": "error", "error": str(e)},
                "features": {
                    "conversation_flow": False,
                    "ai_responses": False,
                    "fallback_mode": False,
                    "whatsapp_integration": False,
                    "lead_collection": False
                },
                "gemini_available": False,
                "fallback_mode": True,
                "error": str(e)
            }

    async def _get_or_create_session(
        self,
        session_id: str,
        platform: str,
        phone_number: Optional[str] = None
    ) -> Dict[str, Any]:
        session_data = await get_user_session(session_id)
        
        if not session_data:
            # Create new session with proper initialization
            session_data = {
                "session_id": session_id,
                "platform": platform,
                "created_at": ensure_utc(datetime.now(timezone.utc)),
                "lead_data": {},
                "message_count": 0,
                "fallback_step": None,
                "phone_submitted": False,
                "gemini_available": True,
                "last_gemini_check": None,
                "fallback_completed": False
            }
            logger.info(f"🆕 Created new session {session_id} for platform {platform}")

        if phone_number:
            session_data["phone_number"] = phone_number

        return session_data

    def _is_quota_error(self, error_message: str) -> bool:
        """Check if error is related to API quota/rate limits."""
        quota_indicators = [
            "429", "quota", "rate limit", "exceeded", "ResourceExhausted",
            "billing", "plan", "free tier", "requests per day"
        ]
        return any(indicator.lower() in str(error_message).lower() for indicator in quota_indicators)

    def _is_phone_number(self, message: str) -> bool:
        """Check if message looks like a Brazilian phone number."""
        clean_message = ''.join(filter(str.isdigit, message))
        return len(clean_message) >= 10 and len(clean_message) <= 13

    async def _get_schema_flow(self) -> Dict[str, Any]:
        """Get schema-based conversation flow with caching."""
        try:
            # Cache for 5 minutes
            if (self.schema_flow_cache is None or 
                self.cache_timestamp is None or
                (datetime.now(timezone.utc) - self.cache_timestamp).seconds > 300):
                
                # Load from ai_schema.json first
                schema_path = "ai_schema.json"
                if os.path.exists(schema_path):
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema_data = json.load(f)
                        fallback_flow = schema_data.get("fallback_flow", {})
                        
                        if fallback_flow.get("enabled", False):
                            self.schema_flow_cache = fallback_flow
                            self.cache_timestamp = datetime.now(timezone.utc)
                            logger.info("📋 Schema-based conversation flow loaded from ai_schema.json")
                            return self.schema_flow_cache
                
                # Fallback to Firebase if schema not available
                firebase_flow = await get_conversation_flow()
                # Convert Firebase format to schema format for compatibility
                self.schema_flow_cache = {
                    "enabled": True,
                    "sequential": True,
                    "steps": [
                        {
                            "id": 0,
                            "field": "review_intro", 
                            "question": "Olá! Para garantir que registramos corretamente suas informações, vamos começar do início. Tudo bem?",
                            "validation": {"type": "confirmation", "required": True}
                        }
                    ] + [
                        {
                            "id": step.get("id", idx),
                            "field": f"step_{step.get('id', idx)}",
                            "question": step.get("question", ""),
                            "validation": {"min_length": 1, "required": True}
                        }
                        for idx, step in enumerate(firebase_flow.get("steps", []), 1)
                    ],
                    "completion_message": firebase_flow.get("completion_message", "Obrigado! Suas informações foram registradas."),
                    "whatsapp_messages": {
                        "welcome_message": "Olá {user_name}! Recebemos sua solicitação e nossa equipe entrará em contato.",
                        "case_summary": "📄 Resumo: Nome: {user_name}, Área: {area}, Situação: {situation}"
                    }
                }
                self.cache_timestamp = datetime.now(timezone.utc)
                logger.info("📋 Fallback to Firebase conversation flow (converted to schema format)")
            
            return self.schema_flow_cache
        except Exception as e:
            logger.error(f"❌ Error loading schema flow: {str(e)}")
            # Return default schema flow if everything fails
            return {
                "enabled": True,
                "sequential": True,
                "steps": [
                    {
                        "id": 0,
                        "field": "review_intro",
                        "question": "Olá! Para garantir que registramos corretamente suas informações, vamos começar do início. Tudo bem?",
                        "validation": {"type": "confirmation", "required": True}
                    },
                    {"id": 1, "field": "name", "question": "Qual é o seu nome completo?", "validation": {"min_length": 2}},
                    {"id": 2, "field": "area_of_law", "question": "Em qual área do direito você precisa de ajuda?", "validation": {"min_length": 3}},
                    {"id": 3, "field": "situation", "question": "Descreva brevemente sua situação.", "validation": {"min_length": 5}},
                    {"id": 4, "field": "phone_request", "question": "Preciso do seu número de WhatsApp:", "validation": {"min_length": 10}}
                ],
                "completion_message": "Obrigado! Suas informações foram registradas.",
                "whatsapp_messages": {
                    "welcome_message": "Olá! Recebemos sua solicitação.",
                    "case_summary": "Resumo: {user_name}, {area}, {situation}"
                }
            }

    async def _get_fallback_response(
        self, 
        session_data: Dict[str, Any], 
        message: str
    ) -> str:
        """
        Firebase-based fallback for WEB platform only.
        Sequential question flow: Step 0 → Step 1 → Step 2 → Step 3 → Step 4 → Phone Collection
        """
        try:
            session_id = session_data["session_id"]
            platform = session_data.get("platform", "web")
            
            # Only run Firebase flow for web platform
            if platform != "web":
                logger.info(f"🚫 Firebase flow skipped for platform {platform}")
                return "Olá! Como posso ajudá-lo?"
            
            logger.info(f"⚡ Firebase fallback activated for web session {session_id}")
            
            # Get schema-based conversation flow
            flow = await self._get_schema_flow()
            steps = flow.get("steps", [])
            
            if not steps:
                logger.error("❌ No steps found in schema flow")
                return "Qual é o seu nome completo?"  # Fallback to step 1
            
            # Sort steps by ID to ensure correct order
            steps = sorted(steps, key=lambda x: x.get("id", 0))
            
            # Initialize fallback_step if not set - ALWAYS start at step 1
            if session_data.get("fallback_step") is None:
                session_data["fallback_step"] = 0  # Always start at step 0
                session_data["lead_data"] = {}  # Initialize lead data
                session_data["fallback_completed"] = False  # Ensure not completed
                await save_user_session(session_id, session_data)
                logger.info(f"🚀 Schema fallback initialized at step 0 for session {session_id}")
                
                # Return first question directly
                first_step = next((s for s in steps if s["id"] == 0), None)
                if first_step:
                    question = self._interpolate_message(first_step["question"], session_data.get("lead_data", {}))
                    logger.info(f"📝 Returning step 0 question: {question[:50]}...")
                    return question
                else:
                    logger.error(f"❌ Step 0 not found in schema, using fallback question")
                    return "Olá! Para garantir que registramos corretamente suas informações, vamos começar do início. Tudo bem?"
            
            current_step_id = session_data["fallback_step"]
            lead_data = session_data.get("lead_data", {})
            
            logger.info(f"📊 Current fallback state - Step: {current_step_id}, Lead data keys: {list(lead_data.keys())}")
            
            # Find current step in sorted steps
            current_step = next((s for s in steps if s["id"] == current_step_id), None)
            if not current_step:
                logger.error(f"❌ Step {current_step_id} not found in schema steps. Available steps: {[s.get('id') for s in steps]}")
                # Reset to step 0 and ensure it exists
                session_data["fallback_step"] = 0
                session_data["lead_data"] = {}
                session_data["fallback_completed"] = False
                await save_user_session(session_id, session_data)
                
                # Find step 0 or create default
                first_step = next((s for s in steps if s.get("id") == 0), None)
                if first_step:
                    question = self._interpolate_message(first_step.get("question", ""), {})
                    logger.info(f"📝 Reset to step 0, returning: {question[:50]}...")
                    return question
                else:
                    logger.error(f"❌ Critical error: Step 0 not found in schema after reset")
                    return "Olá! Para garantir que registramos corretamente suas informações, vamos começar do início. Tudo bem?"
            
            # Process user's answer if provided and not empty
            step_key = current_step.get("field", f"step_{current_step_id}")
            
            # If user provided a meaningful answer
            if message and message.strip() and len(message.strip()) > 0:
                # Check if we already have an answer for this step
                if step_key in lead_data:
                    # Answer already stored, but let's validate if we should advance
                    logger.info(f"📝 Answer already exists for step {current_step_id}: {lead_data[step_key][:30]}...")
                    # Check if we should advance to next step
                    if self._should_advance_step_schema(lead_data[step_key], current_step):
                        # Move to next step
                        pass  # Will be handled below
                    else:
                        # Re-prompt current step
                        question = self._interpolate_message(current_step["question"], lead_data)
                        return question
                else:
                    # Validate and store the answer
                    normalized_answer = self._validate_and_normalize_answer_schema(message, current_step)
                    
                    if not self._should_advance_step_schema(normalized_answer, current_step):
                        # Re-prompt same step with validation message
                        logger.info(f"🔄 Invalid answer '{normalized_answer[:30]}...' for step {current_step_id}, re-prompting")
                        validation_msg = current_step.get("error_message", "Por favor, forneça uma resposta válida.")
                        question = self._interpolate_message(current_step["question"], lead_data)
                        return f"{validation_msg}\n\n{question}"
                    
                    # Store the valid answer
                    lead_data[step_key] = normalized_answer
                    session_data["lead_data"] = lead_data
                    
                    logger.info(f"💾 Stored answer for step {current_step_id}: {normalized_answer[:30]}...")
                
                # Find next step in sequence
                next_step_id = current_step_id + 1
                next_step = next((s for s in steps if s["id"] == next_step_id), None)
                
                if next_step:
                    # Advance to next step
                    session_data["fallback_step"] = next_step_id
                    await save_user_session(session_id, session_data)
                    logger.info(f"➡️ Advanced to step {next_step_id} for session {session_id}")
                    return self._interpolate_message(next_step.get("question", ""), lead_data)
                else:
                    # All steps completed - mark as completed and ask for phone
                    session_data["fallback_completed"] = True
                    await save_user_session(session_id, session_data)
                    logger.info(f"✅ Schema fallback flow completed for session {session_id}")
                    return "Obrigado pelas informações! Para finalizar, preciso do seu número de WhatsApp com DDD (exemplo: 11999999999):"
            else:
                # No meaningful message provided, return current question
                logger.info(f"📝 No meaningful message provided, returning current step {current_step_id} question")
                return self._interpolate_message(current_step.get("question", ""), lead_data)
            
            # Fallback: return current question
            logger.info(f"📝 Fallback: returning current step {current_step_id} question")
            return self._interpolate_message(current_step.get("question", ""), lead_data)
            
        except Exception as e:
            logger.error(f"❌ Error in Firebase fallback: {str(e)}")
            # Always fallback to step 1 on error
            return "Olá! Para garantir que registramos corretamente suas informações, vamos começar do início. Tudo bem?"

    def _interpolate_message(self, message: str, lead_data: Dict[str, Any]) -> str:
        """Interpolate variables in message template."""
        try:
            if not message:
                return "Como posso ajudá-lo?"
                
            # Map common field names to user-friendly variables
            interpolation_data = {
                "user_name": lead_data.get("name", lead_data.get("step_1", "")),
                "area": lead_data.get("area_of_law", lead_data.get("step_2", "")),
                "situation": lead_data.get("situation", lead_data.get("step_3", "")),
                "phone": lead_data.get("phone", "")
            }
            
            # Only interpolate if we have the required data
            for key, value in interpolation_data.items():
                if value and f"{{{key}}}" in message:
                    message = message.replace(f"{{{key}}}", value)
            
            return message
        except Exception as e:
            logger.error(f"❌ Error interpolating message: {str(e)}")
            return message

    def _validate_and_normalize_answer_schema(self, answer: str, step_config: Dict[str, Any]) -> str:
        """Validate and normalize answers based on schema configuration."""
        answer = answer.strip()
        step_id = step_config.get("id", 0)
        validation = step_config.get("validation", {})
        
        # Apply normalization map if available
        normalize_map = validation.get("normalize_map", {})
        if normalize_map:
            answer_lower = answer.lower()
            for keyword, normalized in normalize_map.items():
                if keyword in answer_lower:
                    return normalized
        
        # Field-specific validation and normalization
        field_type = validation.get("type", "")
        
        if field_type == "name" or step_id == 1:
            # Ensure proper name format (capitalize each word)
            return " ".join(word.capitalize() for word in answer.split())
        elif field_type == "area" or step_id == 2:
            return answer.title()
        elif field_type == "description" or step_id == 3:
            return answer  # Accept any description
        elif field_type == "phone":
            # Clean phone number
            return ''.join(filter(str.isdigit, answer))
        
        return answer

    def _should_advance_step_schema(self, answer: str, step_config: Dict[str, Any]) -> bool:
        """Determine if answer is sufficient to advance to next step based on schema."""
        answer = answer.strip()
        validation = step_config.get("validation", {})
        min_length = validation.get("min_length", 1)
        required = validation.get("required", True)
        step_id = step_config.get("id", 0)
        
        # Check if required and empty
        if required and (not answer or len(answer) < 1):
            return False
        
        # Check minimum length requirement
        if len(answer) < min_length:
            return False
        
        # Step-specific validation
        if step_id == 0:  # Review intro confirmation
            # Accept confirmation variations
            answer_lower = answer.lower()
            confirmation_responses = ['sim', 'ok', 'tudo bem', 'pode ser', 'claro', 'yes', 'certo', 'vamos', 'confirmo', 'vamos lá', 'perfeito', 'beleza']
            return any(response in answer_lower for response in confirmation_responses)
        elif step_id == 1:  # Name validation
            # Require at least 2 words for full name
            words = answer.split()
            return len(words) >= 2 and all(len(word) >= 2 for word in words)
        elif step_id == 2:  # Area validation
            # Accept any area with minimum length
            return len(answer) >= 3
        elif step_id == 3:  # Situation validation
            # Require meaningful description
            return len(answer) >= 5
        elif step_id == 4:  # Meeting preference validation
            # Accept yes/no variations
            answer_lower = answer.lower()
            valid_responses = ['sim', 'não', 'nao', 'yes', 'no', 'quero', 'gostaria', 'pode ser', 'ok', 'claro']
            return any(response in answer_lower for response in valid_responses)
        
        # Default validation - just check minimum length
        return len(answer) >= min_length

    async def _handle_phone_collection(
        self, 
        phone_message: str, 
        session_id: str, 
        session_data: Dict[str, Any]
    ) -> str:
        """
        Handle phone number collection and send final WhatsApp message.
        """
        try:
            # Clean and validate phone number
            phone_clean = ''.join(filter(str.isdigit, phone_message))
            
            # Validate Brazilian phone number format
            if len(phone_clean) < 10 or len(phone_clean) > 13:
                return "Número inválido. Por favor, digite no formato com DDD (exemplo: 11999999999):"

            # Format phone number for WhatsApp
            if len(phone_clean) == 10:  # Add 9th digit for mobile
                phone_formatted = f"55{phone_clean[:2]}9{phone_clean[2:]}"
            elif len(phone_clean) == 11:  # Already has 9th digit
                phone_formatted = f"55{phone_clean}"
            elif phone_clean.startswith("55"):
                phone_formatted = phone_clean
            else:
                phone_formatted = f"55{phone_clean}"

            whatsapp_number = f"{phone_formatted}@s.whatsapp.net"

            # Update session data
            session_data.update({
                "phone_number": phone_clean,
                "phone_formatted": phone_formatted,
                "phone_submitted": True,
                "last_updated": ensure_utc(datetime.now(timezone.utc))
            })
            
            # Store phone in lead_data
            session_data["lead_data"]["phone"] = phone_clean
            await save_user_session(session_id, session_data)

            # Build answers array for lead saving
            lead_data = session_data.get("lead_data", {})
            answers = []
            
            # Get conversation flow to map step IDs to answers
            flow = await self._get_schema_flow()
            steps = flow.get("steps", [])
            
            for step in sorted(steps, key=lambda x: x.get("id", 0)):
                step_key = step.get("field", f"step_{step['id']}")
                answer = lead_data.get(step_key, "")
                if answer:
                    answers.append({"id": step["id"], "answer": answer})
            
            # Add phone as final answer
            if phone_clean:
                answers.append({"id": len(steps) + 1, "answer": phone_clean})

            # Save lead data
            try:
                await save_lead_data({"answers": answers})
                logger.info(f"💾 Lead saved for session {session_id}: {len(answers)} answers")
            except Exception as save_error:
                logger.error(f"❌ Error saving lead: {str(save_error)}")

            # Prepare data for final WhatsApp message
            user_name = lead_data.get("name", lead_data.get("step_1", lead_data.get("step_0", "Cliente")))
            area = lead_data.get("area_of_law", lead_data.get("step_2", "não informada"))
            situation_full = lead_data.get("situation", lead_data.get("step_3", "não detalhada"))
            situation = situation_full[:150]
            if len(situation_full) > 150:
                situation += "..."

            # Create the final WhatsApp message with the exact format requested
            final_whatsapp_message = f"""Olá {user_name}! 👋

Recebemos sua solicitação através do nosso site e estamos aqui para ajudá-lo com questões jurídicas.

Nossa equipe especializada está pronta para analisar seu caso. 
Vamos continuar nossa conversa aqui no WhatsApp para maior comodidade. 🤝

📄 Resumo do caso enviado pelo cliente:

- 👤 Nome: {user_name}
- 📌 Área de atuação: {area}
- 📝 Situação: {situation}

Essas informações foram coletadas na landing page e estão vinculadas a este contato."""

            # Send single final WhatsApp message
            whatsapp_success = False
            try:
                # Send final message to user
                await baileys_service.send_whatsapp_message(whatsapp_number, final_whatsapp_message)
                logger.info(f"📤 Final WhatsApp message sent to user {phone_formatted}")
                
                whatsapp_success = True
                
            except Exception as whatsapp_error:
                logger.error(f"❌ Error sending WhatsApp messages: {str(whatsapp_error)}")
                whatsapp_success = False

            # Return confirmation message for web interface
            final_message = f"""Número confirmado: {phone_clean} 📱

Perfeito! Suas informações foram registradas com sucesso. Nossa equipe entrará em contato em breve.

{'✅ Mensagem enviada para seu WhatsApp!' if whatsapp_success else '⚠️ Houve um problema ao enviar a mensagem do WhatsApp, mas suas informações foram salvas.'}"""

            return final_message

        except Exception as e:
            logger.error(f"❌ Error handling phone collection: {str(e)}")
            return "Ocorreu um erro ao processar seu número. Por favor, tente novamente ou entre em contato conosco diretamente."

    async def process_message(
        self,
        message: str,
        session_id: str,
        phone_number: Optional[str] = None,
        platform: str = "web"
    ) -> Dict[str, Any]:
        """
        Main message processing with platform-specific handling.
        - Web: Uses Firebase fallback flow
        - WhatsApp: Uses AI responses only (no Firebase flow repetition)
        """
        try:
            logger.info(f"🎯 Processing message - Session: {session_id}, Platform: {platform}")
            logger.info(f"📝 Message content: '{message[:100]}...' (length: {len(message)})")

            session_data = await self._get_or_create_session(session_id, platform, phone_number)
            logger.info(f"📊 Session state - Fallback step: {session_data.get('fallback_step')}, Completed: {session_data.get('fallback_completed')}, Phone submitted: {session_data.get('phone_submitted')}")

            # Handle phone collection for web platform only
            if (platform == "web" and
                session_data.get("fallback_completed") and 
                not session_data.get("phone_submitted") and 
                self._is_phone_number(message)):
                
                logger.info(f"📱 Processing phone number submission in fallback mode")
                phone_response = await self._handle_phone_collection(message, session_id, session_data)
                return {
                    "response_type": "phone_collected_fallback",
                    "platform": platform,
                    "session_id": session_id,
                    "response": phone_response,
                    "phone_submitted": True,
                    "message_count": session_data.get("message_count", 0) + 1
                }

            # Platform-specific handling
            if platform == "whatsapp":
                # WhatsApp: Use AI responses only, no Firebase flow
                logger.info(f"📱 WhatsApp platform - using AI responses only")
                try:
                    # Prepare context from any existing lead data
                    lead_data = session_data.get("lead_data", {})
                    context = {
                        "platform": "whatsapp",
                        "name": lead_data.get("name", lead_data.get("step_1", "Não informado")),
                        "area_of_law": lead_data.get("area_of_law", lead_data.get("step_2", "Não informada")),
                        "situation": lead_data.get("situation", lead_data.get("step_3", "Não detalhada"))
                    }
                    
                    # Call AI with timeout
                    ai_response = await asyncio.wait_for(
                        ai_orchestrator.generate_response(
                            message,
                            session_id,
                            context=context
                        ),
                        timeout=self.gemini_timeout
                    )
                    
                    if ai_response and isinstance(ai_response, str) and ai_response.strip():
                        session_data["last_message"] = message
                        session_data["last_response"] = ai_response
                        session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
                        session_data["message_count"] = session_data.get("message_count", 0) + 1
                        await save_user_session(session_id, session_data)

                        return {
                            "response_type": "ai_whatsapp",
                            "platform": platform,
                            "session_id": session_id,
                            "response": ai_response,
                            "ai_mode": True,
                            "message_count": session_data.get("message_count", 1)
                        }
                    else:
                        # AI failed, provide simple fallback
                        fallback_response = "Obrigado pela sua mensagem. Nossa equipe analisará sua solicitação e retornará em breve."
                        return {
                            "response_type": "whatsapp_fallback",
                            "platform": platform,
                            "session_id": session_id,
                            "response": fallback_response,
                            "ai_mode": False,
                            "message_count": session_data.get("message_count", 0) + 1
                        }
                        
                except Exception as ai_error:
                    logger.error(f"❌ AI error for WhatsApp: {str(ai_error)}")
                    fallback_response = "Obrigado pela sua mensagem. Nossa equipe analisará sua solicitação e retornará em breve."
                    return {
                        "response_type": "whatsapp_error_fallback",
                        "platform": platform,
                        "session_id": session_id,
                        "response": fallback_response,
                        "ai_mode": False,
                        "error": str(ai_error),
                        "message_count": session_data.get("message_count", 0) + 1
                    }
            
            elif platform == "web":
                # Web: Use Firebase fallback flow
                logger.info(f"🌐 Web platform - using Firebase fallback flow")
                fallback_response = await self._get_fallback_response(session_data, message)
                
                # Update session
                session_data["last_message"] = message
                session_data["last_response"] = fallback_response
                session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
                session_data["message_count"] = session_data.get("message_count", 0) + 1
                await save_user_session(session_id, session_data)
                
                return {
                    "response_type": "web_firebase_flow",
                    "platform": platform,
                    "session_id": session_id,
                    "response": fallback_response,
                    "ai_mode": False,
                    "fallback_step": session_data.get("fallback_step"),
                    "fallback_completed": session_data.get("fallback_completed", False),
                    "lead_data": session_data.get("lead_data", {}),
                    "message_count": session_data.get("message_count", 1)
                }
            else:
                # Unknown platform - provide generic response
                logger.warning(f"⚠️ Unknown platform: {platform}")
                return {
                    "response_type": "unknown_platform",
                    "platform": platform,
                    "session_id": session_id,
                    "response": "Como posso ajudá-lo?",
                    "ai_mode": False,
                    "message_count": session_data.get("message_count", 0) + 1
                }

        except Exception as e:
            logger.error(f"❌ Error in orchestration: {str(e)}")
            return {
                "response_type": "error",
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
        """
        Handle phone number submission from web interface.
        """
        try:
            session_data = await get_user_session(session_id) or {}
            response = await self._handle_phone_collection(phone_number, session_id, session_data)
            return {
                "status": "success",
                "message": response,
                "phone_submitted": True
            }
        except Exception as e:
            logger.error(f"❌ Error in handle_phone_number_submission: {str(e)}")
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
                "fallback_step": session_data.get("fallback_step"),
                "fallback_completed": session_data.get("fallback_completed", False),
                "phone_submitted": session_data.get("phone_submitted", False),
                "lead_data": session_data.get("lead_data", {}),
                "message_count": session_data.get("message_count", 0),
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated")
            }
        except Exception as e:
            logger.error(f"❌ Error getting session context: {str(e)}")
            return {"exists": False, "error": str(e)}


# Global instance
intelligent_orchestrator = IntelligentHybridOrchestrator()
hybrid_orchestrator = intelligent_orchestrator