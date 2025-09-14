import logging
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
        self.gemini_available = True  # Default to True, will be updated based on actual status
        self.gemini_timeout = 15.0  # 15 second timeout for Gemini calls
        self.law_firm_number = "+5511918368812"  # Internal notification number
        self.firebase_flow_cache = None  # Cache for Firebase flow
        self.cache_timestamp = None  # Cache timestamp
        
    async def get_gemini_health_status(self) -> Dict[str, Any]:
        """
        Safe health check for Gemini AI service.
        Returns status without raising exceptions.
        """
        try:
            # Quick test of Gemini availability
            import asyncio
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
        session_data = await get_user_session(session_id) or {
            "session_id": session_id,
            "platform": platform,
            "created_at": ensure_utc(datetime.now(timezone.utc)),
            "lead_data": {},
            "message_count": 0,
            "fallback_step": None,  # Current step ID in fallback mode
            "phone_submitted": False,
            "gemini_available": True,
            "last_gemini_check": None,
            "fallback_completed": False
        }

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

    async def _attempt_gemini_response(
        self, 
        message: str, 
        session_id: str, 
        session_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Attempt to get response from Gemini AI with timeout and error handling.
        Returns None if Gemini is unavailable or fails.
        """
        if not session_data.get("gemini_available", True):
            logger.info(f"🚫 Gemini marked unavailable for session {session_id}")
            return None

        try:
            logger.info(f"🤖 Attempting Gemini AI response for session {session_id}")
            
            # Prepare context from collected lead data
            lead_data = session_data.get("lead_data", {})
            context = {
                "platform": session_data.get("platform", "web"),
                "name": lead_data.get("step_1", "Não informado"),
                "area_of_law": lead_data.get("step_2", "Não informada"),
                "situation": lead_data.get("step_3", "Não detalhada")
            }
            
            import asyncio
            
            # Call Gemini with timeout
            gemini_response = await asyncio.wait_for(
                ai_orchestrator.generate_response(
                    message,
                    session_id,
                    context=context
                ),
                timeout=self.gemini_timeout
            )
            
            # Validate response
            if (gemini_response and 
                isinstance(gemini_response, str) and 
                gemini_response.strip() and
                not self._is_quota_error(gemini_response)):
                
                logger.info(f"✅ Valid Gemini response received for session {session_id}")
                
                # Mark Gemini as available if it was previously unavailable
                if not session_data.get("gemini_available", True):
                    session_data["gemini_available"] = True
                    session_data["last_gemini_check"] = ensure_utc(datetime.now(timezone.utc))
                    await save_user_session(session_id, session_data)
                    logger.info(f"🔄 Gemini restored for session {session_id}")
                
                return gemini_response
            else:
                logger.warning(f"⚠️ Invalid Gemini response for session {session_id}")
                return None
                
        except asyncio.TimeoutError:
            logger.error(f"⏰ Gemini timeout for session {session_id}")
            await self._mark_gemini_unavailable(session_id, session_data, "timeout")
            return None
        except Exception as e:
            error_str = str(e)
            logger.error(f"❌ Gemini error for session {session_id}: {error_str}")
            
            # Check if it's a quota/rate limit error
            if self._is_quota_error(error_str):
                await self._mark_gemini_unavailable(session_id, session_data, f"quota: {error_str}")
            else:
                await self._mark_gemini_unavailable(session_id, session_data, f"error: {error_str}")
            
            return None

    async def _mark_gemini_unavailable(
        self, 
        session_id: str, 
        session_data: Dict[str, Any], 
        reason: str
    ):
        """Mark Gemini as unavailable for this session."""
        session_data["gemini_available"] = False
        session_data["last_gemini_check"] = ensure_utc(datetime.now(timezone.utc))
        await save_user_session(session_id, session_data)
        logger.warning(f"🚫 Gemini marked unavailable for session {session_id}: {reason}")

    async def _get_firebase_flow(self) -> Dict[str, Any]:
        """Get Firebase conversation flow with caching."""
        try:
            # Cache for 5 minutes
            if (self.firebase_flow_cache is None or 
                self.cache_timestamp is None or
                (datetime.now(timezone.utc) - self.cache_timestamp).seconds > 300):
                
                self.firebase_flow_cache = await get_conversation_flow()
                self.cache_timestamp = datetime.now(timezone.utc)
                logger.info("📋 Firebase conversation flow loaded and cached")
            
            return self.firebase_flow_cache
        except Exception as e:
            logger.error(f"❌ Error loading Firebase flow: {str(e)}")
            # Return default flow if Firebase fails
            return {
                "steps": [
                    {"id": 1, "question": "Qual é o seu nome completo?"},
                    {"id": 2, "question": "Em qual área do direito você precisa de ajuda?"},
                    {"id": 3, "question": "Descreva brevemente sua situação."},
                    {"id": 4, "question": "Gostaria de agendar uma consulta?"}
                ],
                "completion_message": "Obrigado! Suas informações foram registradas."
            }

    async def _get_fallback_response(
        self, 
        session_data: Dict[str, Any], 
        message: str
    ) -> str:
        """
        STRICT Firebase fallback: Sequential question flow, no randomization.
        Enforces exact order: Step 1 → Step 2 → Step 3 → Step 4 → Phone Collection
        """
        try:
            session_id = session_data["session_id"]
            logger.info(f"⚡ STRICT Firebase fallback activated for session {session_id}")
            
            # Get Firebase conversation flow
            flow = await self._get_firebase_flow()
            steps = flow.get("steps", [])
            
            if not steps:
                logger.error("❌ No steps found in Firebase flow")
                return "Qual é o seu nome completo?"  # Fallback to step 1
            
            # Sort steps by ID to ensure correct order
            steps = sorted(steps, key=lambda x: x.get("id", 0))
            
            # Initialize fallback_step if not set - ALWAYS start at step 1
            if session_data.get("fallback_step") is None:
                session_data["fallback_step"] = 1  # Always start at step 1
                session_data["lead_data"] = {}  # Initialize lead data
                await save_user_session(session_id, session_data)
                logger.info(f"🚀 STRICT fallback initialized at step 1 for session {session_id}")
                
                # Return first question immediately
                first_step = next((s for s in steps if s["id"] == 1), None)
                if first_step:
                    return first_step["question"]
                else:
                    return "Qual é o seu nome completo?"
            
            current_step_id = session_data["fallback_step"]
            lead_data = session_data.get("lead_data", {})
            
            # Find current step in sorted steps
            current_step = next((s for s in steps if s["id"] == current_step_id), None)
            if not current_step:
                logger.error(f"❌ Step {current_step_id} not found, resetting to step 1")
                session_data["fallback_step"] = 1
                await save_user_session(session_id, session_data)
                first_step = next((s for s in steps if s["id"] == 1), None)
                return first_step["question"] if first_step else "Qual é o seu nome completo?"
            
            # Process user's answer if provided
            step_key = f"step_{current_step_id}"
            
            # If user provided an answer and we haven't stored it yet
            if message and message.strip():
                # Check if we already have an answer for this step
                if step_key in lead_data:
                    # Answer already stored, move to next step logic
                    logger.info(f"📝 Answer already stored for step {current_step_id}, checking next step")
                else:
                    # Validate and store the answer
                    normalized_answer = self._validate_and_normalize_answer(message, current_step_id)
                    
                    if not self._should_advance_step(normalized_answer, current_step_id):
                        # Re-prompt same step with validation message
                        logger.info(f"🔄 Invalid answer for step {current_step_id}, re-prompting")
                        validation_msg = self._get_validation_message(current_step_id)
                        return f"{validation_msg}\n\n{current_step['question']}"
                    
                    # Store the valid answer
                    lead_data[step_key] = normalized_answer
                    session_data["lead_data"] = lead_data
                    await save_user_session(session_id, session_data)
                    
                    logger.info(f"💾 Stored answer for step {current_step_id}: {normalized_answer[:50]}...")
                
                # Find next step in sequence
                next_step_id = current_step_id + 1
                next_step = next((s for s in steps if s["id"] == next_step_id), None)
                
                if next_step:
                    # Advance to next step
                    session_data["fallback_step"] = next_step_id
                    await save_user_session(session_id, session_data)
                    logger.info(f"➡️ Advanced to step {next_step_id} for session {session_id}")
                    return next_step["question"]
                else:
                    # All steps completed - mark as completed and ask for phone
                    session_data["fallback_completed"] = True
                    await save_user_session(session_id, session_data)
                    logger.info(f"✅ Firebase fallback flow completed for session {session_id}")
                    return "Obrigado pelas informações! Para finalizar, preciso do seu número de WhatsApp com DDD (exemplo: 11999999999):"
            
            # If no message provided or we're just starting, return current question
            return current_step["question"]
            
        except Exception as e:
            logger.error(f"❌ Error in STRICT Firebase fallback: {str(e)}")
            # Always fallback to step 1 on error
            return "Qual é o seu nome completo?"

    def _get_validation_message(self, step_id: int) -> str:
        """Get validation message for specific step."""
        validation_messages = {
            1: "Por favor, informe seu nome completo (nome e sobrenome).",
            2: "Por favor, escolha uma das áreas: Penal, Civil, Trabalhista, Família ou Empresarial.",
            3: "Por favor, descreva sua situação com mais detalhes (mínimo 3 caracteres).",
            4: "Por favor, responda com 'Sim' ou 'Não'."
        }
        return validation_messages.get(step_id, "Por favor, forneça uma resposta válida.")

    def _validate_and_normalize_answer(self, answer: str, step_id: int) -> str:
        """Validate and normalize answers for specific steps."""
        answer = answer.strip()
        
        # Step 1: Name validation
        if step_id == 1:
            return answer  # Accept any non-empty name
        
        # Step 2: Area of law - normalize common variations
        elif step_id == 2:
            area_map = {
                "penal": "Penal",
                "criminal": "Penal", 
                "crime": "Penal",
                "civil": "Civil",
                "civel": "Civil",
                "trabalhista": "Trabalhista",
                "trabalho": "Trabalhista",
                "trabalhador": "Trabalhista",
                "família": "Família",
                "familia": "Família",
                "divórcio": "Família",
                "divorcio": "Família",
                "casamento": "Família",
                "empresarial": "Empresarial",
                "empresa": "Empresarial",
                "comercial": "Empresarial",
                "negócio": "Empresarial",
                "negocio": "Empresarial"
            }
            
            answer_lower = answer.lower()
            for keyword, normalized in area_map.items():
                if keyword in answer_lower:
                    return normalized
            
            # If no match found, return original but capitalized
            return answer.title()
        
        # Step 3: Situation description
        elif step_id == 3:
            return answer  # Accept any description
        
        # Step 4: Meeting preference
        elif step_id == 4:
            answer_lower = answer.lower()
            if any(word in answer_lower for word in ["sim", "yes", "quero", "gostaria", "aceito", "ok", "pode", "claro"]):
                return "Sim"
            elif any(word in answer_lower for word in ["não", "nao", "no", "nope", "não quero", "nao quero"]):
                return "Não"
            else:
                return answer  # Return original if unclear
        
        return answer

    def _should_advance_step(self, answer: str, step_id: int) -> bool:
        """Determine if answer is sufficient to advance to next step."""
        answer = answer.strip()
        
        # Reject empty answers
        if len(answer) < 1:
            return False
            
        # Step 1: Name - require at least 2 words (first and last name)
        if step_id == 1:
            words = answer.split()
            return len(words) >= 2 and all(len(word) >= 2 for word in words)
            
        # Step 2: Area - require at least 3 characters
        elif step_id == 2:
            return len(answer) >= 3
            
        # Step 3: Situation - require at least 5 characters for meaningful description
        elif step_id == 3:
            return len(answer) >= 5
            
        # Step 4: Meeting preference - accept any answer
        elif step_id == 4:
            return len(answer) >= 1
        
        # Default: require minimum length
        return len(answer) >= 2

    async def _handle_phone_collection(
        self, 
        phone_message: str, 
        session_id: str, 
        session_data: Dict[str, Any]
    ) -> str:
        """
        Handle phone number collection, validation, and WhatsApp integration.
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
            flow = await self._get_firebase_flow()
            steps = flow.get("steps", [])
            
            for step in sorted(steps, key=lambda x: x.get("id", 0)):
                step_key = f"step_{step['id']}"
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

            # Prepare WhatsApp messages
            user_name = lead_data.get("step_1", "Cliente")
            area = lead_data.get("step_2", "não informada")
            situation = lead_data.get("step_3", "não detalhada")[:100]
            if len(lead_data.get("step_3", "")) > 100:
                situation += "..."

            # Welcome message for user
            welcome_message = f"""Olá {user_name}! 👋

Recebemos sua solicitação através do nosso site.

📝 Área: {area}
📖 Situação: {situation}

Nossa equipe analisará seu caso e entrará em contato em breve. Podemos continuar nossa conversa aqui no WhatsApp.

Como posso ajudá-lo hoje? 🤝"""

            # Internal notification for law firm
            notification_message = f"""🔔 *Nova Lead Capturada (Fallback)*

👤 *Cliente:* {user_name}
📱 *Telefone:* {phone_clean}
🏛️ *Área:* {area}
📝 *Situação:* {situation}
🆔 *Sessão:* {session_id}
⏰ *Data:* {datetime.now().strftime('%d/%m/%Y às %H:%M')}
🤖 *Origem:* Fallback Firebase (Gemini indisponível)

_Lead capturada automaticamente pelo sistema de fallback._"""

            # Send WhatsApp messages
            whatsapp_success = False
            try:
                # Send welcome message to user
                await baileys_service.send_whatsapp_message(whatsapp_number, welcome_message)
                logger.info(f"📤 Welcome message sent to user {phone_formatted}")
                
                # Send notification to law firm
                law_firm_whatsapp = f"55{self.law_firm_number.replace('+', '').replace('-', '')}@s.whatsapp.net"
                await baileys_service.send_whatsapp_message(law_firm_whatsapp, notification_message)
                logger.info(f"📤 Internal notification sent to {self.law_firm_number}")
                
                whatsapp_success = True
                
            except Exception as whatsapp_error:
                logger.error(f"❌ Error sending WhatsApp messages: {str(whatsapp_error)}")
                whatsapp_success = False

            # Get completion message from flow or use default
            flow = await self._get_firebase_flow()
            completion_message = flow.get("completion_message", 
                "Perfeito! Suas informações foram registradas com sucesso. Nossa equipe entrará em contato em breve.")

            # Add phone confirmation to completion message
            final_message = f"""Número confirmado: {phone_clean} 📱

{completion_message}

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
        Main message processing with AI-first approach and STRICT Firebase fallback.
        """
        try:
            logger.info(f"🎯 Processing message - Session: {session_id}, Platform: {platform}")

            session_data = await self._get_or_create_session(session_id, platform, phone_number)

            # Check if we're collecting phone number in fallback mode
            if (session_data.get("fallback_completed") and 
                not session_data.get("phone_submitted") and 
                self._is_phone_number(message)):
                
                phone_response = await self._handle_phone_collection(message, session_id, session_data)
                return {
                    "response_type": "phone_collected_fallback",
                    "platform": platform,
                    "session_id": session_id,
                    "response": phone_response,
                    "phone_submitted": True,
                    "message_count": session_data.get("message_count", 0) + 1
                }

            # Try Gemini first (AI-first approach)
            ai_response = await self._attempt_gemini_response(message, session_id, session_data)
            
            if ai_response:
                # Gemini succeeded - use AI response
                session_data["last_message"] = message
                session_data["last_response"] = ai_response
                session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
                session_data["message_count"] = session_data.get("message_count", 0) + 1
                await save_user_session(session_id, session_data)

                return {
                    "response_type": "ai_intelligent",
                    "platform": platform,
                    "session_id": session_id,
                    "response": ai_response,
                    "ai_mode": True,
                    "gemini_available": True,
                    "message_count": session_data.get("message_count", 1)
                }
            
            # Gemini failed - use STRICT Firebase fallback
            logger.info(f"⚡ Using STRICT Firebase fallback for session {session_id}")
            fallback_response = await self._get_fallback_response(session_data, message)

            # Update session
            session_data["last_message"] = message
            session_data["last_response"] = fallback_response
            session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
            session_data["message_count"] = session_data.get("message_count", 0) + 1
            await save_user_session(session_id, session_data)

            return {
                "response_type": "fallback_firebase",
                "platform": platform,
                "session_id": session_id,
                "response": fallback_response,
                "ai_mode": False,
                "gemini_available": False,
                "fallback_step": session_data.get("fallback_step"),
                "fallback_completed": session_data.get("fallback_completed", False),
                "message_count": session_data.get("message_count", 1)
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
                "gemini_available": session_data.get("gemini_available", True),
                "last_gemini_check": session_data.get("last_gemini_check"),
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