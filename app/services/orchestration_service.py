import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.services.firebase_service import (
    get_user_session,
    save_user_session,
    save_lead_data,
    get_conversation_flow,
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
        self.gemini_timeout = 15.0  # 15 second timeout for Gemini calls
        self.law_firm_number = "+5511918368812"  # Internal notification number

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

    def _validate_and_normalize_answer(self, answer: str, step_id: int) -> str:
        """Validate and normalize answers for specific steps."""
        answer = answer.strip()
        
        # Step 2 is typically area of law - normalize common variations
        if step_id == 2:
            area_map = {
                "penal": "Penal",
                "criminal": "Penal",
                "civil": "Civil",
                "trabalhista": "Trabalhista",
                "trabalho": "Trabalhista",
                "família": "Família",
                "familia": "Família",
                "divórcio": "Família",
                "divorcio": "Família",
                "empresarial": "Empresarial",
                "empresa": "Empresarial",
                "comercial": "Empresarial"
            }
            
            for keyword, normalized in area_map.items():
                if keyword in answer.lower():
                    return normalized
        
        return answer

    def _should_advance_step(self, answer: str, step_id: int) -> bool:
        """Determine if answer is sufficient to advance to next step."""
        answer = answer.strip()
        
        # Reject very short or obviously unrelated answers
        if len(answer) < 2:
            return False
            
        # For name step, require at least two words
        if step_id == 1:
            return len(answer.split()) >= 2
            
        # For other steps, require minimum length
        return len(answer) >= 3

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

    async def _get_fallback_response(
        self, 
        session_data: Dict[str, Any], 
        message: str
    ) -> str:
        """
        Handle fallback conversation flow using Firestore conversation flow.
        This is a deterministic state machine that never skips steps.
        """
        try:
            session_id = session_data["session_id"]
            logger.info(f"⚡ Activating Firebase fallback for session {session_id}")
            
            # Get conversation flow from Firestore
            flow = await get_conversation_flow()
            steps = flow.get("steps", [])
            
            if not steps:
                logger.error("❌ No steps found in conversation flow")
                return "Desculpe, ocorreu um erro interno. Nossa equipe foi notificada."
            
            # Initialize fallback_step if not set
            if session_data.get("fallback_step") is None:
                session_data["fallback_step"] = steps[0]["id"]
                await save_user_session(session_id, session_data)
                logger.info(f"🚀 Initialized fallback at step {steps[0]['id']} for session {session_id}")
            
            current_step_id = session_data["fallback_step"]
            lead_data = session_data.get("lead_data", {})
            
            # Find current step
            current_step = next((s for s in steps if s["id"] == current_step_id), None)
            if not current_step:
                logger.error(f"❌ Step {current_step_id} not found in flow")
                return "Desculpe, ocorreu um erro interno. Nossa equipe foi notificada."
            
            # If we have a message and haven't stored answer for current step yet
            step_key = f"step_{current_step_id}"
            if message and message.strip() and step_key not in lead_data:
                
                # Validate answer
                normalized_answer = self._validate_and_normalize_answer(message, current_step_id)
                
                if not self._should_advance_step(normalized_answer, current_step_id):
                    # Re-prompt same step
                    logger.info(f"🔄 Re-prompting step {current_step_id} - insufficient answer")
                    return current_step["question"]
                
                # Store the answer
                lead_data[step_key] = normalized_answer
                session_data["lead_data"] = lead_data
                
                logger.info(f"💾 Stored answer for step {current_step_id}: {normalized_answer[:50]}...")
                
                # Find next step
                next_step = None
                for i, step in enumerate(steps):
                    if step["id"] == current_step_id and i + 1 < len(steps):
                        next_step = steps[i + 1]
                        break
                
                if next_step:
                    # Advance to next step
                    session_data["fallback_step"] = next_step["id"]
                    await save_user_session(session_id, session_data)
                    logger.info(f"➡️ Advanced to step {next_step['id']} for session {session_id}")
                    return next_step["question"]
                else:
                    # All steps completed - mark as completed and ask for phone
                    session_data["fallback_completed"] = True
                    await save_user_session(session_id, session_data)
                    logger.info(f"✅ Fallback flow completed for session {session_id}")
                    return "Obrigado pelas informações! Para finalizar, preciso do seu número de WhatsApp com DDD (exemplo: 11999999999):"
            
            # If fallback is completed but no phone yet
            elif session_data.get("fallback_completed") and not session_data.get("phone_submitted"):
                if message and self._is_phone_number(message):
                    return await self._handle_phone_collection(message, session_id, session_data)
                else:
                    return "Por favor, informe seu número de WhatsApp com DDD para continuarmos o atendimento."
            
            # Return current step question
            return current_step["question"]
            
        except Exception as e:
            logger.error(f"❌ Error in fallback system: {str(e)}")
            return "Como posso ajudá-lo com questões jurídicas hoje?"

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
            flow = await get_conversation_flow()
            steps = flow.get("steps", [])
            
            for step in steps:
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
            flow = await get_conversation_flow()
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
        Main message processing with AI-first approach and Firebase fallback.
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
            
            # Gemini failed - use Firebase fallback
            logger.info(f"⚡ Using Firebase fallback for session {session_id}")
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