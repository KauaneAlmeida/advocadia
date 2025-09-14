import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.services.firebase_service import (
    get_user_session,
    save_user_session,
    save_lead_data,
    get_fallback_questions,
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
        self.lead_fields = ["name", "area_of_law", "situation"]
        self.gemini_available = True
        self.last_gemini_check = None

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
            "fallback_step": 1,  # Track current step in fallback flow
            "phone_submitted": False
        }

        if phone_number:
            session_data["phone_number"] = phone_number

        return session_data

    def _extract_lead_info(self, message: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        extracted = {}

        # Nome - look for full names (first + last name)
        if " " in message.strip() and len(message.strip().split()) >= 2 and not session_data["lead_data"].get("name"):
            # Check if it looks like a name (not a question or description)
            if not any(word in message.lower() for word in ["como", "onde", "quando", "porque", "preciso", "quero", "tenho"]):
                extracted["name"] = message.strip().title()

        # Área do Direito - more comprehensive detection
        areas_map = {
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
        
        for keyword, area in areas_map.items():
            if keyword in message.lower() and not session_data["lead_data"].get("area_of_law"):
                extracted["area_of_law"] = area
                break

        # Situação - capture longer descriptions
        if len(message.strip()) > 10 and not session_data["lead_data"].get("situation"):
            # Check if it's a description (not a name or simple answer)
            if any(word in message.lower() for word in ["problema", "situação", "caso", "preciso", "tenho", "aconteceu", "quero"]):
                extracted["situation"] = message.strip()

        return extracted

    def _prepare_ai_context(self, session_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        lead_data = session_data.get("lead_data", {})
        return {
            "platform": platform,
            "name": lead_data.get("name", "Não informado"),
            "area_of_law": lead_data.get("area_of_law", "Não informada"),
            "situation": lead_data.get("situation", "Não detalhada")
        }

    def _should_save_lead(self, session_data: Dict[str, Any]) -> bool:
        lead_data = session_data.get("lead_data", {})
        return all(lead_data.get(field) for field in self.lead_fields)

    async def _save_lead_if_ready(self, session_data: Dict[str, Any]) -> None:
        """
        Converte o lead_data para o formato 'answers' do Firestore e salva.
        """
        lead_data = session_data.get("lead_data", {})
        if self._should_save_lead(session_data):
            answers = []
            if "name" in lead_data:
                answers.append({"id": 1, "answer": lead_data["name"]})
            if "area_of_law" in lead_data:
                answers.append({"id": 2, "answer": lead_data["area_of_law"]})
            if "situation" in lead_data:
                answers.append({"id": 3, "answer": lead_data["situation"]})
            if "wants_meeting" in lead_data:
                answers.append({"id": 4, "answer": lead_data["wants_meeting"]})

            await save_lead_data({"answers": answers})
            logger.info(f"💾 Lead salvo no Firestore: {answers}")

    def _is_quota_error(self, error_message: str) -> bool:
        """Check if error is related to API quota/rate limits."""
        quota_indicators = [
            "429", "quota", "rate limit", "exceeded", "ResourceExhausted",
            "billing", "plan", "free tier", "requests per day"
        ]
        return any(indicator.lower() in str(error_message).lower() for indicator in quota_indicators)

    def _is_phone_number(self, message: str) -> bool:
        """Check if message looks like a phone number."""
        clean_message = ''.join(filter(str.isdigit, message))
        return len(clean_message) >= 10 and len(clean_message) <= 13

    async def process_message(
        self,
        message: str,
        session_id: str,
        phone_number: Optional[str] = None,
        platform: str = "web"
    ) -> Dict[str, Any]:
        try:
            logger.info(f"🎯 Processing message - Session: {session_id}, Platform: {platform}")

            session_data = await self._get_or_create_session(session_id, platform, phone_number)

            # Check if we're collecting phone number
            if (self._should_save_lead(session_data) and 
                not session_data.get("phone_submitted") and 
                self._is_phone_number(message)):
                return await self._handle_phone_collection(message, session_id, session_data)

            # Extract lead information from message
            extracted_info = self._extract_lead_info(message, session_data)
            if extracted_info:
                session_data["lead_data"].update(extracted_info)
                await save_user_session(session_id, session_data)
                logger.info(f"📝 Updated lead data: {extracted_info}")

            ai_response = None
            
            # Only try Gemini if we think it's available
            if self.gemini_available:
                try:
                    logger.info("🤖 Attempting Gemini AI response...")
                    context = self._prepare_ai_context(session_data, platform)
                    
                    import asyncio
                    gemini_response = await asyncio.wait_for(
                        ai_orchestrator.generate_response(
                            message,
                            session_id,
                            context=context
                        ),
                        timeout=15.0
                    )
                    
                    # Check if Gemini response is valid
                    if (gemini_response and 
                        isinstance(gemini_response, str) and 
                        gemini_response.strip() and
                        not self._is_quota_error(gemini_response)):
                        ai_response = gemini_response
                        logger.info("✅ Valid Gemini response received")
                    else:
                        logger.warning(f"⚠️ Invalid Gemini response detected")
                        ai_response = None
                        
                except Exception as e:
                    logger.warning(f"⚠️ Gemini AI failed: {str(e)}")
                    
                    # Check if it's a quota error and disable Gemini temporarily
                    if self._is_quota_error(str(e)):
                        logger.error("🚫 Gemini API quota exceeded - switching to fallback mode")
                        self.gemini_available = False
                        self.last_gemini_check = datetime.now(timezone.utc)
                    
                    ai_response = None
            else:
                logger.info("⚠️ Gemini API unavailable - using fallback directly")

            # Use fallback system when Gemini is unavailable
            if not ai_response:
                logger.info("⚡ Activating fallback system...")
                ai_response = await self._get_fallback_response(session_data, message)

            # Ensure we always have a response
            if not ai_response or not ai_response.strip():
                ai_response = "Olá! Como posso ajudá-lo com questões jurídicas hoje?"
                logger.warning("🚨 Using emergency fallback response")

            # Save lead if ready
            if self._should_save_lead(session_data):
                await self._save_lead_if_ready(session_data)

            # Update session
            session_data["last_message"] = message
            session_data["last_response"] = ai_response
            session_data["last_updated"] = ensure_utc(datetime.now(timezone.utc))
            session_data["message_count"] = session_data.get("message_count", 0) + 1
            await save_user_session(session_id, session_data)

            return {
                "response_type": "ai_intelligent" if self.gemini_available else "fallback_intelligent",
                "platform": platform,
                "session_id": session_id,
                "response": ai_response,
                "ai_mode": bool(ai_response),
                "lead_data": session_data.get("lead_data", {}),
                "message_count": session_data.get("message_count", 1),
                "gemini_available": self.gemini_available
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

    async def _get_fallback_response(self, session_data: Dict[str, Any], message: str) -> str:
        """
        Clean fallback system without AI status messages.
        Provides natural conversation flow for lead collection.
        """
        try:
            lead_data = session_data.get("lead_data", {})
            fallback_step = session_data.get("fallback_step", 1)
            
            # Step 1: Collect Name
            if not lead_data.get("name"):
                if fallback_step == 1:
                    session_data["fallback_step"] = 1
                    await save_user_session(session_data["session_id"], session_data)
                    return "Olá! Para começar, qual é o seu nome completo?"
                else:
                    return "Por favor, me informe seu nome completo para continuarmos."
            
            # Step 2: Collect Area of Law
            elif not lead_data.get("area_of_law"):
                name = lead_data.get("name", "").split()[0]  # First name only
                if fallback_step <= 2:
                    session_data["fallback_step"] = 2
                    await save_user_session(session_data["session_id"], session_data)
                    return f"Obrigado, {name}! Em qual área jurídica você precisa de ajuda?\n\n• Penal\n• Civil\n• Trabalhista\n• Família\n• Empresarial"
                else:
                    return "Qual área do direito se relaciona com sua situação? (Penal, Civil, Trabalhista, Família ou Empresarial)"
            
            # Step 3: Collect Situation Description
            elif not lead_data.get("situation"):
                if fallback_step <= 3:
                    session_data["fallback_step"] = 3
                    await save_user_session(session_data["session_id"], session_data)
                    return "Perfeito! Agora, pode descrever brevemente sua situação ou problema jurídico?"
                else:
                    return "Por favor, conte-me um pouco sobre sua situação para que possamos ajudá-lo melhor."
            
            # Step 4: Collect Phone Number
            elif not session_data.get("phone_submitted"):
                if fallback_step <= 4:
                    session_data["fallback_step"] = 4
                    await save_user_session(session_data["session_id"], session_data)
                    return "Obrigado pelas informações! Para finalizar, preciso do seu número de WhatsApp com DDD (exemplo: 11999999999):"
                else:
                    return "Por favor, informe seu número de WhatsApp com DDD para continuarmos o atendimento."
            
            # All information collected
            else:
                name = lead_data.get("name", "").split()[0]
                return f"Perfeito, {name}! Suas informações foram registradas. Nossa equipe especializada analisará seu caso e entrará em contato em breve. Há mais alguma coisa que gostaria de mencionar?"

        except Exception as e:
            logger.error(f"❌ Error in fallback system: {str(e)}")
            return "Como posso ajudá-lo com questões jurídicas hoje?"

    async def _handle_phone_collection(self, phone_message: str, session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle phone number collection and WhatsApp integration.
        """
        try:
            # Clean and validate phone number
            phone_clean = ''.join(filter(str.isdigit, phone_message))
            
            # Validate Brazilian phone number format
            if len(phone_clean) < 10 or len(phone_clean) > 13:
                return {
                    "response_type": "validation_error",
                    "session_id": session_id,
                    "response": "Número inválido. Por favor, digite no formato com DDD (exemplo: 11999999999):",
                    "lead_data": session_data.get("lead_data", {}),
                    "message_count": session_data.get("message_count", 1)
                }

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
                "fallback_step": 5,
                "last_updated": ensure_utc(datetime.now(timezone.utc))
            })
            await save_user_session(session_id, session_data)

            # Save final lead data
            await self._save_lead_if_ready(session_data)

            # Prepare WhatsApp messages
            lead_data = session_data.get("lead_data", {})
            user_name = lead_data.get("name", "Cliente")
            area = lead_data.get("area_of_law", "não informada")
            situation = lead_data.get("situation", "não detalhada")[:100] + ("..." if len(lead_data.get("situation", "")) > 100 else "")

            # Welcome message for user
            welcome_message = f"""Olá {user_name}! 👋

Recebemos sua solicitação através do nosso site e estamos aqui para ajudá-lo.

📝 Área de interesse: {area}
📖 Situação: {situation}

Nossa equipe especializada analisará seu caso e entrará em contato em breve. Vamos continuar nossa conversa aqui no WhatsApp para maior comodidade.

Como posso ajudá-lo hoje? 🤝"""

            # Internal notification for law firm
            notification_message = f"""🔔 *Nova Lead Capturada*

👤 *Cliente:* {user_name}
📱 *Telefone:* {phone_clean}
🏛️ *Área:* {area}
📝 *Situação:* {situation}
🆔 *Sessão:* {session_id}
⏰ *Data:* {datetime.now().strftime('%d/%m/%Y às %H:%M')}

_Lead capturada automaticamente pelo sistema._"""

            # Send WhatsApp messages
            whatsapp_success = False
            try:
                # Send welcome message to user
                await baileys_service.send_whatsapp_message(whatsapp_number, welcome_message)
                
                # Send notification to law firm (using configured phone number)
                law_firm_number = "+5511918368812"  # From your config
                await baileys_service.send_whatsapp_message(f"55{law_firm_number.replace('+', '').replace('-', '')}@s.whatsapp.net", notification_message)
                
                whatsapp_success = True
                logger.info(f"✅ WhatsApp messages sent successfully to {phone_formatted}")
                
            except Exception as whatsapp_error:
                logger.error(f"❌ Error sending WhatsApp messages: {str(whatsapp_error)}")
                whatsapp_success = False

            # Confirmation response
            confirmation_response = f"""Perfeito! Número confirmado: {phone_clean} 📱

✅ Suas informações foram registradas com sucesso.
📞 Nossa equipe entrará em contato em breve.
💬 Você também receberá uma mensagem no WhatsApp para continuarmos o atendimento.

Obrigado por escolher nossos serviços jurídicos!"""

            return {
                "response_type": "phone_collected",
                "session_id": session_id,
                "response": confirmation_response,
                "lead_data": lead_data,
                "phone_number": phone_clean,
                "whatsapp_sent": whatsapp_success,
                "phone_submitted": True,
                "message_count": session_data.get("message_count", 1) + 1
            }

        except Exception as e:
            logger.error(f"❌ Error handling phone collection: {str(e)}")
            return {
                "response_type": "error",
                "session_id": session_id,
                "response": "Ocorreu um erro ao processar seu número. Por favor, tente novamente ou entre em contato conosco diretamente.",
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
            return await self._handle_phone_collection(phone_number, session_id, session_data)
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

            lead_data = session_data.get("lead_data", {})
            return {
                "exists": True,
                "session_id": session_id,
                "platform": session_data.get("platform", "unknown"),
                "fallback_step": session_data.get("fallback_step", 1),
                "phone_submitted": session_data.get("phone_submitted", False),
                "lead_data": lead_data,
                "lead_complete": self._should_save_lead(session_data),
                "message_count": session_data.get("message_count", 0),
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated"),
                "gemini_available": self.gemini_available
            }
        except Exception as e:
            logger.error(f"❌ Error getting session context: {str(e)}")
            return {"exists": False, "error": str(e)}


# Global instance
intelligent_orchestrator = IntelligentHybridOrchestrator()
hybrid_orchestrator = intelligent_orchestrator