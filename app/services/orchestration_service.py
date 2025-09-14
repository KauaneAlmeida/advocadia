import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.services.firebase_service import (
    get_user_session,
    save_user_session,
    save_lead_data,
    get_fallback_questions,   # ✅ Import para fallback
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
            "message_count": 0
        }

        if phone_number:
            session_data["phone_number"] = phone_number

        return session_data

    def _extract_lead_info(self, message: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        extracted = {}

        # Nome
        if " " in message and not session_data["lead_data"].get("name"):
            extracted["name"] = message.strip().title()

        # Área do Direito
        areas = ["Penal", "Civil", "Trabalhista", "Família", "Empresarial"]
        for area in areas:
            if area.lower() in message.lower() and not session_data["lead_data"].get("area_of_law"):
                extracted["area_of_law"] = area
                break

        # Situação / Problema
        if any(word in message.lower() for word in ["problema", "situação", "caso", "agressão", "divórcio"]):
            if not session_data["lead_data"].get("situation"):
                extracted["situation"] = message

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

            extracted_info = self._extract_lead_info(message, session_data)
            if extracted_info:
                session_data["lead_data"].update(extracted_info)
                await save_user_session(session_id, session_data)
                logger.info(f"📝 Updated lead data: {extracted_info}")

            context = self._prepare_ai_context(session_data, platform)

            ai_response = None
            try:
                ai_response = await ai_orchestrator.generate_response(
                    message,
                    session_id,
                    context=context
                )
            except Exception as e:
                logger.warning(f"⚠️ AI fallback activated (Gemini error): {str(e)}")

            # ------------------------------
            # 🔹 Fallback inteligente
            # ------------------------------
            if not ai_response:
                logger.info("⚡ AI response is empty, trying Firebase fallback")
                fallback_questions = await get_fallback_questions()

                if fallback_questions:
                    ai_response = "No momento estou com instabilidade, mas podemos seguir com estas perguntas:\n\n"
                    for q in fallback_questions:
                        ai_response += f"- {q}\n"
                else:
                    # Fluxo fixo
                    lead_data = session_data.get("lead_data", {})
                    if not lead_data.get("name"):
                        ai_response = "Olá! Qual é o seu nome completo?"
                    elif not lead_data.get("area_of_law"):
                        ai_response = "Ok, obrigado. Em qual área jurídica você precisa de ajuda? (Penal, Civil, Trabalhista, Família ou Empresarial)"
                    elif not lead_data.get("situation"):
                        ai_response = "Entendi. Pode descrever brevemente a sua situação?"
                    else:
                        ai_response = "Perfeito, já tenho suas informações. Agora, poderia informar seu número de WhatsApp para continuarmos por lá?"

            if self._should_save_lead(session_data):
                await self._save_lead_if_ready(session_data)

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
                "ai_mode": bool(ai_response),
                "lead_data": session_data.get("lead_data", {}),
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
        try:
            logger.info(f"📱 Handling phone submission: {phone_number} for session: {session_id}")
            session_data = await get_user_session(session_id) or {}

            # Sanitizar número
            phone_clean = ''.join(filter(str.isdigit, phone_number))

            # Validar se é número BR
            if len(phone_clean) == 10:  # sem nono dígito
                phone_formatted = f"55{phone_clean[:2]}9{phone_clean[2:]}"
            elif len(phone_clean) == 11:  # já tem nono dígito
                phone_formatted = f"55{phone_clean}"
            elif phone_clean.startswith("55"):
                phone_formatted = phone_clean
            else:
                raise ValueError("Número inválido, use DDD + número.")

            whatsapp_number = f"{phone_formatted}@s.whatsapp.net"

            session_data.update({
                "phone_number": phone_clean,
                "phone_formatted": phone_formatted,
                "phone_submitted": True,
                "platform_transition": "web_to_whatsapp"
            })
            await save_user_session(session_id, session_data)

            # 🔹 Monta mensagem inicial
            lead_data = session_data.get("lead_data", {})
            user_name = lead_data.get("name", "Cliente")
            area = lead_data.get("area_of_law", "não informada")
            situation = lead_data.get("situation", "não detalhada")

            whatsapp_message = f"""Olá {user_name}! 👋

Recebemos sua solicitação através do nosso site e estamos aqui para ajudá-lo com questões jurídicas.

📝 Área de interesse: {area}  
📖 Situação: {situation}  

Nossa equipe especializada está pronta para analisar seu caso. Vamos continuar nossa conversa aqui no WhatsApp para maior comodidade.

Como posso ajudá-lo hoje? 🤝"""

            # 🔹 Segundo bloco: resumo
            whatsapp_summary = f"""📁 Resumo do caso enviado pelo cliente:  
- Nome: {user_name}  
- Área: {area}  
- Situação relatada: {situation}"""

            # Enviar via Baileys
            try:
                await baileys_service.send_message(whatsapp_number, whatsapp_message)
                await baileys_service.send_message(whatsapp_number, whatsapp_summary)
                logger.info(f"✅ Mensagens enviadas para {phone_formatted} via WhatsApp")
            except Exception as e:
                logger.error(f"❌ Erro ao enviar mensagem no WhatsApp: {str(e)}")
                return {
                    "status": "error",
                    "message": "Número salvo, mas não foi possível enviar mensagem no WhatsApp.",
                    "error": str(e),
                    "lead_data": lead_data
                }

            return {
                "status": "success",
                "message": f"Mensagens enviadas para {phone_formatted}",
                "lead_data": lead_data
            }

        except Exception as e:
            logger.error(f"❌ Error in handle_phone_number_submission: {str(e)}")
            return {
                "status": "error",
                "message": "Erro ao processar número de WhatsApp",
                "error": str(e)
            }


intelligent_orchestrator = IntelligentHybridOrchestrator()
hybrid_orchestrator = intelligent_orchestrator
