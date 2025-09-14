"""
Conversation Flow Service

This module manages the guided conversation flow for law firm client intake.
It handles step-by-step questions, user responses, and transitions to AI chat.

The conversation flow is stored in Firebase and can be updated by lawyers
without modifying the code.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from app.services.firebase_service import (
    get_conversation_flow,
    save_lead_data,
    get_user_session,
    save_user_session,
    update_lead_data
)
from app.services.ai_service import process_chat_message
from app.services.baileys_service import baileys_service

# Configure logging
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Gerencia o fluxo de conversa do chatbot jurídico.
    """

    def __init__(self):
        self.flow_cache = None
        self.cache_timestamp = None

    async def get_flow(self) -> Dict[str, Any]:
        """Pega fluxo de conversa (com cache de 5 min)."""
        if (self.flow_cache is None or
            (datetime.now() - self.cache_timestamp).seconds > 300):
            self.flow_cache = await get_conversation_flow()
            self.cache_timestamp = datetime.now()
        return self.flow_cache

    async def start_conversation(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Inicia conversa nova."""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())

            flow = await self.get_flow()

            session_data = {
                "session_id": session_id,
                "current_step": 1,
                "responses": {},
                "flow_completed": False,
                "ai_mode": False,
                "phone_collected": False,
                "started_at": datetime.now(),
                "last_updated": datetime.now()
            }

            await save_user_session(session_id, session_data)

            first_step = next((s for s in flow["steps"] if s["id"] == 1), None)
            if not first_step:
                raise ValueError("Nenhum primeiro passo encontrado no fluxo")

            logger.info(f"✅ Conversa iniciada: {session_id}")

            return {
                "session_id": session_id,
                "question": first_step["question"],
                "step_id": first_step["id"],
                "is_final_step": len(flow["steps"]) == 1,
                "flow_completed": False,
                "ai_mode": False,
                "phone_collected": False
            }

        except Exception as e:
            logger.error(f"❌ Erro ao iniciar conversa: {str(e)}")
            raise

    async def process_response(self, session_id: str, user_response: str) -> Dict[str, Any]:
        """Processa resposta do usuário e devolve próxima pergunta ou resposta da IA."""
        try:
            session_data = await get_user_session(session_id)
            if not session_data:
                return await self.start_conversation(session_id)

            # Se terminou o fluxo mas ainda não pegou telefone
            if session_data.get("flow_completed") and not session_data.get("phone_collected"):
                return await self._handle_phone_collection(session_id, session_data, user_response)

            # Se já está no modo IA
            if session_data.get("ai_mode"):
                ai_response = await process_chat_message(user_response, session_id=session_id)
                return {
                    "session_id": session_id,
                    "response": ai_response,
                    "ai_mode": True,
                    "flow_completed": True,
                    "phone_collected": session_data.get("phone_collected", False)
                }

            flow = await self.get_flow()
            current_step = session_data.get("current_step", 1)
            current_step_data = next((s for s in flow["steps"] if s["id"] == current_step), None)

            if not current_step_data:
                return await self._switch_to_ai_mode(session_id, user_response)

            # Salva resposta
            field_name = current_step_data.get("field", f"step_{current_step}")
            session_data["responses"][field_name] = user_response.strip()
            session_data["last_updated"] = datetime.now()

            # Próximo passo
            next_step = current_step + 1
            next_step_data = next((s for s in flow["steps"] if s["id"] == next_step), None)

            if next_step_data:
                session_data["current_step"] = next_step
                await save_user_session(session_id, session_data)

                return {
                    "session_id": session_id,
                    "question": next_step_data["question"],
                    "step_id": next_step_data["id"],
                    "is_final_step": next_step == len(flow["steps"]),
                    "flow_completed": False,
                    "ai_mode": False,
                    "phone_collected": False
                }
            else:
                return await self._complete_flow(session_id, session_data, flow)

        except Exception as e:
            logger.error(f"❌ Erro processando resposta ({session_id}): {str(e)}")
            return await self._switch_to_ai_mode(session_id, user_response)

    async def _complete_flow(self, session_id: str, session_data: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, Any]:
        """Finaliza coleta de dados e pede telefone."""
        try:
            responses = session_data.get("responses", {})
            lead_data = {
                "name": responses.get("name", "Desconhecido"),
                "area_of_law": responses.get("area_of_law", "Não informado"),
                "situation": responses.get("situation", "Não informado"),
                "wants_meeting": responses.get("wants_meeting", "Não informado"),
                "session_id": session_id,
                "completed_at": datetime.now(),
                "status": "intake_completed"
            }

            lead_id = await save_lead_data(lead_data)

            session_data.update({
                "flow_completed": True,
                "ai_mode": False,
                "phone_collected": False,
                "lead_id": lead_id,
                "completed_at": datetime.now(),
                "last_updated": datetime.now()
            })

            await save_user_session(session_id, session_data)

            phone_message = "Obrigado pelas informações 🙏 Agora, me passe seu número com DDD (ex: 11999999999):"

            return {
                "session_id": session_id,
                "question": phone_message,
                "flow_completed": True,
                "ai_mode": False,
                "phone_collected": False,
                "lead_saved": True,
                "lead_id": lead_id,
                "collecting_phone": True
            }

        except Exception as e:
            logger.error(f"❌ Erro finalizando fluxo: {str(e)}")
            return await self._switch_to_ai_mode(session_id, "Obrigado pelas informações!")

    async def _handle_phone_collection(self, session_id: str, session_data: Dict[str, Any], user_response: str) -> Dict[str, Any]:
        """Coleta e valida número de telefone."""
        try:
            phone_clean = ''.join(filter(str.isdigit, user_response))

            if len(phone_clean) < 10 or len(phone_clean) > 11:
                return {
                    "session_id": session_id,
                    "question": "Número inválido 😕 Digite no formato com DDD (ex: 11999999999):",
                    "flow_completed": True,
                    "ai_mode": False,
                    "phone_collected": False,
                    "collecting_phone": True,
                    "validation_error": True
                }

            if len(phone_clean) == 10:
                phone_formatted = f"55{phone_clean[:2]}9{phone_clean[2:]}"
            else:
                phone_formatted = f"55{phone_clean}"

            session_data.update({
                "phone_collected": True,
                "ai_mode": True,
                "phone_number": phone_clean,
                "phone_formatted": phone_formatted,
                "last_updated": datetime.now()
            })

            await save_user_session(session_id, session_data)

            # Atualiza lead com telefone
            await update_lead_data(session_data.get("lead_id"), {
                "phone_number": phone_clean,
                "phone_formatted": phone_formatted,
                "status": "phone_collected",
                "updated_at": datetime.now()
            })

            # Mensagem resumo
            responses = session_data.get("responses", {})
            user_name = responses.get("name", "Cliente")
            whatsapp_message = (
                f"Olá {user_name}! 👋\n\n"
                f"Recebemos suas informações e nossa equipe vai entrar em contato.\n\n"
                f"📋 Área: {responses.get('area_of_law', 'Não informada')}\n"
                f"📝 Situação: {responses.get('situation', 'Não informada')[:80]}..."
            )

            whatsapp_success = False
            try:
                whatsapp_success = await baileys_service.send_whatsapp_message(
                    f"{phone_formatted}@s.whatsapp.net", whatsapp_message
                )
            except Exception as err:
                logger.error(f"❌ Erro enviando mensagem no WhatsApp: {str(err)}")

            confirmation_message = (
                f"Perfeito! Número confirmado: {phone_clean} 📱\n\n"
                f"✅ Suas informações foram registradas.\n"
                f"👨‍💼 Nossa equipe entrará em contato em breve."
            )

            return {
                "session_id": session_id,
                "response": confirmation_message,
                "flow_completed": True,
                "ai_mode": True,
                "phone_collected": True,
                "whatsapp_sent": whatsapp_success,
                "phone_number": phone_clean
            }

        except Exception as e:
            logger.error(f"❌ Erro na coleta do telefone: {str(e)}")
            return {
                "session_id": session_id,
                "response": "Erro ao processar seu número. Vamos continuar! Como posso te ajudar?",
                "flow_completed": True,
                "ai_mode": True,
                "phone_collected": False,
                "error": str(e)
            }

    async def _switch_to_ai_mode(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Troca para modo IA."""
        try:
            session_data = await get_user_session(session_id) or {}
            session_data.update({
                "ai_mode": True,
                "flow_completed": True,
                "phone_collected": False,  # não marcar como coletado sem ter número
                "switched_to_ai_at": datetime.now(),
                "last_updated": datetime.now()
            })
            await save_user_session(session_id, session_data)

            ai_response = await process_chat_message(user_message, session_id=session_id)

            return {
                "session_id": session_id,
                "response": ai_response,
                "ai_mode": True,
                "flow_completed": True,
                "phone_collected": False
            }

        except Exception as e:
            logger.error(f"❌ Erro trocando para modo IA: {str(e)}")
            return {
                "session_id": session_id,
                "response": "Estou aqui para te ajudar com questões jurídicas. Como posso te auxiliar?",
                "ai_mode": True,
                "flow_completed": True,
                "phone_collected": False
            }

    async def get_conversation_status(self, session_id: str) -> Dict[str, Any]:
        """Retorna status atual da conversa."""
        try:
            session_data = await get_user_session(session_id)
            if not session_data:
                return {"exists": False}

            flow = await self.get_flow()
            current_step = session_data.get("current_step", 1)

            return {
                "exists": True,
                "session_id": session_id,
                "current_step": current_step,
                "total_steps": len(flow["steps"]),
                "flow_completed": session_data.get("flow_completed", False),
                "ai_mode": session_data.get("ai_mode", False),
                "phone_collected": session_data.get("phone_collected", False),
                "responses_collected": len(session_data.get("responses", {})),
                "started_at": session_data.get("started_at"),
                "last_updated": session_data.get("last_updated")
            }

        except Exception as e:
            logger.error(f"❌ Erro ao pegar status da conversa: {str(e)}")
            return {"exists": False, "error": str(e)}

# Instância global
conversation_manager = ConversationManager()
