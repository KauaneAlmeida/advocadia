"""
LangChain + Gemini Integration Service

Este módulo integra o LangChain com o Google Gemini para gerenciamento de
conversas inteligentes, memória e geração de respostas contextuais.
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Global conversation memories
conversation_memories: Dict[str, ConversationBufferWindowMemory] = {}


class AIOrchestrator:
    """AI Orchestrator using LangChain + Gemini for intelligent conversation management."""

    def __init__(self):
        self.llm = None
        self.system_prompt = None
        self.chain = None
        self._initialize_llm()
        self._load_system_prompt()
        self._setup_chain()

    def _initialize_llm(self):
        """Initialize Gemini LLM via LangChain."""
        try:
            # Get API key from environment - try both variable names
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.warning("⚠️ GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
                self.llm = None
                return

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                max_tokens=1000,
                timeout=30,
                convert_system_message_to_human=True
            )
            logger.info("✅ LangChain + Gemini LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing LLM: {str(e)}")
            self.llm = None

    def _load_system_prompt(self):
        """Load system prompt from .env, JSON file, or use default."""
        try:
            env_prompt = os.getenv("AI_SYSTEM_PROMPT")
            if env_prompt:
                self.system_prompt = env_prompt
                logger.info("✅ System prompt loaded from environment variable")
                return

            schema_file = "ai_schema.json"
            if os.path.exists(schema_file):
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)
                    self.system_prompt = schema_data.get("system_prompt", "")
                    if self.system_prompt:
                        logger.info("✅ System prompt loaded from ai_schema.json")
                        return

            self.system_prompt = self._get_default_system_prompt()
            logger.info("✅ Using default system prompt")
            
        except Exception as e:
            logger.error(f"❌ Error loading system prompt: {str(e)}")
            self.system_prompt = self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Default system prompt para coleta de informações jurídicas no WhatsApp."""
        return """Você é um assistente virtual de um escritório de advocacia no Brasil. 
Seu papel é apenas **coletar informações básicas do cliente** para que um advogado humano dê continuidade.


## INFORMAÇÕES A COLETAR:
1. Nome completo.
2. Área jurídica (Penal, Civil, Trabalhista, Família, Empresarial).
3. Breve descrição da situação.
4. Número de WhatsApp válido (com DDD).
5. Encerrar agradecendo e avisando que o time jurídico entrará em contato.

## REGRAS IMPORTANTES:
- Sempre responda em português brasileiro.
- Não repita a mesma pergunta da mesma forma** se o cliente não souber responder; reformule de forma natural.
- Nunca ofereça agendamento automático ou horários de consulta.
- Não escreva textos longos: use no máximo 2 frases por resposta.
- Confirme cada informação antes de seguir para a próxima.
- A ordem da coleta é: Nome completo → Área jurídica → Descrição da situação → Número de WhatsApp.
- Peça o número de WhatsApp **somente no final**.
- Use linguagem simples, direta e acolhedora.
- Sempre caminhe para coletar todas as informações, sem pressionar.

## FORMATO DA CONVERSA:
- Seja objetivo e humano, como em uma conversa normal de WhatsApp.
- Sempre finalize cada mensagem com uma pergunta que leve o cliente a responder.
- Se já tiver a resposta de algum item no contexto, não repita a pergunta.


Você **não agenda consultas**, apenas coleta as informações e organiza para o time jurídico."""

    def _setup_chain(self):
        """Create LangChain conversation chain."""
        try:
            if self.llm is None:
                logger.warning("⚠️ Cannot setup chain - LLM not initialized")
                self.chain = None
                return
                
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])

            self.chain = (
                RunnablePassthrough.assign(
                    history=lambda x: self._get_session_history(
                        x.get("session_id", "default")
                    )
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("✅ LangChain conversation chain setup complete")
        except Exception as e:
            logger.error(f"❌ Error setting up chain: {str(e)}")
            self.chain = None

    def _get_session_history(self, session_id: str) -> list:
        """Get session conversation history."""
        if session_id not in conversation_memories:
            conversation_memories[session_id] = ConversationBufferWindowMemory(
                k=10, return_messages=True
            )
        return conversation_memories[session_id].chat_memory.messages

    async def generate_response(
        self, 
        message: str, 
        session_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate AI response using LangChain + Gemini with context."""
        try:
            if self.llm is None:
                raise Exception("LLM not initialized - check API key configuration")
                
            if session_id not in conversation_memories:
                conversation_memories[session_id] = ConversationBufferWindowMemory(
                    k=10, return_messages=True
                )

            memory = conversation_memories[session_id]
            
            contextual_message = message
            if context and isinstance(context, dict):
                context_info = []
                if context.get("name"):
                    context_info.append(f"Nome: {context['name']}")
                if context.get("area_of_law"):
                    context_info.append(f"Área jurídica: {context['area_of_law']}")
                if context.get("situation"):
                    context_info.append(f"Situação: {context['situation']}")
                if context.get("platform"):
                    context_info.append(f"Plataforma: {context['platform']}")
                
                if context_info:
                    contextual_message = f"[Contexto: {'; '.join(context_info)}] {message}"

            # Add timeout and better error handling
            
            try:
                response = await asyncio.wait_for(
                    self.chain.ainvoke({
                        "input": contextual_message, 
                        "session_id": session_id
                    }),
                    timeout=15.0  # 15 second timeout
                )
            except asyncio.TimeoutError:
                logger.error("⏰ Gemini API request timed out")
                raise Exception("API timeout - quota may be exceeded")
            except Exception as api_error:
                # Check for quota/rate limit errors
                error_str = str(api_error).lower()
                if any(indicator in error_str for indicator in ["429", "quota", "rate limit", "resourceexhausted", "billing"]):
                    logger.error(f"🚫 Gemini API quota/rate limit error: {api_error}")
                    raise Exception(f"Quota exceeded: {api_error}")
                else:
                    logger.error(f"❌ Gemini API error: {api_error}")
                    raise api_error

            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response)

            logger.info(f"✅ Generated AI response for session {session_id}")
            return response

        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            # Re-raise the exception so orchestrator can handle it properly
            raise e

    def _get_fallback_response(self) -> str:
        """Fallback response when AI fails."""
        return (
            "Peço desculpas, mas estou enfrentando dificuldades técnicas no momento.\n\n"
            "Para garantir que você receba o melhor atendimento jurídico, recomendo "
            "que entre em contato diretamente com nossa equipe pelo telefone "
            "ou agende uma consulta presencial."
        )

    def clear_session_memory(self, session_id: str):
        """Clear memory for a specific session."""
        if session_id in conversation_memories:
            del conversation_memories[session_id]
            logger.info(f"🧹 Cleared memory for session {session_id}")

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for a session."""
        if session_id not in conversation_memories:
            return {"messages": 0, "summary": "No conversation history"}

        messages = conversation_memories[session_id].chat_memory.messages
        return {
            "messages": len(messages),
            "last_messages": [
                {
                    "type": "human" if isinstance(m, HumanMessage) else "ai",
                    "content": m.content[:100] + ("..." if len(m.content) > 100 else ""),
                }
                for m in messages[-4:]
            ],
        }

    def get_system_prompt(self) -> str:
        """Get current system prompt."""
        return self.system_prompt


# Global AI orchestrator instance
ai_orchestrator = AIOrchestrator()


# Convenience functions for backward compatibility
async def process_chat_message(
    message: str, 
    session_id: str = "default", 
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Process chat message with LangChain + Gemini."""
    return await ai_orchestrator.generate_response(message, session_id, context)


def clear_conversation_memory(session_id: str):
    """Clear conversation memory for session."""
    ai_orchestrator.clear_session_memory(session_id)


def get_conversation_summary(session_id: str) -> Dict[str, Any]:
    """Get conversation summary."""
    return ai_orchestrator.get_conversation_summary(session_id)


async def get_ai_service_status() -> Dict[str, Any]:
    """Get AI service status."""
    try:
        # Quick test without generating a full response to avoid quota usage
        api_key_configured = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        
        if not api_key_configured:
            return {
                "service": "ai_service",
                "status": "configuration_required",
                "error": "API key not configured",
                "api_key_configured": False,
                "configuration_required": True,
            }
        
        # Test LLM initialization without making API calls
        if ai_orchestrator.llm is None:
            return {
                "service": "ai_service",
                "status": "error",
                "error": "LLM not initialized",
                "api_key_configured": api_key_configured,
                "configuration_required": True,
            }

        return {
            "service": "ai_service",
            "status": "active",
            "message": "LangChain + Gemini operational",
            "llm_initialized": True,
            "system_prompt_configured": bool(ai_orchestrator.system_prompt),
            "api_key_configured": api_key_configured,
            "features": [
                "langchain_integration",
                "gemini_api",
                "conversation_memory",
                "session_management",
                "context_awareness",
                "brazilian_portuguese_responses",
            ],
        }
    except Exception as e:
        logger.error(f"❌ Error checking AI service status: {str(e)}")
        return {
            "service": "ai_service",
            "status": "error",
            "error": str(e),
            "configuration_required": True,
            "api_key_configured": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        }


# Alias for compatibility
process_with_langchain = process_chat_message
