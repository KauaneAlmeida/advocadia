"""
Integration tests for Firebase fallback system.

These tests verify the fallback behavior when Gemini AI is unavailable.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.orchestration_service import intelligent_orchestrator
from app.services.firebase_service import save_user_session, get_user_session


class TestFirebaseFallbackIntegration:
    """Test the complete Firebase fallback flow."""

    @pytest.fixture
    def mock_conversation_flow(self):
        """Mock conversation flow from Firestore."""
        return {
            "steps": [
                {"id": 1, "question": "Qual é o seu nome completo?"},
                {"id": 2, "question": "Em qual área do direito você precisa de ajuda?"},
                {"id": 3, "question": "Descreva brevemente sua situação."},
                {"id": 4, "question": "Gostaria de agendar uma consulta?"}
            ],
            "completion_message": "Obrigado! Suas informações foram registradas."
        }

    @pytest.fixture
    def mock_baileys_service(self):
        """Mock Baileys WhatsApp service."""
        with patch('app.services.orchestration_service.baileys_service') as mock:
            mock.send_whatsapp_message = AsyncMock(return_value=True)
            yield mock

    @pytest.fixture
    def mock_firebase_services(self, mock_conversation_flow):
        """Mock Firebase services."""
        with patch('app.services.orchestration_service.get_conversation_flow') as mock_flow, \
             patch('app.services.orchestration_service.save_user_session') as mock_save, \
             patch('app.services.orchestration_service.get_user_session') as mock_get, \
             patch('app.services.orchestration_service.save_lead_data') as mock_lead:
            
            mock_flow.return_value = mock_conversation_flow
            mock_save.return_value = True
            mock_get.return_value = None
            mock_lead.return_value = "lead_123"
            
            yield {
                'flow': mock_flow,
                'save': mock_save,
                'get': mock_get,
                'lead': mock_lead
            }

    @pytest.fixture
    def mock_gemini_failure(self):
        """Mock Gemini AI to always fail with quota error."""
        with patch('app.services.orchestration_service.ai_orchestrator') as mock:
            mock.generate_response = AsyncMock(
                side_effect=Exception("429 Quota exceeded")
            )
            yield mock

    @pytest.mark.asyncio
    async def test_gemini_available_normal_flow(self, mock_firebase_services):
        """Test normal AI flow when Gemini is available."""
        with patch('app.services.orchestration_service.ai_orchestrator') as mock_ai:
            mock_ai.generate_response = AsyncMock(
                return_value="Olá! Como posso ajudá-lo com questões jurídicas?"
            )
            
            result = await intelligent_orchestrator.process_message(
                "Olá", "test_session", platform="web"
            )
            
            assert result["response_type"] == "ai_intelligent"
            assert result["ai_mode"] is True
            assert result["gemini_available"] is True
            assert "Como posso ajudá-lo" in result["response"]

    @pytest.mark.asyncio
    async def test_fallback_activation_on_gemini_failure(
        self, 
        mock_firebase_services, 
        mock_gemini_failure,
        mock_baileys_service
    ):
        """Test fallback activation when Gemini fails."""
        result = await intelligent_orchestrator.process_message(
            "Olá", "test_session", platform="web"
        )
        
        assert result["response_type"] == "fallback_firebase"
        assert result["ai_mode"] is False
        assert result["gemini_available"] is False
        assert result["response"] == "Qual é o seu nome completo?"

    @pytest.mark.asyncio
    async def test_complete_fallback_flow(
        self, 
        mock_firebase_services, 
        mock_gemini_failure,
        mock_baileys_service
    ):
        """Test complete fallback conversation flow."""
        session_id = "test_complete_flow"
        
        # Step 1: Initial message triggers fallback
        result1 = await intelligent_orchestrator.process_message(
            "Olá", session_id, platform="web"
        )
        assert result1["response"] == "Qual é o seu nome completo?"
        
        # Step 2: Provide name
        result2 = await intelligent_orchestrator.process_message(
            "João Silva", session_id, platform="web"
        )
        assert result2["response"] == "Em qual área do direito você precisa de ajuda?"
        
        # Step 3: Provide area
        result3 = await intelligent_orchestrator.process_message(
            "Penal", session_id, platform="web"
        )
        assert result3["response"] == "Descreva brevemente sua situação."
        
        # Step 4: Provide situation
        result4 = await intelligent_orchestrator.process_message(
            "Preciso de ajuda com um processo criminal", session_id, platform="web"
        )
        assert result4["response"] == "Gostaria de agendar uma consulta?"
        
        # Step 5: Final answer
        result5 = await intelligent_orchestrator.process_message(
            "Sim", session_id, platform="web"
        )
        assert "WhatsApp" in result5["response"]
        assert result5["fallback_completed"] is True

    @pytest.mark.asyncio
    async def test_phone_collection_and_whatsapp_integration(
        self, 
        mock_firebase_services, 
        mock_gemini_failure,
        mock_baileys_service
    ):
        """Test phone number collection and WhatsApp message sending."""
        session_id = "test_phone_collection"
        
        # Complete the flow first (simulate completed state)
        mock_firebase_services['get'].return_value = {
            "session_id": session_id,
            "fallback_completed": True,
            "phone_submitted": False,
            "lead_data": {
                "step_1": "João Silva",
                "step_2": "Penal", 
                "step_3": "Processo criminal",
                "step_4": "Sim"
            },
            "gemini_available": False
        }
        
        # Submit phone number
        result = await intelligent_orchestrator.process_message(
            "11999999999", session_id, platform="web"
        )
        
        assert result["response_type"] == "phone_collected_fallback"
        assert result["phone_submitted"] is True
        assert "confirmado" in result["response"]
        
        # Verify WhatsApp messages were sent
        assert mock_baileys_service.send_whatsapp_message.call_count == 2
        
        # Check user message
        user_call = mock_baileys_service.send_whatsapp_message.call_args_list[0]
        assert "5511999999999@s.whatsapp.net" in user_call[0][0]
        assert "João Silva" in user_call[0][1]
        
        # Check internal notification
        internal_call = mock_baileys_service.send_whatsapp_message.call_args_list[1]
        assert "5511918368812@s.whatsapp.net" in internal_call[0][0]
        assert "Nova Lead Capturada" in internal_call[0][1]

    @pytest.mark.asyncio
    async def test_gemini_recovery(
        self, 
        mock_firebase_services,
        mock_baileys_service
    ):
        """Test Gemini recovery after being marked unavailable."""
        session_id = "test_recovery"
        
        # Start with Gemini unavailable
        mock_firebase_services['get'].return_value = {
            "session_id": session_id,
            "gemini_available": False,
            "fallback_step": 1,
            "lead_data": {}
        }
        
        # Mock Gemini to succeed this time
        with patch('app.services.orchestration_service.ai_orchestrator') as mock_ai:
            mock_ai.generate_response = AsyncMock(
                return_value="Olá! Como posso ajudá-lo?"
            )
            
            result = await intelligent_orchestrator.process_message(
                "Olá novamente", session_id, platform="web"
            )
            
            assert result["response_type"] == "ai_intelligent"
            assert result["gemini_available"] is True
            assert "Como posso ajudá-lo" in result["response"]

    @pytest.mark.asyncio
    async def test_answer_validation_and_reprompting(
        self, 
        mock_firebase_services, 
        mock_gemini_failure,
        mock_baileys_service
    ):
        """Test answer validation and re-prompting for insufficient answers."""
        session_id = "test_validation"
        
        # Start fallback
        result1 = await intelligent_orchestrator.process_message(
            "Olá", session_id, platform="web"
        )
        assert result1["response"] == "Qual é o seu nome completo?"
        
        # Provide insufficient answer (too short)
        result2 = await intelligent_orchestrator.process_message(
            "João", session_id, platform="web"
        )
        # Should re-prompt same question
        assert result2["response"] == "Qual é o seu nome completo?"
        
        # Provide sufficient answer
        result3 = await intelligent_orchestrator.process_message(
            "João Silva Santos", session_id, platform="web"
        )
        # Should advance to next step
        assert result3["response"] == "Em qual área do direito você precisa de ajuda?"

    @pytest.mark.asyncio
    async def test_area_normalization(
        self, 
        mock_firebase_services, 
        mock_gemini_failure,
        mock_baileys_service
    ):
        """Test area of law answer normalization."""
        session_id = "test_normalization"
        
        # Get to step 2 (area question)
        await intelligent_orchestrator.process_message("Olá", session_id)
        await intelligent_orchestrator.process_message("João Silva", session_id)
        
        # Test normalization of "criminal" to "Penal"
        result = await intelligent_orchestrator.process_message(
            "criminal", session_id, platform="web"
        )
        
        # Should advance to next step
        assert result["response"] == "Descreva brevemente sua situação."
        
        # Verify normalized answer was stored
        # This would be verified by checking the session data in a real test

    def test_phone_number_validation(self):
        """Test phone number validation logic."""
        orchestrator = intelligent_orchestrator
        
        # Valid Brazilian numbers
        assert orchestrator._is_phone_number("11999999999")
        assert orchestrator._is_phone_number("5511999999999")
        assert orchestrator._is_phone_number("(11) 99999-9999")
        
        # Invalid numbers
        assert not orchestrator._is_phone_number("123")
        assert not orchestrator._is_phone_number("abc")
        assert not orchestrator._is_phone_number("11999")

    def test_quota_error_detection(self):
        """Test quota error detection logic."""
        orchestrator = intelligent_orchestrator
        
        # Should detect quota errors
        assert orchestrator._is_quota_error("429 Too Many Requests")
        assert orchestrator._is_quota_error("Quota exceeded")
        assert orchestrator._is_quota_error("ResourceExhausted")
        assert orchestrator._is_quota_error("billing issue")
        
        # Should not detect regular errors
        assert not orchestrator._is_quota_error("Network error")
        assert not orchestrator._is_quota_error("Invalid request")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])