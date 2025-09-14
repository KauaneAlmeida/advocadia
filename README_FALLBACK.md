# Firebase Fallback System for Gemini AI Unavailability

This document describes the Firebase-driven fallback system that activates when Gemini AI is unavailable due to quota limits, timeouts, or other errors.

## Overview

The system maintains an **AI-first approach** - always attempting Gemini AI first, but seamlessly falling back to a deterministic Firebase-driven conversation flow when AI is unavailable. The fallback reads structured conversation flows from Firestore and operates as a state machine that never skips steps.

## Key Features

- **AI-First Behavior**: Always attempts Gemini first with 15-second timeout
- **Automatic Fallback**: Switches to Firebase flow only when Gemini fails
- **Deterministic State Machine**: Never skips steps, follows Firestore flow exactly
- **Automatic Recovery**: Resumes AI mode when Gemini becomes available again
- **WhatsApp Integration**: Sends messages via existing Baileys service
- **Lead Collection**: Saves structured lead data to Firestore

## Architecture

### Flow Decision Logic

```
Incoming Message
       ↓
Try Gemini AI (15s timeout)
       ↓
   Success? ──Yes──→ Use AI Response
       ↓
      No
       ↓
Mark Gemini Unavailable
       ↓
Use Firebase Fallback Flow
       ↓
Follow Firestore Steps 1→2→3→4→Phone
       ↓
Send WhatsApp Messages
       ↓
Complete & Save Lead
```

### Data Structure

#### Firestore Collections

**`conversation_flows/law_firm_intake`**
```json
{
  "steps": [
    {"id": 1, "question": "Qual é o seu nome completo?"},
    {"id": 2, "question": "Em qual área do direito você precisa de ajuda?"},
    {"id": 3, "question": "Descreva brevemente sua situação."},
    {"id": 4, "question": "Gostaria de agendar uma consulta?"}
  ],
  "completion_message": "Obrigado! Suas informações foram registradas."
}
```

**`user_sessions/<session_id>`**
```json
{
  "session_id": "web_123456",
  "platform": "web",
  "fallback_step": 2,
  "fallback_completed": false,
  "phone_submitted": false,
  "gemini_available": false,
  "last_gemini_check": "2024-01-15T10:30:00Z",
  "lead_data": {
    "step_1": "João Silva",
    "step_2": "Penal",
    "phone": "11999999999"
  },
  "message_count": 5
}
```

**`leads` collection**
```json
{
  "answers": [
    {"id": 1, "answer": "João Silva"},
    {"id": 2, "answer": "Penal"},
    {"id": 3, "answer": "Processo criminal"},
    {"id": 4, "answer": "Sim"},
    {"id": 5, "answer": "11999999999"}
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "new",
  "source": "chatbot_fallback"
}
```

## Implementation Details

### Core Components

1. **`IntelligentHybridOrchestrator`** (`app/services/orchestration_service.py`)
   - Main orchestration logic
   - Gemini availability tracking
   - Fallback state machine
   - Phone collection handling

2. **Firebase Services** (`app/services/firebase_service.py`)
   - Conversation flow retrieval
   - Session management
   - Lead data persistence

3. **Baileys Integration** (`app/services/baileys_service.py`)
   - WhatsApp message sending
   - User notifications
   - Internal team alerts

### Key Methods

#### `process_message(message, session_id, platform)`
Main entry point that:
1. Attempts Gemini AI response
2. Falls back to Firebase flow if AI fails
3. Handles phone collection
4. Manages session state

#### `_attempt_gemini_response(message, session_id, session_data)`
- Calls Gemini with 15-second timeout
- Detects quota/rate limit errors
- Marks Gemini unavailable on failure
- Automatically restores availability on success

#### `_get_fallback_response(session_data, message)`
- Reads Firestore conversation flow
- Implements deterministic state machine
- Validates answers before advancing
- Never skips steps

#### `_handle_phone_collection(phone_message, session_id, session_data)`
- Validates Brazilian phone numbers
- Formats for WhatsApp integration
- Sends welcome and notification messages
- Saves complete lead data

### Error Detection

The system detects Gemini unavailability through:
- **Timeout errors**: 15-second request timeout
- **Quota errors**: 429 status, "quota exceeded", "ResourceExhausted"
- **API errors**: Network failures, authentication issues

### Answer Validation

- **Name step**: Requires at least 2 words
- **Area step**: Normalizes common variations (criminal→Penal, trabalho→Trabalhista)
- **Situation step**: Minimum 3 characters
- **Phone step**: Brazilian format validation (10-13 digits)

## Configuration

### Environment Variables

```bash
# Firebase Configuration
FIREBASE_CREDENTIALS=firebase-key.json

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key

# WhatsApp Integration
WHATSAPP_BOT_URL=http://law_firm_whatsapp_bot:3000
WHATSAPP_PHONE_NUMBER=+5511918368812
```

### Firestore Setup

1. Create Firebase project
2. Enable Firestore Database
3. Create collections:
   - `conversation_flows`
   - `user_sessions`
   - `leads`

### Default Flow Creation

The system automatically creates a default conversation flow if none exists:

```python
default_flow = {
    "steps": [
        {"id": 1, "question": "Olá! Para começar, qual é o seu nome completo?"},
        {"id": 2, "question": "Em qual área do direito você precisa de ajuda?\n\n• Penal\n• Civil\n• Trabalhista\n• Família\n• Empresarial"},
        {"id": 3, "question": "Por favor, descreva brevemente sua situação ou problema jurídico."},
        {"id": 4, "question": "Gostaria de agendar uma consulta com nosso advogado especializado? (Sim ou Não)"}
    ],
    "completion_message": "Perfeito! Suas informações foram registradas com sucesso..."
}
```

## Testing

### Manual Testing Steps

1. **Normal AI Flow**
   ```bash
   curl -X POST http://localhost:8000/api/v1/conversation/respond \
     -H "Content-Type: application/json" \
     -d '{"message": "Olá", "session_id": "test_ai"}'
   ```

2. **Force Fallback** (simulate Gemini quota exceeded)
   - Temporarily disable Gemini API key
   - Send messages and verify fallback activation

3. **Complete Fallback Flow**
   ```bash
   # Step 1: Name
   curl -X POST http://localhost:8000/api/v1/conversation/respond \
     -H "Content-Type: application/json" \
     -d '{"message": "João Silva", "session_id": "test_fallback"}'
   
   # Step 2: Area
   curl -X POST http://localhost:8000/api/v1/conversation/respond \
     -H "Content-Type: application/json" \
     -d '{"message": "Penal", "session_id": "test_fallback"}'
   
   # Step 3: Situation
   curl -X POST http://localhost:8000/api/v1/conversation/respond \
     -H "Content-Type: application/json" \
     -d '{"message": "Preciso de ajuda com processo criminal", "session_id": "test_fallback"}'
   
   # Step 4: Meeting
   curl -X POST http://localhost:8000/api/v1/conversation/respond \
     -H "Content-Type: application/json" \
     -d '{"message": "Sim", "session_id": "test_fallback"}'
   
   # Phone Collection
   curl -X POST http://localhost:8000/api/v1/conversation/respond \
     -H "Content-Type: application/json" \
     -d '{"message": "11999999999", "session_id": "test_fallback"}'
   ```

4. **Verify WhatsApp Integration**
   - Check WhatsApp bot logs for sent messages
   - Verify user receives welcome message
   - Verify law firm receives notification

### Automated Tests

Run the integration tests:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run fallback tests
python -m pytest tests/test_fallback_integration.py -v
```

## Monitoring & Debugging

### Log Messages

The system provides detailed logging:

```
🤖 Attempting Gemini AI response for session web_123
✅ Valid Gemini response received for session web_123
🚫 Gemini marked unavailable for session web_123: quota: 429 Quota exceeded
⚡ Activating Firebase fallback for session web_123
🚀 Initialized fallback at step 1 for session web_123
💾 Stored answer for step 1: João Silva Santos
➡️ Advanced to step 2 for session web_123
📤 Welcome message sent to user 5511999999999
📤 Internal notification sent to +5511918368812
🔄 Gemini restored for session web_123
```

### Health Check

Check system status:

```bash
curl http://localhost:8000/health
```

Response includes fallback status:
```json
{
  "status": "healthy",
  "services": {
    "gemini_ai": "quota_exceeded",
    "fallback_mode": true
  },
  "fallback_mode": true
}
```

### Session Status

Check individual session status:

```bash
curl http://localhost:8000/api/v1/conversation/status/test_session
```

## Customization

### Modifying Conversation Flow

Update questions directly in Firestore:

1. Open Firebase Console
2. Navigate to Firestore Database
3. Edit `conversation_flows/law_firm_intake`
4. Modify `steps` array or `completion_message`
5. Changes take effect immediately

### Adding Validation Rules

Edit `_validate_and_normalize_answer()` in `orchestration_service.py`:

```python
def _validate_and_normalize_answer(self, answer: str, step_id: int) -> str:
    if step_id == 2:  # Area of law
        # Add new area mappings
        area_map = {
            "tributário": "Tributário",
            "previdenciário": "Previdenciário"
        }
```

### WhatsApp Message Templates

Modify message templates in `_handle_phone_collection()`:

```python
welcome_message = f"""Olá {user_name}! 👋

Sua mensagem personalizada aqui...
"""
```

## Troubleshooting

### Common Issues

1. **Fallback not activating**
   - Check Gemini API key configuration
   - Verify timeout settings (15 seconds)
   - Check error detection logic

2. **Steps being skipped**
   - Verify `_should_advance_step()` logic
   - Check answer validation rules
   - Ensure session persistence

3. **WhatsApp messages not sending**
   - Check Baileys service status
   - Verify phone number formatting
   - Check WhatsApp bot connection

4. **Lead data not saving**
   - Verify Firebase credentials
   - Check Firestore permissions
   - Review `save_lead_data()` calls

### Debug Commands

```bash
# Check conversation flow
curl http://localhost:8000/api/v1/conversation/flow

# Check session data
curl http://localhost:8000/api/v1/conversation/status/SESSION_ID

# Check WhatsApp status
curl http://localhost:3000/api/qr-status

# View logs
docker logs -f law_firm_backend
docker logs -f law_firm_whatsapp_bot
```

## Production Considerations

1. **Gemini Quota Management**
   - Monitor API usage
   - Implement rate limiting
   - Consider multiple API keys

2. **Session Cleanup**
   - Implement session expiration
   - Clean up old session data
   - Archive completed leads

3. **Error Handling**
   - Implement retry logic
   - Add circuit breakers
   - Monitor error rates

4. **Performance**
   - Cache conversation flows
   - Optimize Firestore queries
   - Monitor response times

This fallback system ensures continuous operation even when AI services are unavailable, providing a seamless user experience while maintaining lead collection capabilities.