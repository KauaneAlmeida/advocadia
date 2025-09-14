# Law Firm AI Chat Backend

A production-ready FastAPI backend for law firm client intake with WhatsApp integration. This application provides a complete solution for automated client consultation, lead generation, and seamless handoff to legal professionals.

## 🚀 Features

- **Guided Conversation Flow**: Fixed sequence of intake questions stored in Firebase
- **Smart Response Handling**: Redirects irrelevant responses back to current question
- **WhatsApp Integration**: Baileys WhatsApp Web API for direct client communication
- **AI-Powered Responses**: Google Gemini integration for legal consultation
- **Lead Management**: Automatic lead capture and storage in Firebase Firestore
- **Phone Collection**: Automated phone number collection with WhatsApp trigger
- **Session Persistence**: Conversation state management across interactions
- **Production Ready**: Docker containerization with health checks and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client/Web    │    │  FastAPI        │    │  WhatsApp Bot   │
│   Frontend      │◄──►│  Backend        │◄──►│  (Baileys)      │
│                 │    │  (Python)       │    │  (Node.js)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Firebase       │    │  WhatsApp       │
                       │  Firestore      │    │  Web API        │
                       └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Google Gemini  │
                       │  AI API         │
                       └─────────────────┘
```

## 📋 Conversation Flow

1. **Guided Intake**: Fixed sequence of questions (name, legal area, situation, meeting preference)
2. **Response Validation**: Irrelevant responses are redirected back to current question
3. **Lead Capture**: Completed responses are saved to Firebase as leads
4. **Phone Collection**: Always asks for WhatsApp phone number after intake
5. **WhatsApp Trigger**: Sends confirmation message via Baileys integration
6. **AI Mode**: Free-form legal consultation using Gemini AI

## 🛠️ Installation & Setup

### Prerequisites

- Docker and Docker Compose
- Firebase project with Firestore enabled
- Google Gemini API key
- WhatsApp account for bot connection

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo>
   cd law-firm-ai-chat
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

3. **Start services**:
   ```bash
   docker compose up --build -d
   ```

4. **Access the application**:
   - **Landing Page with Chat**: http://localhost:8080
   - **WhatsApp QR Code**: http://localhost:3000/qr
   - **Backend API**: http://localhost:8000
   - **Health Check**: http://localhost:8000/health

5. **Connect WhatsApp**:
   - Open http://localhost:3000/qr
   - Scan QR code with WhatsApp
   - Wait for "Connected" status

6. **Test the system**:
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Test conversation flow
   curl -X POST http://localhost:8000/api/v1/conversation/start
   
   # Test chat directly
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello", "session_id": "test"}'
   ```

### Environment Variables

Required variables in `.env`:

```bash
# Firebase Configuration
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"

# Google Gemini AI
GEMINI_API_KEY=your-gemini-api-key

# WhatsApp Configuration
WHATSAPP_PHONE_NUMBER=+5511918368812
WHATSAPP_BOT_URL=http://law_firm_whatsapp_bot:3000
FASTAPI_WEBHOOK_URL=http://law_firm_backend:8000/api/v1/whatsapp/webhook
```

## 📚 API Endpoints

### Conversation Flow
- `POST /api/v1/conversation/start` - Start new intake conversation
- `POST /api/v1/conversation/respond` - Process user responses
- `GET /api/v1/conversation/status/{session_id}` - Get conversation status
- `GET /api/v1/conversation/flow` - Get current flow configuration

### Chat & AI
- `POST /api/v1/chat` - Direct AI chat (for completed flows)
- `GET /api/v1/chat/status` - AI service status

### WhatsApp Integration
- `GET /api/v1/whatsapp/status` - WhatsApp connection status
- `POST /api/v1/whatsapp/send` - Send WhatsApp message

### System
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

## 🔧 Configuration

### Firebase Setup

1. Create Firebase project at https://console.firebase.google.com
2. Enable Firestore Database
3. Create service account and download credentials
4. Set environment variables in `.env`

### Conversation Flow Management

The conversation flow is stored in Firebase Firestore at:
`conversation_flows/law_firm_intake`

Lawyers can update questions directly in Firebase Console without code changes.

Default flow structure:
```json
{
  "steps": [
    {
      "id": 1,
      "question": "Qual é o seu nome completo?",
      "field": "name",
      "required": true
    },
    {
      "id": 2, 
      "question": "Em qual área do direito você precisa de ajuda?",
      "field": "area_of_law",
      "required": true
    }
  ]
}
```

### AI Configuration

The system uses Google Gemini for legal consultation responses with:
- Professional, empathetic tone
- Brazilian Portuguese responses
- Legal disclaimer requirements
- Consultation recommendations
- Response length optimization

## 🐳 Docker Services

### Backend Service (`law_firm_backend`)
- **Port**: 8000
- **Technology**: FastAPI (Python)
- **Features**: Conversation flow, AI integration, Firebase
- **Health Check**: `http://localhost:8000/health`

### Frontend Service (`law_firm_frontend`)
- **Port**: 8080
- **Technology**: Nginx + Static HTML/JS
- **Features**: Landing page with integrated chat
- **URL**: `http://localhost:8080`

### WhatsApp Bot Service (`law_firm_whatsapp_bot`)
- **Port**: 3000
- **Technology**: Baileys (Node.js)
- **Features**: WhatsApp Web API, QR code generation, message sending
- **Health Check**: `http://localhost:3000/health`
- **QR Code**: `http://localhost:3000/qr`

## 📊 Monitoring & Logging

### Health Checks
Both services include comprehensive health checks:
- Service availability
- Dependency connectivity (Firebase, WhatsApp)
- Resource utilization

### Logging
Structured logging throughout the application:
- Request/response tracking
- Error handling and debugging
- Conversation flow progression
- WhatsApp message status

### Resource Limits
Docker services are configured with resource limits:
- Backend: 512MB RAM, 0.5 CPU
- WhatsApp Bot: 256MB RAM, 0.3 CPU

## 🔒 Security & Best Practices

- **Non-root containers**: Services run as non-root users
- **Environment isolation**: Sensitive data in environment variables
- **Input validation**: Pydantic models for request validation
- **Error handling**: Comprehensive exception handling
- **Session management**: Secure session storage in Firebase
- **Rate limiting**: Ready for implementation
- **CORS configuration**: Configurable for production

## 🚀 Production Deployment

### Docker Compose (Recommended)
```bash
# Production deployment
docker compose -f docker-compose.yml up -d

# View logs
docker compose logs -f

# Scale services
docker compose up --scale law_firm_backend=2 -d
```

### Environment-Specific Configuration
- Development: Use `.env` file
- Production: Set environment variables directly
- Staging: Use separate Firebase project

## 📈 Scaling Considerations

- **Horizontal Scaling**: Multiple backend instances behind load balancer
- **Database**: Firebase Firestore auto-scales
- **WhatsApp**: Single instance per phone number (Baileys limitation)
- **AI API**: Gemini API handles concurrent requests
- **Session Storage**: Firebase supports high concurrency

## 🛠️ Development

### Local Development
```bash
# Backend only
cd app
uvicorn main:app --reload --port 8000

# WhatsApp bot only
node whatsapp_baileys.js

# Full stack
docker compose up --build
```

### Frontend Development
```bash
# Serve frontend locally for development
cd frontend
python -m http.server 8080
# or
npx serve -p 8080
```

### Testing
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Test full conversation flow
curl -X POST http://localhost:8000/api/v1/conversation/start
curl -X POST http://localhost:8000/api/v1/conversation/respond \
  -H "Content-Type: application/json" \
  -d '{"message": "João Silva", "session_id": "test_session"}'
```

## 📝 Customization

### Adding New Questions
1. Open Firebase Console
2. Navigate to Firestore Database
3. Edit `conversation_flows/law_firm_intake`
4. Add new step with `id`, `question`, `field`, `required`
5. Changes take effect immediately

### Modifying AI Responses
Edit the system prompt in `app/services/ai_service.py`:
```python
LEGAL_AI_SYSTEM_PROMPT = """
Your custom legal AI prompt here...
"""
```

### WhatsApp Message Templates
Customize messages in `app/services/conversation_service.py`:
```python
whatsapp_message = f"""
Your custom WhatsApp message template...
"""
```

## 🐛 Troubleshooting

### Common Issues

1. **Frontend not connecting to backend**:
   - Check if all services are running: `docker compose ps`
   - Verify CORS configuration in FastAPI
   - Check nginx proxy configuration
   - Test API directly: `curl http://localhost:8000/health`

1. **WhatsApp not connecting**:
   - Check QR code at http://localhost:3000/qr
   - Ensure phone has internet connection
   - Try restarting WhatsApp bot container

2. **Firebase connection errors**:
   - Verify environment variables
   - Check service account permissions
   - Ensure Firestore is enabled

3. **AI responses not working**:
   - Verify GEMINI_API_KEY is set
   - Check API quota limits
   - Review logs for specific errors

### Logs and Debugging
```bash
# View all logs
docker compose logs -f

# Individual service logs
docker compose logs -f law_firm_backend
docker compose logs -f law_firm_frontend
docker compose logs -f law_firm_whatsapp_bot

# Follow specific container with timestamps
docker logs -f --timestamps law_firm_backend
```

### Service URLs
- **Landing Page**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **WhatsApp QR**: http://localhost:3000/qr
- **Health Checks**: 
  - Backend: http://localhost:8000/health
  - WhatsApp: http://localhost:3000/health

## 📞 Support

For technical support:
1. Check the logs for error details
2. Verify all environment variables are set
3. Test individual services with health checks
4. Review Firebase Console for data issues

## 📄 License

This project is provided for educational and commercial use. Please ensure compliance with WhatsApp Business API terms of service and local regulations regarding automated legal consultation.

---

**Built with ❤️ for legal professionals who want to automate client intake while maintaining personal touch through WhatsApp integration.**