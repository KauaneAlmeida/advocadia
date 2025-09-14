"""
Baileys WhatsApp Service

This service communicates with the dedicated `whatsapp_bot` container over HTTP.
It handles message sending, status checking, and connection management.
"""

import requests
import logging
import asyncio
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class BaileysWhatsAppService:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("WHATSAPP_BOT_URL", "http://whatsapp_bot:3000")
        self.timeout = 30
        self.max_retries = 3
        
    async def initialize(self):
        """Initialize connection to WhatsApp bot service."""
        try:
            logger.info(f"🔌 Initializing connection to WhatsApp bot at {self.base_url}")
            
            # Test connection with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=10)
                    if response.status_code == 200:
                        logger.info("✅ WhatsApp bot service connection established")
                        return True
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"⚠️ Connection attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(2)
                    else:
                        logger.error(f"❌ Failed to connect to WhatsApp bot after {self.max_retries} attempts: {str(e)}")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error initializing WhatsApp bot connection: {str(e)}")
            return False

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up WhatsApp service resources")

    async def send_whatsapp_message(self, phone_number: str, message: str) -> bool:
        """
        Send a WhatsApp message via whatsapp_bot API.
        
        Args:
            phone_number (str): Phone number in format like "5511999999999@s.whatsapp.net"
            message (str): Message to send
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            # Ensure phone number has correct format
            if "@s.whatsapp.net" not in phone_number:
                # Clean phone number and add WhatsApp format
                clean_phone = ''.join(filter(str.isdigit, phone_number))
                if not clean_phone.startswith("55"):
                    clean_phone = f"55{clean_phone}"
                phone_number = f"{clean_phone}@s.whatsapp.net"
            
            payload = {
                "to": phone_number,
                "message": message
            }
            
            logger.info(f"📤 Sending WhatsApp message to {phone_number[:15]}...")
            
            response = requests.post(
                f"{self.base_url}/send-message",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info(f"✅ WhatsApp message sent successfully to {phone_number[:15]}")
                    return True
                else:
                    logger.error(f"❌ WhatsApp API returned error: {result.get('error', 'Unknown error')}")
                    return False
            else:
                logger.error(f"❌ WhatsApp API request failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("⏰ WhatsApp message request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("🔌 Failed to connect to WhatsApp bot service")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending WhatsApp message: {str(e)}")
            return False

    async def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status from whatsapp_bot API.
        
        Returns:
            Dict[str, Any]: Status information
        """
        try:
            response = requests.get(f"{self.base_url}/api/qr-status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "connected" if data.get("isConnected") else "disconnected",
                    "service": "baileys_whatsapp",
                    "connected": data.get("isConnected", False),
                    "has_qr": data.get("hasQR", False),
                    "phone_number": data.get("phoneNumber", "unknown"),
                    "timestamp": data.get("timestamp"),
                    "qr_url": f"{self.base_url}/qr" if not data.get("isConnected") else None
                }
            else:
                return {
                    "status": "error",
                    "service": "baileys_whatsapp",
                    "connected": False,
                    "error": f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "service_unavailable",
                "service": "baileys_whatsapp", 
                "connected": False,
                "error": "Cannot connect to WhatsApp bot service"
            }
        except Exception as e:
            logger.error(f"❌ Error getting WhatsApp status: {str(e)}")
            return {
                "status": "error",
                "service": "baileys_whatsapp",
                "connected": False,
                "error": str(e)
            }

    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of WhatsApp bot service.
        
        Returns:
            Dict[str, Any]: Health status
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Global instance
baileys_service = BaileysWhatsAppService()

# Wrapper functions for backward compatibility
async def send_baileys_message(phone_number: str, message: str) -> bool:
    """Send WhatsApp message via Baileys service."""
    return await baileys_service.send_whatsapp_message(phone_number, message)

async def get_baileys_status() -> Dict[str, Any]:
    """Get Baileys service status."""
    return await baileys_service.get_connection_status()