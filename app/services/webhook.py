import httpx
from typing import Dict, Any
from loguru import logger

class WebhookService:
    @staticmethod
    async def submit_result(webhook_url: str, payload: Dict[str, Any]) -> None:
        """
        Submits the scraping result to the external webhook.
        """
        if not webhook_url:
            logger.warning("No webhook URL provided. Skipping submission.")
            return

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(webhook_url, json=payload, timeout=10.0)
                if response.status_code != 200:
                    logger.error(f"Webhook submission failed with status {response.status_code}: {response.text}")
                else:
                    logger.info(f"Webhook submission successful.")
            except httpx.RequestError as e:
                logger.error(f"Webhook submission error: {str(e)}")
