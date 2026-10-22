
import os
import json
import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError

# -------------------- Logging --------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC = os.getenv("TOPIC_TRANSACTIONS", "transactions")

# -------------------- Request Model --------------------

class Txn(BaseModel):
    Time: float = Field(..., description="Seconds elapsed")
    Amount: float
    V: list[float] = Field(
        ...,
        description="Transaction features V1 through V28"
    )

# -------------------- Kafka Producer --------------------

def create_producer():
    max_retries = 10
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Connecting to Kafka at %s (attempt %d/%d)",
                BROKER,
                attempt,
                max_retries
            )

            producer = KafkaProducer(
                bootstrap_servers=[BROKER],
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                request_timeout_ms=10000,
                max_block_ms=10000
            )

            logger.info("Kafka producer initialized successfully")
            return producer

        except (KafkaTimeoutError, KafkaError, OSError, Exception) as exc:
            logger.warning("Kafka connection failed: %s", exc)

            if attempt < max_retries:
                logger.info("Retrying in %d seconds...", retry_delay)
                time.sleep(retry_delay)

    logger.error("Could not connect to Kafka after %d attempts", max_retries)
    return None


producer = create_producer()

# -------------------- FastAPI Application --------------------

app = FastAPI(title="Webhook to Kafka")

# -------------------- Health Check --------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "kafka_connected": producer is not None
    }

# -------------------- Ingest Transaction --------------------

@app.post("/txn")
def ingest(txn: Txn):
    if producer is None:
        raise HTTPException(
            status_code=503,
            detail="Kafka is unavailable. Transaction was not queued."
        )

    # Convert request into the format expected by the stream processor.
    payload = {
        "Time": txn.Time,
        "Amount": txn.Amount
    }

    for i, value in enumerate(txn.V, start=1):
        payload[f"V{i}"] = value

    try:
        # Wait for Kafka to acknowledge the message.
        future = producer.send(TOPIC, payload)
        metadata = future.get(timeout=15)

        logger.info(
            "Transaction sent to topic=%s partition=%s offset=%s",
            metadata.topic,
            metadata.partition,
            metadata.offset
        )

        return {
            "status": "queued",
            "topic": metadata.topic,
            "partition": metadata.partition,
            "offset": metadata.offset
        }

    except Exception as exc:
        logger.exception("Failed to send transaction to Kafka")

        raise HTTPException(
            status_code=503,
            detail=f"Failed to send transaction to Kafka: {str(exc)}"
        )

# -------------------- Shutdown --------------------

@app.on_event("shutdown")
def shutdown_event():
    if producer is not None:
        producer.close(timeout=10)
        logger.info("Kafka producer closed")
