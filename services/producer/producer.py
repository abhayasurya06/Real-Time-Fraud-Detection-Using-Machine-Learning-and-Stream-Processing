
import os
import csv
import time
import json
import logging
import random

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("Kafka not available, running locally")
    KafkaProducer = None
    KafkaError = Exception


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
RATE_MS = int(os.getenv("PRODUCER_RATE_MS", "10"))
TOPIC = os.getenv("TOPIC_TRANSACTIONS", "transactions")
DATA_FILE = os.getenv("DATA_FILE", "/app/data/creditcard.csv")
BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
REPEAT = os.getenv("REPEAT", "false").lower() == "true"
SAMPLE_SIZE = 5000


def create_producer_with_retry():
    if KafkaProducer is None:
        logger.error("Kafka library is unavailable.")
        return None

    for attempt in range(30):
        try:
            logger.info(
                "Connecting to Kafka at %s (attempt %d)",
                BROKER,
                attempt + 1
            )

            kafka_producer = KafkaProducer(
                bootstrap_servers=BROKER,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                request_timeout_ms=30000,
                retry_backoff_ms=1000
            )

            logger.info("Connected to Kafka")
            return kafka_producer

        except KafkaError as error:
            logger.warning("Kafka connection failed: %s", error)
            time.sleep(5)

        except Exception as error:
            logger.error("Unexpected Kafka error: %s", error)
            time.sleep(5)

    raise RuntimeError("Failed to connect to Kafka after 30 attempts")


producer = create_producer_with_retry()


def run_once():
    if not os.path.exists(DATA_FILE):
        logger.error("Dataset not found: %s", DATA_FILE)
        return

    if producer is None:
        logger.error("No Kafka producer available. No transactions sent.")
        return

    logger.info("Reading dataset: %s", DATA_FILE)

    # Read the CSV header and randomly sample up to 1,000 data rows.
    with open(DATA_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames

        if not headers:
            logger.error("Dataset has no header.")
            return

        sampled_rows = random.sample(
            list(reader),
            min(SAMPLE_SIZE, count_csv_rows(DATA_FILE))
        )

    random.shuffle(sampled_rows)

    logger.info("Selected %d random transactions", len(sampled_rows))
    fraud_count = sum(
        int(float(row.get("Class", 0))) == 1
        for row in sampled_rows
    )
    normal_count = len(sampled_rows) - fraud_count

    logger.info("Actual normal transactions: %d", normal_count)
    logger.info("Actual fraud transactions: %d", fraud_count)

    sent_count = 0

    try:
        for index, row in enumerate(sampled_rows, start=1):
            transaction = {}

            # Convert numeric fields, including Amount and Class.
            for key, value in row.items():
                try:
                    transaction[key] = float(value)
                except (ValueError, TypeError):
                    transaction[key] = value

            # Add a processing timestamp without replacing dataset Time.
            transaction["ts"] = time.time()

            # Send each sampled transaction once.
            producer.send(TOPIC, transaction)
            sent_count += 1

            # Print actual amount and dataset label for visibility.
            if index % 50 == 0 or index == 1 or index == len(sampled_rows):
                logger.info(
                    "Queued %d/%d | Amount: %.2f | Actual Class: %s",
                    index,
                    len(sampled_rows),
                    float(transaction.get("Amount", 0)),
                    transaction.get("Class", "unknown")
                )

            if RATE_MS > 0:
                time.sleep(RATE_MS / 1000.0)

        # Wait until queued messages have been delivered to Kafka.
        producer.flush(timeout=60)

        logger.info(
            "Finished sending %d of %d sampled transactions",
            sent_count,
            len(sampled_rows)
        )

    except Exception:
        logger.exception("Error while sending transactions")
        raise


def count_csv_rows(path):
    """Count data rows, excluding the CSV header."""
    with open(path, "r", encoding="utf-8", newline="") as file:
        return max(sum(1 for _ in csv.reader(file)) - 1, 0)


if __name__ == "__main__":
    logger.info("Starting transaction producer")

    try:
        while True:
            run_once()

            if not REPEAT:
                break

            logger.info("Starting another random batch")
    finally:
        if producer is not None:
            producer.flush(timeout=60)
            producer.close(timeout=10)

        logger.info("Producer stopped")
