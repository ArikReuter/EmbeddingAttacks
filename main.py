from Embeddings.EmbeddingModelOpenAI import EmbeddingModelOpenAI
from data import data_utils

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    embedded_data = data_utils.load_embedded_data()
    logger.info(f"Successfully loaded embedded data for {len(embedded_data)} texts.")

