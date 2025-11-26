import logging
import sys
from tqdm import tqdm
from config import Config
from core.engine import AuditEngine
from core.data_loader import AuditDataLoader
from core.io_manager import IOManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Main")

def main():
    try:
        # 1. Init Config
        cfg = Config()

        # 2. Init Modules
        io_mgr = IOManager()
        data_loader = AuditDataLoader("data/prompts.json")
        engine = AuditEngine(cfg)

        total_prompts = len(data_loader.data)
        logger.info(f"Starting audit on {total_prompts} vectors with Batch Size {cfg.BATCH_SIZE}")

        # 3. Execution Loop
        # tqdm used for a professional progress bar
        pbar = tqdm(total=total_prompts, desc="Auditing")

        for batch in data_loader.get_batches(cfg.BATCH_SIZE):
            try:
                # Run inference
                batch_results = engine.generate_batch(batch)

                # Stream results immediately
                for res in batch_results:
                    io_mgr.stream_result(res)

            except Exception as batch_err:
                logger.error(f"Batch execution failed: {batch_err}")
                # Log failed batch IDs to a separate file or continue
                continue

            pbar.update(len(batch))

        pbar.close()
        logger.info("Audit cycle complete.")

    except KeyboardInterrupt:
        logger.info("Audit interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal System Error: {e}")

if __name__ == "__main__":
    main()
