import logging

logging.basicConfig(level=logging.INFO, filename="./logs/demo.txt",filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 声明了一个 Logger 对象
logger = logging.getLogger(__name__)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
