import logging
import os
from datetime import datetime

base_path = "logs"
if not os.path.exists(base_path):
    os.makedirs(base_path)
# 获取当前时间
now = datetime.now()
# 格式化为字符串
time_string = now.strftime("%Y-%m-%d-%H-%M-%S")  # 格式: 年-月-日 时:分:秒
log_file_name = f"{time_string}.log"
file_path = os.path.join(base_path, log_file_name)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    handlers=[
        logging.FileHandler(file_path, encoding="utf-8"),  # 输出到文件
        logging.StreamHandler()         # 输出到控制台
    ]
)

logger = logging.getLogger(__name__)

