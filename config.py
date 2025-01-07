# 账户的设置
KEY_PATH = r"./key/ancient-jigsaw-442804-q8-85d6ae8e78eb.json"
SERVICE_ACCOUNT = r"gee-555@ancient-jigsaw-442804-q8.iam.gserviceaccount.com"

# 代理的设置
PROXY_URL = r"127.0.0.1:7890"

# 临时文件，最终数据存储文件，状态存档文件的路径
TEMP_PATH = r"./Downloads/temp"
DATA_PATH = r"./Downloads/MODIS"
STATUS_PATH = r"./Downloads/status/"

# 下载和镶嵌的线程数量
MAX_WORKERS = 10  # earthengine.googleapis.com. Connection pool size: 10
MOSAIC_WORKERS = 2

# 下载参数
# 定义区域和分块
ROI = [[[73.5, 18],
        [135.5, 18],
        [135.5, 54],
        [73.5, 54],
        [73.5, 18]]]
SPLIT_LENGTH = 5
# 时间范围
START_DATA = "2023-07-01"
END_DATA = "2023-12-31"
# 频率
FREQ = "1D"
# 数据集要求
DATA_COLLECTION = 'MODIS/061/MYD09GA'
DATA_BAND = ['sur_refl_b02', 'sur_refl_b01', 'QC_500m']
DATA_MASK = ['sur_refl_b02', 'sur_refl_b01']
# 影像分辨率
SCALE = 500

# 质量波段名称,波段数量,目标质量值
QUALITY_BAND = "QC_500m"  # selection
NUM_BANDS = 2
TARGET_QUALITY = 15
# 裁剪文件路径
SHAPEDATA_PATH = r"./shapedata/china_bound.geojson"
