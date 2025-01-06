import ee
import geemap
import os
import pandas as pd
import numpy as np
import datetime
import fiona
import glob
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import json
from log import logger

TEMP_PATH = r"./Downloads/temp"
DATA_PATH = r"./Downloads/MODIS"
STATUS_PATH = r"./Downloads/status/"

# 下载和镶嵌的线程数量
MAX_WORKERS = 12
MOSAIC_WORKERS = 2  # 镶嵌线程池最大任务数

# 镶嵌线程池
mosaic_executor = ThreadPoolExecutor(max_workers=MOSAIC_WORKERS)


def initialize_gee(service_account, key_file, proxy_url):
    """初始化 Google Earth Engine 环境"""
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    ee.Initialize(credentials)
    logger.info('Google Earth Engine 已初始化')


def calculate_splits(bbox, split_length):
    """根据 ROI 的边界框计算分块的数量"""
    width = bbox[2][0] - bbox[0][0]
    height = bbox[2][1] - bbox[0][1]
    num_width = int(np.ceil(width / split_length))
    num_height = int(np.ceil(height / split_length))
    logger.info(f"ROI 宽度: {width}, 高度: {height}, 分块数: {num_width}x{num_height}")
    return num_width, num_height


def mask_cloud_and_water(sentinel2, selection, num_bands=2, target_quality=15):
    """掩膜云和水体"""
    QA = sentinel2.select(selection).toInt()
    mask = ee.Image.constant(1)
    for i in range(num_bands):  # 遍历波段
        startBit = 2 + i * 4
        bandQuality = QA.rightShift(startBit).bitwiseAnd(15)
        mask = mask.min(bandQuality.neq(target_quality))
    return sentinel2.updateMask(mask)


def load_download_status(status_file):
    """加载分块下载状态"""
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {}


def save_download_status(status_file, status):
    """保存分块下载状态"""
    with open(status_file, 'w') as f:
        json.dump(status, f)


def download_segment(w, h, data, prefix, split_length, bbox, scale, status, status_file, retries=5, hold_time=10):
    """下载单个分块影像"""
    xmin = bbox[0][0] + w * split_length
    ymin = bbox[0][1] + h * split_length
    xmax = xmin + split_length
    ymax = ymin + split_length
    split_roi = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
    pre_fn = f"{prefix}_{w}_{h}.tif"

    if status.get(pre_fn) == "completed":
        logger.info(f"{pre_fn} 已完成，跳过")
        return

    temp_fn = os.path.join(TEMP_PATH, pre_fn)
    success = False
    for attempt in range(retries):
        try:
            geemap.ee_export_image(data, filename=temp_fn, scale=scale, region=split_roi, file_per_band=False)
            logger.info(f"分块下载成功: {temp_fn}")
            status[pre_fn] = "completed"
            save_download_status(status_file, status)
            success = True
            break
        except Exception as e:
            logger.error(f"下载失败: {e}，重试 ({attempt + 1}/{retries})")
            time.sleep(hold_time)

    if not success:
        logger.error(f"超过最大重试次数，下载失败: {temp_fn}")
        status[pre_fn] = "failed"
        save_download_status(status_file, status)


def mosaic_image(prefix):
    """
    镶嵌分块影像，并删除相关的临时文件。
    """
    mosaic_status_file = os.path.join(STATUS_PATH, f"{prefix}_mosaic_status.json")

    # 检查是否已经镶嵌完成
    if os.path.exists(mosaic_status_file):
        with open(mosaic_status_file, 'r') as f:
            mosaic_status = json.load(f)
        if mosaic_status.get('status') == 'completed':
            logger.info(f"镶嵌任务已完成: {prefix}, 跳过")
            return
    else:
        # 初始化镶嵌状态
        mosaic_status = {"status": "incomplete"}
        with open(mosaic_status_file, 'w') as f:
            json.dump(mosaic_status, f)

    try:
        logger.info(f"开始镶嵌影像: {prefix}")
        search_criteria = os.path.join(TEMP_PATH, f"{prefix}_*.tif")
        tiffs = glob.glob(search_criteria)

        if tiffs:
            # 打开所有分块影像文件
            src_files_to_mosaic = [rasterio.open(fp) for fp in tiffs]

            # 执行镶嵌
            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update(
                {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans}
            )

            # 保存镶嵌后的影像
            mosaic_name = os.path.join(DATA_PATH, f"{prefix}.tif")
            with rasterio.open(mosaic_name, "w", **out_meta) as dest:
                dest.write(mosaic)
            logger.info(f"镶嵌完成: {mosaic_name}")

            # 关闭影像文件句柄
            for src in src_files_to_mosaic:
                src.close()

            # 删除临时文件
            for temp_file in tiffs:
                try:
                    os.remove(temp_file)
                    logger.info(f"删除临时文件: {temp_file}")
                except Exception as e:
                    logger.error(f"删除临时文件失败: {temp_file}, 错误: {e}")

            # 更新镶嵌状态为完成
            mosaic_status['status'] = 'completed'
            with open(mosaic_status_file, 'w') as f:
                json.dump(mosaic_status, f)

        else:
            logger.warning(f"未找到分块影像: {prefix}")

    except Exception as e:
        logger.error(f"镶嵌失败: {e}")


def process_image(date_range, prefix, roi, scale, split_length, bbox, num_width, num_height,
                  max_workers, data_collection, data_band, data_mask):
    """
    处理每个日期的影像下载和镶嵌任务（镶嵌在独立线程中进行）。
    """
    status_file = f"{STATUS_PATH}{prefix}_status.json"
    logger.info(f"status_file={status_file}")
    mosaic_name = os.path.join(DATA_PATH, f"{prefix}.tif")

    if os.path.exists(mosaic_name):
        logger.info(f"{mosaic_name} 已存在，跳过")
        return

    # 准备影像
    start, end = date_range
    sentinel2 = ee.ImageCollection(data_collection) \
        .select(data_band) \
        .filterDate(start, end) \
        .filterBounds(roi).median().clip(roi)
    data = mask_cloud_and_water(sentinel2=sentinel2, selection="QC_500m", num_bands=len(data_mask),
                                target_quality=15).normalizedDifference(data_mask)

    # 加载下载状态
    status = load_download_status(status_file)

    # 并行下载分块
    logger.info(f"开始并行下载分块 ({num_width}x{num_height})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_segment, w, h, data, prefix, split_length, bbox, scale, status, status_file)
            for w in range(num_width) for h in range(num_height)
        ]

        # 等待所有分块下载完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"任务失败: {e}")

    # 分块下载完成后，提交镶嵌任务到镶嵌线程池
    logger.info(f"提交镶嵌任务到线程池: {prefix}")
    mosaic_executor.submit(mosaic_image, prefix)


def resume_incomplete_tasks():
    """检查并恢复未完成的镶嵌任务"""
    logger.info("检查未完成的镶嵌任务...")
    for status_file in glob.glob(os.path.join(STATUS_PATH, "*_mosaic_status.json")):
        with open(status_file, 'r') as f:
            mosaic_status = json.load(f)
        if mosaic_status.get('status') == 'incomplete':
            prefix = os.path.basename(status_file).replace("_mosaic_status.json", "")
            logger.info(f"恢复未完成的镶嵌任务: {prefix}")
            mosaic_executor.submit(mosaic_image, prefix)


def main():
    """主程序入口"""
    for dir_path in [STATUS_PATH, DATA_PATH, TEMP_PATH]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建目录: {dir_path}")
    # 初始化
    initialize_gee(
        service_account="gee-555@ancient-jigsaw-442804-q8.iam.gserviceaccount.com",
        key_file=r"D:\YJ_data_download\GEE\data\key\ancient-jigsaw-442804-q8-85d6ae8e78eb.json",
        proxy_url="127.0.0.1:7890"
    )

    # 恢复未完成的任务
    resume_incomplete_tasks()

    # 定义区域和分块
    roi = ee.Geometry.Polygon(
        [[[73.5, 18],
          [135.5, 18],
          [135.5, 54],
          [73.5, 54],
          [73.5, 18]]]
    )
    bbox = roi.bounds().getInfo()['coordinates'][0]
    num_width, num_height = calculate_splits(bbox, split_length=5)

    # 时间范围
    start_date = "2023-07-01"
    end_date = "2023-12-31"
    mdate = pd.date_range(start=start_date, end=end_date, freq="1D").strftime("%Y-%m-%d")
    edate = mdate[1:]
    sdate = mdate[:-1]

    # 数据要求
    data_collection = 'MODIS/061/MYD09GA'
    data_band = ['sur_refl_b02', 'sur_refl_b01', 'QC_500m']
    data_mask = ['sur_refl_b02', 'sur_refl_b01']

    # 下载和处理
    for i, (start, end) in enumerate(zip(sdate, edate)):
        prefix = f"NDVI_{start}_{datetime.datetime.strptime(start, '%Y-%m-%d').strftime('%j')}"
        process_image((start, end), prefix, roi=roi, scale=500, split_length=5, bbox=bbox, num_width=num_width,
                      num_height=num_height, max_workers=MAX_WORKERS, data_collection=data_collection,
                      data_band=data_band,
                      data_mask=data_mask
                      )


if __name__ == "__main__":
    main()
