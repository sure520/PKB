from langchain_community.chat_models import QwenChat as QwenChatNew
from langchain_core.messages import HumanMessage
import base64
from pathlib import Path
from PIL import Image
import io


def extract_text_with_langchain_modern_qwen_ocr(image_path):
    """使用新版 LangChain + Qwen VL OCR 提取图片文字"""
    try:
        logger.info(f"开始通过新版 LangChain 处理图片: {image_path}")

        # 初始化模型
        model = QwenChatNew(model="qwen-vl-ocr")

        # 读取图像并编码为 Base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        # 构建消息内容
        message = HumanMessage(
            content=[
                {"type": "text", "text": "请识别这张图片中的所有文字内容，并以清晰的方式返回文本结果。"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_string}"}
            ]
        )

        # 调用模型
        response = model.invoke([message])

        # 获取 OCR 文本结果
        ocr_text = response.content.strip()
        logger.info(f"OCR提取完成，提取文本长度: {len(ocr_text)}")
        return ocr_text

    except Exception as e:
        logger.error(f"OCR提取文本失败: {str(e)}")
        logger.error(traceback.format_exc())
        return f"OCR提取失败: {str(e)}"
