from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 模型加载 ==========
MODEL_PATH = "best.onnx"
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 获取输入形状（例如 [1,3,640,640]）
input_shape = session.get_inputs()[0].shape

# ========== 类别列表（请根据你的训练顺序调整） ==========
# 注意：顺序必须与训练时完全一致
class_names = [
    "致命鹅膏菌",
    "鸡中菌",
    "蓝牦牛杆菌",
    "毒蝇伞",
    "松茸",
    "白毒伞",
    "绿褶菌",
    "牛肝菌"
]
# 如果你有更多类别，请在此补充完整

# ========== 预处理函数 ==========
def preprocess_image(image_bytes: bytes):
    """将上传的图片转换为模型输入张量"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 调整大小到模型输入尺寸
    img = img.resize((input_shape[2], input_shape[3]))
    # 转换为 numpy 数组并归一化（归一化方式根据训练数据调整）
    img_array = np.array(img).astype(np.float32) / 255.0
    # 转换为通道优先格式 (C, H, W) -> (1, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========== 后处理函数 ==========
def postprocess(outputs):
    """
    解析 ONNX 输出（假设 YOLOv8 格式：输出形状 [1, 84, 8400]）
    将检测结果转换为前端期望的列表格式
    """
    # 假设 outputs[0] 是模型主输出，形状为 [1, 84, 8400]
    predictions = outputs[0]                # shape: [1, 84, 8400]
    predictions = predictions[0]            # shape: [84, 8400]
    predictions = predictions.transpose(1, 0)  # shape: [8400, 84]

    detect_list = []
    for pred in predictions:
        # 置信度（第5个值，索引4）
        conf = pred[4]
        if conf < 0.5:   # 置信度阈值
            continue

        # 类别得分（第5个之后，索引5开始）
        class_scores = pred[5:]
        class_id = int(np.argmax(class_scores))
        class_name = class_names[class_id]

        # 边界框（格式：中心点 x, 中心点 y, 宽度 w, 高度 h）
        x_center, y_center, w, h = pred[:4]
        # 转换为左上角和右下角坐标 (x1, y1, x2, y2)
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        detect_list.append({
            "class_name": class_name,
            "confidence": float(conf),
            "box": [float(x1), float(y1), float(x2), float(y2)]
        })

    return detect_list

# ========== API 路由 ==========
@app.post("/api/plant-detect")
async def plant_detect(file: UploadFile = File(...)):
    try:
        # 1. 读取图片并预处理
        contents = await file.read()
        input_tensor = preprocess_image(contents)

        # 2. 模型推理
        outputs = session.run([output_name], {input_name: input_tensor})

        # 3. 解析输出
        detect_list = postprocess(outputs)

        # 4. 返回结果
        if not detect_list:
            return {"code": 200, "msg": "未检测到任何菌类", "data": {"detections": []}}
        return {
            "code": 200,
            "msg": "识别成功",
            "data": {"detections": detect_list}
        }
    except Exception as e:
        # 记录详细错误日志，并返回便于前端调试的错误信息
        logging.exception(e)
        return {"code": 500, "msg": f"服务器内部错误: {str(e)}"}

@app.get("/")
async def root():
    return {"msg": "青芜植物识别后端运行中", "model_status": "loaded"}

# ========== 启动说明 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)