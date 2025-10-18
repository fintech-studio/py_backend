import torch
from chronos import BaseChronosPipeline
import numpy as np
from numpy.linalg import norm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

router = APIRouter(prefix="/stock_prediction", tags=["Predict"])

log_buffer = []


class PredictRequest(BaseModel):
    data_numpy: list
    context_length: int = 192
    prediction_length: int = 12


@router.post('/predict')
def predict(req: PredictRequest):
    try:
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            current_dir, "..", "output", "gooood", "checkpoint-final")

        # 轉為 numpy 並確保時序為 oldest -> newest
        data = np.array(req.data_numpy, dtype=float)
        # 前端 DB 查詢通常是 DESC (newest first)，反向成 chronological
        data = data[::-1]
        lens = len(data)
        if lens < req.context_length + 1:
            raise Exception("Not enough data to evaluate")

        pipeline = BaseChronosPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        # 取最後面的 context_length（最近的序列）
        context_data = data[-req.context_length:].tolist()
        context_tensor = torch.tensor(
            context_data, dtype=torch.float32).unsqueeze(0)

        quantiles, mean = pipeline.predict_quantiles(
            context=context_tensor,
            prediction_length=req.prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        # 保證回傳純 Python list
        mean_arr = np.asarray(mean[0]).tolist()

        return {"mean": mean_arr}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/long_term_eval')
def long_term_eval(req: PredictRequest):
    try:
        # 先將資料轉為 numpy 並整理為 chronological (oldest -> newest)
        data = np.array(req.data_numpy, dtype=float)
        # 若前端傳 newest-first，反轉為 oldest-first
        data = data[::-1]

        # 若資料太長，保留最近的部分 (最近 = 最後面的元素)
        if data.shape[0] > 3000:
            data = data[-3000:]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            current_dir, "..", "output", "gooood", "checkpoint-final")

        lens = len(data)
        if lens < req.context_length + req.prediction_length:
            raise Exception("Not enough data to perform long term evaluation")

        pipeline = BaseChronosPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        index = req.context_length
        true_values = []
        pred_values = []

        # index 以 chronological (oldest->newest) 的位置前進
        while index <= lens - req.prediction_length:
            context_data = data[index - req.context_length: index].tolist()
            context_tensor = torch.tensor(
                context_data, dtype=torch.float32).unsqueeze(0)

            quantiles, mean = pipeline.predict_quantiles(
                context=context_tensor,
                prediction_length=req.prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
            )

            true_val = data[index: index + req.prediction_length].tolist()

            true_values.extend(true_val)
            pred_values.extend(np.asarray(mean[0]).tolist())

            index += req.prediction_length

        # 計算 cosine similarity，避免除以零
        true_arr = np.array(true_values, dtype=float)
        pred_arr = np.array(pred_values, dtype=float)
        denom = (norm(true_arr) * norm(pred_arr)) + 1e-9
        con_sim = float(np.dot(true_arr, pred_arr) / denom)

        return {"predict": pred_values,
                "true_value": true_values,
                "sim": con_sim}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/pylog')
def pylog():
    return log_buffer.pop()


if __name__ == "__main__":
    try:
        req = PredictRequest(data_numpy=[
            0.139, 0.982, 0.275, 0.434, 0.613, 0.791, 0.388, 0.119,
            0.147, 0.692, 0.730, 0.325, 0.006, 0.825, 0.220, 0.506,
            0.848, 0.396, 0.147, 0.453, 0.209, 0.047, 0.674, 0.034,
            0.765, 0.168, 0.943, 0.412, 0.191, 0.234, 0.971, 0.154,
            0.713, 0.599, 0.277, 0.349, 0.950, 0.682, 0.287, 0.993,
            0.148, 0.213, 0.528, 0.719, 0.210, 0.653, 0.082, 0.909,
            0.997, 0.414, 0.329, 0.521, 0.101, 0.832, 0.704, 0.775,
            0.417, 0.053, 0.234, 0.568, 0.876, 0.993, 0.341, 0.112,
            0.419, 0.209, 0.298, 0.627, 0.567, 0.944, 0.861, 0.339,
            0.235, 0.714, 0.198, 0.583, 0.001, 0.671, 0.917, 0.327,
            0.159, 0.731, 0.676, 0.210, 0.495, 0.761, 0.248, 0.347,
            0.836, 0.470, 0.140, 0.362, 0.676, 0.582, 0.818, 0.397,
            0.587, 0.168, 0.907, 0.283, 0.267, 0.509, 0.174, 0.687,
            0.964, 0.115, 0.234, 0.637, 0.765, 0.382, 0.940, 0.745,
            0.301, 0.489, 0.591, 0.119, 0.808, 0.245, 0.374, 0.654,
            0.034, 0.283, 0.184, 0.724, 0.957, 0.329, 0.765, 0.827,
            0.028, 0.104, 0.681, 0.226, 0.219, 0.749, 0.950, 0.237,
            0.863, 0.014, 0.434, 0.735, 0.274, 0.660, 0.572, 0.098,
            0.791, 0.051, 0.427, 0.198, 0.806, 0.338, 0.109, 0.905,
            0.438, 0.783, 0.241, 0.265, 0.762, 0.672, 0.380, 0.874,
            0.108, 0.256, 0.512, 0.037, 0.601, 0.697, 0.230, 0.416,
            0.742, 0.903, 0.018, 0.695, 0.091, 0.367, 0.701, 0.864,
            0.158, 0.311, 0.270, 0.021, 0.859, 0.116, 0.094, 0.576,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ], context_length=192, prediction_length=12)

        res = predict(req)
        print(res)

    except Exception:
        import traceback

        print("Error occurred:")
        traceback.print_exc()
