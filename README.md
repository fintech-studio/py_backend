# 股票預測後端 API

基於 Chronos 時間序列預測模型的股票預測 FastAPI 後端服務。

## 功能特色

- 使用 Chronos 預訓練模型進行股票價格預測
- 支援短期和長期預測評估
- GPU 加速推理

## 技術架構

- **框架**: FastAPI
- **預測模型**: Chronos-Forecast
- **深度學習**: PyTorch
- **數據處理**: NumPy, Pandas
- **部署**: Uvicorn

## 環境需求

- Python 3.10+
- CUDA 支援的 GPU (推薦)
- Conda 環境管理

## 安裝與設定

1. 建立 Conda 環境：

```bash
conda env create -f environment.yml
conda activate py_backend
```

或使用以下提供的腳本：

```bash
conda create --name <environment_name> python=3.10 -y
conda activate <environment_name>
conda install conda-forge::uvicorn -y
conda install conda-forge::fastapi -y
conda install pandas -y
pip3 install chronos-forecasting
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

2. 啟動服務：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## API 端點

### 股票預測 (`/stock_prediction`)

#### POST `/stock_prediction/predict`

單次預測股票價格

**請求參數**:

```json
{
  "data_numpy": [股價數據陣列],
  "context_length": 192,
  "prediction_length": 12
}
```

**回應**:

```json
{
  "mean": [預測價格陣列]
}
```

#### POST `/stock_prediction/long_term_eval`

長期預測評估與相似度分析

**請求參數**:

```json
{
  "data_numpy": [股價數據陣列],
  "context_length": 192,
  "prediction_length": 12
}
```

**回應**:

```json
{
  "predict": [預測值陣列],
  "true_value": [實際值陣列],
  "sim": 餘弦相似度分數
}
```

## 專案結構

```
py_backend_test/
├── app/
│   ├── main.py                     # 主應用程式
│   ├── routers/
│   │   ├── stock_prediction.py     # 股票預測路由
│   │   └── concept.py
│   ├── output/
│   │   └── gooood/
│   │       └── checkpoint-final/   # 預訓練模型
├── conda_env_setup                 # Conda 環境設定腳本
├── environment.yml                 # Conda 環境配置
└── README.md
```

## 注意事項

- 輸入數據應為時間序列格式 (oldest → newest)
- 需要足夠的歷史數據 (至少 context_length + prediction_length)
- 模型檔案需放置在正確路徑
- 建議使用 GPU 以獲得最佳性能
