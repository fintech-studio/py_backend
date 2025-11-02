# 股票預測與回測後端 API

基於 Chronos 時間序列預測模型的股票預測與回測系統 FastAPI 後端服務。

## 功能特色

- **股票預測**: 使用 Chronos 預訓練模型進行股票價格預測
- **長期評估**: 支援長期預測評估與相似度分析
- **回測系統**: 從 SQL Server 資料庫生成交易信號回測題目
- **GPU 加速**: 支援 CUDA GPU 加速推理

## 技術架構

- **框架**: FastAPI
- **預測模型**: Chronos-Forecast
- **深度學習**: PyTorch
- **資料庫**: SQL Server
- **數據處理**: NumPy, Pandas
- **部署**: Uvicorn

## 環境需求

- Python 3.10+
- CUDA 支援的 GPU (推薦)
- SQL Server 資料庫連線
- Conda 環境管理

## 安裝與設定

1. 建立 Conda 環境：

```bash
conda env create -f environment.yml
conda activate py_backend
```

或使用以下提供的腳本手動安裝：

```bash
conda create --name <environment_name> python=3.10 -y
conda activate <environment_name>
conda install conda-forge::uvicorn -y
conda install conda-forge::fastapi -y
conda install pandas -y
pip3 install pyodbc -y
pip3 install chronos-forecasting
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

2. 啟動服務：

```bash
cd app
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

### 回測系統 (`/backtesting`)

#### POST `/backtesting/gen_q`

從資料庫生成回測題目

**請求參數**:

```json
{
  "server": "資料庫伺服器",
  "database": "資料庫名稱",
  "table": "資料表名稱",
  "user": "使用者名稱",
  "password": "密碼"
}
```

**回應**:

```json
{
  "symbol": "股票代號",
  "previous_prices": [歷史價格資料],
  "after_prices": [後續價格資料],
  "previous_indicates": {技術指標資料},
  "correct_ans": "buy/sell",
  "explanations": ["指標說明"]
}
```

## 專案結構

```
py_backend/
├── app/
│   ├── main.py                          # 主應用程式
│   ├── routers/
│   │   ├── stock_prediction.py          # 股票預測路由
│   │   ├── backtesting.py               # 回測系統路由
│   │   └── backtesting_module/
│   │       └── db.py                    # 資料庫操作模組
│   ├── output/
│   │   └── gooood/
│   │       └── checkpoint-final/        # Chronos 預訓練模型
├── environment.yml                      # Conda 環境配置
└── README.md
```

## 模型說明

- 使用 Chronos 預訓練時間序列預測模型
- 模型位於 `app/output/gooood/checkpoint-final/`
- 支援 GPU 推理 (torch.bfloat16)
- 預設 context_length: 192, prediction_length: 12

## 資料庫配置

回測系統需要連接 SQL Server 資料庫，包含以下資料表：

- `trade_signals_1d`: 交易信號資料
- `stock_data_1d`: 股票日線資料

## 注意事項

- 輸入數據應為時間序列格式 (oldest → newest)
- 需要足夠的歷史數據 (至少 context_length + prediction_length)
- 回測系統要求至少 140 根 K 線歷史資料
- 建議使用 GPU 以獲得最佳預測性能
- 資料庫連線需要正確的 ODBC Driver 17 for SQL Server
