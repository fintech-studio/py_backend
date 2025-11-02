from pydantic import BaseModel
from typing import List, Dict, Any, Literal
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from routers.backtesting_module import db
import json

backtesting_router = APIRouter(prefix="/backtesting", tags=["Backtesting"])


# --- 定義資料模型 ---
class PriceData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class Question(BaseModel):
    symbol: str
    previous_prices: List[PriceData]
    after_prices: List[PriceData]
    previous_indicates: Dict[str, Any]
    correct_ans: Literal["buy", "sell"]
    explanations: List[str]
    profitLoss: float | None = None


class BacktestingRequest(BaseModel):
    server: str
    database: str
    table: str
    user: str
    password: str


@backtesting_router.post("/gen_q", response_model=Question)
def gen_question(request: BacktestingRequest):
    """
    從資料庫隨機生成一個回測題目，
    若歷史資料不足 140 根 K 線，會自動重新抽樣。
    """

    server = "127.0.0.1,1433"
    database = "market_stock_tw"
    table = "trade_signals_1d"
    user = request.user
    password = request.password

    records_with_trading_signal = db.get_trading_signals(
        server=server, database=database, table=table,
        user=user, password=password
    )

    if records_with_trading_signal.empty:
        raise RuntimeError("❌ 沒有任何 Trade_Signal 資料可用")

    # --- 最多嘗試 10 次 ---
    retry_limit = 10
    for attempt in range(retry_limit):
        record = records_with_trading_signal.sample(n=1).to_dict("records")[0]

        trading_signal = record.get("Trade_Signal")
        symbol = record.get("symbol")
        target_date = record.get("datetime")
        correct_ans = "buy" if "買" in str(trading_signal) else "sell"

        explanations = [
            f"指標{str(k)}: {str(v)}"
            for k, v in record.items()
            if k
            not in [
                "Sell_Signals",
                "Buy_Signals",
                "Signal_Strength",
                "Trade_Signal",
                "close_price",
                "symbol",
                "datetime",
                "id",
            ]
            and v != ""
        ]

        prev_data = db.get_previous_stock_records_by_date(
            server=server,
            database=database,
            user=user,
            password=password,
            symbol=symbol,
            target_date=target_date,
        )
        if not isinstance(prev_data, dict):
            prev_data = {"candlesticks": [], "technical_indicator": {}}

        after_data = db.get_after_stock_records_by_date(
            server=server,
            database=database,
            user=user,
            password=password,
            symbol=symbol,
            target_date=target_date,
        )
        if not isinstance(after_data, dict):
            after_data = {"candlesticks": []}

        # --- 資料正規化 ---
        def normalize_prices(records: list[dict]) -> list[dict]:
            result = []
            for r in records:
                date = r.get("datetime") or r.get("date")
                open_ = r.get("open") or r.get("open_price")
                high_ = r.get("high") or r.get("high_price")
                low_ = r.get("low") or r.get("low_price")
                close_ = r.get("close") or r.get("close_price")
                volume_ = r.get("volume")

                if not all([date, open_, high_, low_, close_]):
                    continue

                result.append(
                    {
                        "date": str(date).replace(" ", "T"),
                        "open": float(open_),
                        "high": float(high_),
                        "low": float(low_),
                        "close": float(close_),
                        "volume": (float(volume_)
                                   if volume_ is not None else 0.0),
                    }
                )
            return result

        previous_prices = normalize_prices(prev_data.get("candlesticks", []))
        after_prices = normalize_prices(after_data.get("candlesticks", []))
        previous_indicates = prev_data.get("technical_indicator", {})

        # --- 若 previous 不足 140，則重抽 ---
        if len(previous_prices) < 140:
            print(
                f"⚠️ 第 {attempt+1} 次抽樣失敗：{symbol} "
                f"僅有 {len(previous_prices)} 根K線 (<140)，重新抽樣中..."
            )
            continue  # 跳到下一次抽樣

        # --- 足夠則建立 Question ---
        result = Question(
            symbol=symbol,
            previous_prices=previous_prices,
            after_prices=after_prices,
            previous_indicates=previous_indicates,
            correct_ans=correct_ans,
            explanations=explanations,
        )

        print("✅ 成功生成題目：")
        print(json.dumps(jsonable_encoder(result), indent=2,
                         ensure_ascii=False))
        return result

    # --- 若嘗試多次仍失敗 ---
    raise RuntimeError("❌ 連續 10 次抽樣仍無法取得足夠的歷史資料（>=140 根 K 線）。")
