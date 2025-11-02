from fastapi import FastAPI
from routers.stock_prediction import router
from routers.backtesting import backtesting_router
import pandas as pd
import os
app = FastAPI()
app.include_router(router)
app.include_router(backtesting_router)


if __name__ == "__main__":
    try:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cpath = os.path.join(current_dir, "NYSE%3ATR.csv")

        ls = pd.read_csv(cpath)["close"]

        from routers.stock_prediction import (PredictRequest, long_term_eval)
        req = PredictRequest(
            data_numpy=ls, context_length=192, prediction_length=12)

        res = long_term_eval(req)
        print(len(res["predict"]))
        print(len(res["actual"]))

    except Exception:
        import traceback
        print("Error occurred:")
        traceback.print_exc()
