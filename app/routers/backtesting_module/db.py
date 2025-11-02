import pandas as pd
import pyodbc


def get_trading_signals(
    server,
    database,
    table,
    user,
    password,
    chunk_size=50000,
):
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={user};PWD={password};"
        f"Trusted_Connection=no;Connection Timeout=30;"
        f"Application Name=TechnicalAnalysis"
    )

    print("🟡 [DEBUG] 嘗試連線資料庫...")
    print(f"    SERVER: {server}")
    print(f"    DATABASE: {database}")
    print(f"    TABLE: {table}")
    print(f"    USER: {user}")
    # Helper: try multiple encodings by creating a fresh connection each time.
    encodings_to_try = ["utf-8", "cp950", "mbcs", "latin-1"]

    def try_read_sql(query, params=None):
        last_err = None
        for enc in encodings_to_try:
            conn = None
            try:
                conn = pyodbc.connect(conn_str)
                # setdecoding may not raise immediately,
                # but will affect how pyodbc decodes column data
                try:
                    conn.setdecoding(pyodbc.SQL_CHAR, encoding=enc)
                    conn.setdecoding(pyodbc.SQL_WCHAR, encoding=enc)
                except Exception as s_err:
                    print(f"⚠️ 無法在此連線設定解碼 {enc}：{s_err}")

                print(f"ℹ️ 嘗試以 '{enc}' 解碼執行 SQL：{query[:80]}...")
                df = pd.read_sql(query, conn, params=params)
                print(f"✅ 使用 '{enc}' 成功讀取 {len(df):,} 筆資料")
                return df, conn, enc
            except Exception as e:
                last_err = e
                print(f"⚠️ 使用 '{enc}' 讀取失敗：{e}")
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
        print(f"❌ 所有解碼嘗試都失敗，最後錯誤：{last_err}")
        return pd.DataFrame(), None, None

    try:
        # --- 檢查表是否存在 ---
        test_query = f"SELECT TOP 1 * FROM {table}"
        test_df, conn, used_encoding = try_read_sql(test_query)
        if conn is None or test_df.empty:
            print(f"❌ 無法從 {table} 讀取資料或資料表為空（encoding used: {used_encoding}）")
            return pd.DataFrame()

        print(f"✅ 資料表 {table} 成功讀取，欄位共 {len(test_df.columns)} 個：")
        print(f"   {list(test_df.columns)}")

        # --- 統計筆數 ---
        count_query = (f"SELECT COUNT(*) FROM {table} "
                       f"WHERE Trade_Signal IS NOT NULL")
        try:
            cursor = conn.cursor()
            row_count = cursor.execute(count_query).fetchval()
        except Exception as e:
            print(f"⚠️ 使用現有連線取得筆數失敗，嘗試直接執行 count_query：{e}")
            count_df, tmp_conn, _ = try_read_sql(count_query)
            if tmp_conn is not None and not count_df.empty:
                row_count = int(count_df.iat[0, 0])
                tmp_conn.close()
            else:
                print("❌ 無法取得筆數，放棄。")
                conn.close()
                return pd.DataFrame()

        print(f"📊 Trade_Signal 不為 NULL 的筆數：{row_count:,}")

        if row_count == 0:
            print("⚠️ 沒有任何 Trade_Signal 資料（可能欄位名不對或值為空）")
            conn.close()
            return pd.DataFrame()

        if row_count <= chunk_size:
            query = (f"SELECT TOP {chunk_size} * FROM {table} "
                     f"WHERE Trade_Signal IS NOT NULL ORDER BY datetime")
            try:
                df = pd.read_sql(query, conn)
                print(f"✅ 一次讀取 {len(df):,} "
                      f"筆資料 (used encoding: {used_encoding})")
            except Exception as e:
                print(f"⚠️ 以 {used_encoding} 讀取主資料失敗：{e}，嘗試其他編碼")
                df, new_conn, new_enc = try_read_sql(query)
                if new_conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = new_conn
                    used_encoding = new_enc
        else:
            print("🟡 資料量過大，改為分批讀取...")
            date_range_query = (f"SELECT MIN(datetime) as min_date, "
                                f"MAX(datetime) as max_date FROM {table}")
            date_range = pd.read_sql(date_range_query, conn)
            min_date = date_range['min_date'].iloc[0]
            max_date = date_range['max_date'].iloc[0]

            chunks = []
            current_date = min_date
            end_date = max_date

            while current_date <= end_date:
                next_date = (pd.to_datetime(current_date) +
                             pd.DateOffset(months=3))
                chunk_query = (
                    f"SELECT * FROM {table} "
                    f"WHERE datetime >= '{current_date}' "
                    f"AND datetime < '{next_date}' "
                    f"AND Trade_Signal IS NOT NULL ORDER BY datetime"
                )
                try:
                    chunk = pd.read_sql(chunk_query, conn)
                except Exception as e:
                    print(f"⚠️ 以 {used_encoding} 讀取 chunk 失敗：{e}，嘗試其他編碼")
                    chunk, new_conn, new_enc = try_read_sql(chunk_query)
                    if new_conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = new_conn
                        used_encoding = new_enc

                chunks.append(chunk)
                print(f"📦 {current_date} 至 {next_date}：{len(chunk):,} 筆")
                current_date = next_date

            df = (pd.concat(chunks, ignore_index=True)
                  if chunks
                  else pd.DataFrame())
            print(f"✅ 共讀取 {len(df):,} 筆資料 (used encoding: {used_encoding})")

        try:
            if df.empty:
                print(f"⚠️ 資料表 {table} 雖可連線，但查無符合條件資料。")
                conn.close()
                return pd.DataFrame()

            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'],
                                                errors='coerce')
            df = df.sort_values('datetime').reset_index(drop=True)

            print(f"✅ 最終 DataFrame 成功建立，共 {len(df):,} 筆。")
            conn.close()
            return df

        except Exception as e:
            print(f"❌ 處理 DataFrame 時發生錯誤: {e}")
            try:
                conn.close()
            except Exception:
                pass
            return pd.DataFrame()

        if df.empty:
            print(f"⚠️ 資料表 {table} 雖可連線，但查無符合條件資料。")
            return pd.DataFrame()

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.sort_values('datetime').reset_index(drop=True)

        print(f"✅ 最終 DataFrame 成功建立，共 {len(df):,} 筆。")
        return df

    except Exception as e:
        print(f"❌ 讀取資料時發生錯誤: {str(e)}", flush=True)
        return pd.DataFrame()


def get_previous_stock_records_by_date(server, database, user, password,
                                       symbol, target_date,
                                       table="stock_data_1d"):
    """取得指定股票在指定日期之前的最新一筆價格資料"""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={user};PWD={password};"
        f"Trusted_Connection=no;Connection Timeout=30;"
        f"Application Name=TechnicalAnalysis"
    )

    query = f"""
        SELECT  *
        FROM {table}
        WHERE symbol = ? AND datetime < ?
        ORDER BY datetime DESC
    """

    try:
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(query, conn, params=[symbol, target_date])
            if df.empty:
                print(f"查無 {symbol} 在 {target_date} 之前的資料")
                return []

            candlesticks = []
            for i in range(len(df)):
                candlesticks.append({
                    "date": str(df.loc[i, "datetime"]),
                    "open": float(df.loc[i, "open_price"]),
                    "high": float(df.loc[i, "high_price"]),
                    "low": float(df.loc[i, "low_price"]),
                    "close": float(df.loc[i, "close_price"]),
                    "volume": (float(df.loc[i, "volume"])
                               if "volume" in df.columns else 0.0)
                })

            # 技術指標
            rsi_5 = df["rsi_5"].tolist()
            rsi_7 = df["rsi_7"].tolist()
            rsi_10 = df["rsi_10"].tolist()
            rsi_14 = df["rsi_14"].tolist()
            rsi_21 = df["rsi_21"].tolist()
            macd = df["macd"].tolist()
            dif = df["dif"].tolist()
            macd_histogram = df["macd_histogram"].tolist()
            rsv = df["rsv"].tolist()
            k_value = df["k_value"].tolist()
            d_value = df["d_value"].tolist()
            j_value = df["j_value"].tolist()
            ma5 = df["ma5"].tolist()
            ma10 = df["ma10"].tolist()
            ma20 = df["ma20"].tolist()
            ma60 = df["ma60"].tolist()
            ema12 = df["ema12"].tolist()
            ema26 = df["ema26"].tolist()
            bb_upper = df["bb_upper"].tolist()
            bb_middle = df["bb_middle"].tolist()
            bb_lower = df["bb_lower"].tolist()
            atr = df["atr"].tolist()
            cci = df["cci"].tolist()
            willr = df["willr"].tolist()
            mom = df["mom"].tolist()

            technical_indicator = {
                "rsi_5": rsi_5,
                "rsi_7": rsi_7,
                "rsi_10": rsi_10,
                "rsi_14": rsi_14,
                "rsi_21": rsi_21,
                "macd": macd,
                "dif": dif,
                "macd_histogram": macd_histogram,
                "rsv": rsv,
                "k_value": k_value,
                "d_value": d_value,
                "j_value": j_value,
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,
                "ma60": ma60,
                "ema12": ema12,
                "ema26": ema26,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "atr": atr,
                "cci": cci,
                "willr": willr,
                "mom": mom,
            }

            return {"candlesticks": candlesticks,
                    "technical_indicator": technical_indicator}

    except Exception as e:
        print(f"讀取資料時發生錯誤: {str(e)}")
        raise Exception(e)


def get_after_stock_records_by_date(server, database, user, password, symbol,
                                    target_date, table="stock_data_1d"):
    """取得指定股票在指定日期之後的第一筆價格資料"""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};DATABASE={database};UID={user};PWD={password};"
        f"Trusted_Connection=no;Connection Timeout=30;"
        f"Application Name=TechnicalAnalysis"
    )

    query = f"""
        SELECT  *
        FROM {table}
        WHERE symbol = ? AND datetime > ?
        ORDER BY datetime ASC
    """

    try:
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(query, conn, params=[symbol, target_date])
            if df.empty:
                print(f"查無 {symbol} 在 {target_date} 之後的資料")
                return []

            candlesticks = []
            for i in range(len(df)):
                candlesticks.append({
                    "date": str(df.loc[i, "datetime"]),
                    "open": float(df.loc[i, "open_price"]),
                    "high": float(df.loc[i, "high_price"]),
                    "low": float(df.loc[i, "low_price"]),
                    "close": float(df.loc[i, "close_price"]),
                    "volume": (float(df.loc[i, "volume"])
                               if "volume" in df.columns else 0.0)
                })

            return {"candlesticks": candlesticks}

    except Exception as e:
        print(f"讀取資料時發生錯誤: {str(e)}")
        return {}
