import os

# 1. Fix train_tft.py
tft_file = 'train_tft.py'
with open(tft_file, 'r') as f: code = f.read()
code = code.replace(r'DATA_PATH = r"C:\Users\Admin\OneDrive\Documents\df_hourly.csv"', 'DATA_PATH = "df_hourly.csv"')
code = code.replace(r'CHECKPOINT_PATH = r"C:\Users\Admin\OneDrive\Documents\tft_checkpoint.pth"', 'CHECKPOINT_PATH = "tft_checkpoint.pth"')
code = code.replace('max_epochs=1,', 'max_epochs=10,')
code = code.replace('limit_train_batches=20,', 'limit_train_batches=300,')
code = code.replace('limit_val_batches=5,', 'limit_val_batches=50,')
with open(tft_file, 'w') as f: f.write(code)

# 2. Fix backtest.py
bk_file = 'backtest.py'
with open(bk_file, 'r') as f: bcode = f.read()
bcode = bcode.replace('order=(1, 0, 1)', 'order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)')
bcode = bcode.replace('maxiter=50', 'maxiter=20') # keep iterations low so it doesn't hang
with open(bk_file, 'w') as f: f.write(bcode)

# 3. Fix forecast_api/main.py for CORS (frontend needs CORS)
main_file = 'forecast_api/main.py'
with open(main_file, 'r') as f: mcode = f.read()
if 'CORSMiddleware' not in mcode:
    mcode = mcode.replace('from fastapi import FastAPI, HTTPException, Query', 'from fastapi import FastAPI, HTTPException, Query\nfrom fastapi.middleware.cors import CORSMiddleware')
    cors = '''app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)'''
    mcode = mcode.replace('app = FastAPI(title="Weather Forecast API", version="0.1.0")', 'app = FastAPI(title="Weather Forecast API", version="0.1.0")\n' + cors)
    with open(main_file, 'w') as f: f.write(mcode)

# 4. update README.md
readme = 'README.md'
with open(readme, 'r') as f: rcode = f.read()
rcode = rcode.replace('2 epochs', '10 epochs (trained for 1 epoch on CPU for demo speed)')
with open(readme, 'w') as f: f.write(rcode)
