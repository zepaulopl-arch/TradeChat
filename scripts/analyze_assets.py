import yaml
import pandas as pd
import os

config_path = r'c:\Users\zepau\ML\projetos\TradeChat\config\data.yaml'
cache_dir = r'c:\Users\zepau\ML\projetos\TradeChat\data_cache'

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

assets = config.get('assets', {})

low_data_assets = []
results = []

for ticker in assets.keys():
    file_ticker = ticker.replace('.', '_')
    file_path = os.path.join(cache_dir, f"prices_{file_ticker}.parquet")
    
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            row_count = len(df)
            results.append((ticker, row_count))
            if row_count < 150:
                low_data_assets.append((ticker, row_count))
        except Exception as e:
            low_data_assets.append((ticker, 0))
    else:
        low_data_assets.append((ticker, 0))

print("---RESULTS---")
for ticker, count in results:
    print(f"{ticker}: {count}")
print("---LOW DATA---")
for ticker, count in low_data_assets:
    print(f"{ticker}: {count}")
print("---END---")
