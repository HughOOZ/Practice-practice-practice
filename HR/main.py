import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('D:\Practice-practice-practice\HR\HR.csv')
    sl_s = df['satisfaction_level']
    sl_s = sl_s.dropna()
    print(len(sl_s))

 