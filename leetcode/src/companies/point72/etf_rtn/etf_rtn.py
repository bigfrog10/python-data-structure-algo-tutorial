import pandas as pd
import pandasql as ps

df_c = pd.DataFrame([
    ['2017-01-03', 'IYZ', 'ATNI', 0.0338],
    ['2017-01-03', 'IYZ', 'CBB', 0.0348],
    ['2017-01-03', 'IYZ', 'CNSL', 0.0351],
    ['2017-01-03', 'IYZ', 'CTL', 0.0555],
    ['2017-01-03', 'IYZ', 'FTR', 0.0423],
    ['2017-01-03', 'IYZ', 'GNCMA', 0.0311],
    ['2017-01-03', 'IYZ', 'GSAT', 0.0488],
    ['2017-01-03', 'IYZ', 'IRDM', 0.0323],
    ['2017-01-03', 'IYZ', 'LVLT', 0.0587],
], columns=['TradeDate', 'ETFTicker', 'ConstituentTicker', 'ConstituentWeight'])

df_r = pd.DataFrame([
    ['2017-01-03', 'IYZ', 0.0443],
    ['2017-01-03', 'ATNI', 0.0457],
    ['2017-01-03', 'CBB', 0.0559],
    ['2017-01-03', 'CNSL', 0.0235],
    ['2017-01-03', 'CTL', 0.0660],
    ['2017-01-03', 'FTR', 0.0592]
], columns=['TradeDate', 'Ticker', 'StockReturn'])

sql = '''
with const as
( SELECT a.TradeDate, a.ETFTicker, a.ConstituentTicker, a.ConstituentWeight, 
         b.StockReturn, a.ConstituentWeight * b.StockReturn as wr
  FROM EtfConstituents as a, StockReturns as b
  WHERE a.TradeDate = b.TradeDate AND a.ConstituentTicker = b.Ticker
), 
calc as
( SELECT TradeDate, ETFTicker, SUM(wr) as CalculatedReturn
  FROM const
  GROUP BY TradeDate, ETFTicker
)
SELECT TradeDate, ETFTicker, StockReturn as ActualREturn, CalculatedReturn, StockReturn - CalculatedReturn as Difference
FROM StockReturns as a, calc as b
WHERE a.TradeDate = b.TradeDate AND a.ETFTicker = b.ETFTicker
'''
