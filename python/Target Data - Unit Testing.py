import pandas as pd
target_columns = ['Open','High','Low','Close','Vol','MarketCap','WMA_90', 'H-L', 'H-PC', 'L-PC', 'True_Range', 'ATR_15',
         'basic_ub_Fast', 'basic_lb_Fast', 'final_ub_Fast', 'final_lb_Fast',
         'Super_Trend_Fast', 'basic_ub_Slow',  'basic_lb_Slow', 'final_ub_Slow',
         'final_lb_Slow', 'Super_Trend_Slow', 'Bull1', 'Bull2', 'Bull3', 'Bull31',
         'Crs_Bulldiff', 'Crs_Bull', 'Direction', 'Bull5', 'Bullish', 'Bear1',
         'Bear2', 'Bear3', 'Bear31', 'Crs_Beardiff', 'Crs_Bear', 'Bear5', 'Bearish',
         'Signal']
column_names = target_columns
file_path = "/Users/michaelnew/Dropbox/Aptana_Workspace/Skyze/Unit_Test/Test_Data/Strategies/SuperTrend CrossOver Screener/Target-Results-SuperTrendCrossScreener-bitcoin.csv"
target_results = pd.read_csv(
                                            file_path,
                                            header=None ,
                                            names = column_names,
                                            index_col=False,
                                            skiprows = 1
                                       )
target_results.head(20)

type(target_results["Bear3"][15])



