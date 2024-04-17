import pandas as pd
import h5py
import pyecharts.options as opts
from pyecharts.charts import Kline,Line,Page
import config
from data_processing import load_Tapes_data_by_date, load_all_Tapes

# interval, 'T' is minute, 'H' is hour, 'D' is day 
def process_data_for_kline(tapes_df,interval='D'):
    tapes_df['timestamp'] = pd.to_datetime(tapes_df['timestamp'], unit='s')
    tapes_df.set_index('timestamp', inplace=True)
    ohlc = tapes_df['price'].resample(interval).ohlc()
    ohlc = ohlc[['open',  'close','low','high']]
    ohlc.dropna(inplace=True)  
    return ohlc

def plot_kline_chart(ohlc):
    kline = Kline()
    kline.add_xaxis(ohlc.index.strftime('%Y-%m-%d %H:%M').tolist())
    kline.add_yaxis("price", ohlc.values.tolist(),
                    itemstyle_opts=opts.ItemStyleOpts(
                        color="#ec0000",
                        color0="#00da3c"
                    ))
    kline.set_global_opts(
        title_opts=opts.TitleOpts(title="KLine"),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(is_scale=True),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        datazoom_opts=[opts.DataZoomOpts()],
    )
    return kline

def plot_line_chart(ohlc):
    line = Line()
    line.add_xaxis(ohlc.index.strftime('%Y-%m-%d %H:%M').tolist())
    line.add_yaxis("Close Price", ohlc['close'].tolist(),
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.5))
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Close Price Trend"),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(is_scale=True),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts()],
    )
    return line



tapes_df = load_all_Tapes(config.AllTapes_hdf5_path)

if tapes_df is not None:
    ohlc_data = process_data_for_kline(tapes_df)
    print(ohlc_data)
    kline_chart = plot_kline_chart(ohlc_data)
    kline_chart.render("kline_chart.html")  
    line_chart = plot_line_chart(ohlc_data)
    line_chart.render("line_chart.html")
    page = Page(layout=Page.SimplePageLayout)
    page.add(kline_chart)
    page.add(line_chart)
    page.render("combined_charts.html")
    print("Combined chart page rendered successfully.")

else:
    print("No data available for the specified date.")
