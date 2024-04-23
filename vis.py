
from pyecharts import options as opts
from pyecharts.charts import Line,Timeline
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
import pandas as pd
import plotly.graph_objects as go
import h5py
import os
import re
import config
from data_processing import load_LOBs_data_by_date, load_Tapes_data_by_date


def lob_depth_chart(asks_df,bids_df,tapes_data):
    asks_df_sorted = asks_df.sort_values(by='price', ascending=True)
    bids_df_sorted = bids_df.sort_values(by='price', ascending=False)

    line = Line(init_opts=opts.InitOpts(bg_color='black'))

    asks_df_sorted['cumulative'] = asks_df_sorted['quantity'].cumsum()
    bids_df_sorted['cumulative'] = bids_df_sorted['quantity'].cumsum()

    asks_df_sorted = pd.concat([
        pd.DataFrame({'price': [asks_df_sorted['price'].min()], 'cumulative': [0]}),
        asks_df_sorted,
        pd.DataFrame({'price': [asks_df_sorted['price'].max()], 'cumulative': [0]}),
        pd.DataFrame({'price': [asks_df_sorted['price'].min()], 'cumulative': [0]})
    ], ignore_index=True)

    bids_df_sorted = pd.concat([
        pd.DataFrame({'price': [bids_df_sorted['price'].max()], 'cumulative': [0]}),
        bids_df_sorted,
        pd.DataFrame({'price': [bids_df_sorted['price'].min()], 'cumulative': [0]}),
        pd.DataFrame({'price': [bids_df_sorted['price'].max()], 'cumulative': [0]})
    ], ignore_index=True)


    line.add_xaxis(xaxis_data=asks_df_sorted['price'].tolist())
    line.add_yaxis(
        series_name="Asks",
        y_axis=asks_df_sorted['cumulative'].tolist(),
        is_step=True,
        is_smooth=False,
        symbol='none',  
        linestyle_opts=opts.LineStyleOpts(color="#FF0000"),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.25,color="#FF0000")
        
    )

    line.add_xaxis(xaxis_data=bids_df_sorted['price'].tolist())
    line.add_yaxis(
        series_name="Bids",
        y_axis=bids_df_sorted['cumulative'].tolist(),
        is_step=True,
        is_smooth=False,
        symbol='none',  
        linestyle_opts=opts.LineStyleOpts(color="#00FF00"),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.25,color="#00FF00")
    )
    
    markline_opts = opts.MarkLineOpts(
        data=[{"xAxis": float(tapes_data)}],  # Assuming tape_data['price'] is correctly aligned with your xaxis
        symbol=['none', 'none'],
        linestyle_opts=opts.LineStyleOpts(color='rgba(255, 255, 255, 0.5)', width=2)
    )
    line.set_series_opts(markline_opts=markline_opts)

    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value", min_=bids_df['price'].min(), max_=asks_df['price'].max(),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        title_opts=opts.TitleOpts(title="LOB Depth Chart"),
        legend_opts=opts.LegendOpts(is_show=False),  
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
    )

        

    # line.render("lob_depth_chart.html")
    return line

                
if __name__=='__main__':
    
    lob_df=load_LOBs_data_by_date(config.LOBsT_hdf5_path,'2025-01-03')
    tapes_df=load_Tapes_data_by_date(config.Tapes_hdf5_path,'2025-01-03')
    timestamps = lob_df['timestamp'].unique()
    tapes_timestamps=tapes_df['timestamp'].unique()
    print(timestamps)
    print(tapes_timestamps)
    timeline = Timeline()
    for ts in timestamps:
        df_update = lob_df[lob_df['timestamp'] == ts]
        new_asks=df_update[df_update['type']=='Ask']
        new_bids=df_update[df_update['type']=='Bid']
        print(f'_____ {ts}_______')
        print(new_asks)
        print(new_bids)
        filtered_tapes = tapes_df[tapes_df['timestamp'] < ts]
        if not filtered_tapes.empty:
            idx = filtered_tapes['timestamp'].idxmax()
            maxtape = tapes_df.loc[idx, ['timestamp', 'price']]  
        else:
            maxtape = pd.DataFrame({'timestamp': [0], 'price': [0]})
        print(f'maxtape is {maxtape}')
        
        depthchart=lob_depth_chart(new_asks,new_bids,maxtape['price'])
        
        ts_datetime = pd.to_datetime(ts).to_pydatetime()
        timeline.add(depthchart, ts_datetime.strftime("%H:%M:%S"))

        if ts>3000:
            break
    timeline.add_schema(play_interval=100) # Play interval in milliseconds
    
    # Render to html file
    timeline.render("lob_depth_chart_with_timeline.html")