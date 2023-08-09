import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from osgeo import gdal, ogr
import argparse
from shapely.geometry import Point
import geopandas as gpd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output



def main():
    app = dash.Dash(__name__)

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('ex_profiles', type= str)
    argument_parser.add_argument('ex_profiles2', type=str, help= 'input line-object vector data source')

    argument_parser.add_argument('raster', type= str)
 
    input_arguments = argument_parser.parse_args()

    ex_profiles_path = input_arguments.ex_profiles
    ex_profiles_path2 = input_arguments.ex_profiles2
    raster_path = input_arguments.raster
 
    raster_dataset = gdal.Open(raster_path)
    band = raster_dataset.GetRasterBand(1)
    band_array = band.ReadAsArray()
    new_band_array = np.where(band_array == -9999, np.nan, band_array)

    # load the profile data csv
    profile = pd.read_csv(ex_profiles_path)
    profile2 = pd.read_csv(ex_profiles_path2)

    # Get unique Line_IDs
    line_ids = profile['Line_ID'].unique()

    subplot_titles = [f"Line ID: {line_id}" for line_id in line_ids]
    fig1 = make_subplots(rows=13, cols=3, subplot_titles=subplot_titles)

    # # add a line plot for each Line_ID to the subplot layout
    # for i, line_id in enumerate(line_ids):
    #     df_line = profile[profile['Line_ID'] == line_id]
    #     df_line2 = profile2[profile2['Line_ID'] == line_id]
    #     fig1.add_trace(
    #         go.Scatter(x=df_line['X'], y=df_line['Z'], mode='lines', name=str(line_id),line=dict(color='lightblue'), showlegend=False),
    #         row=i//3 + 1,  # Determine row index
    #         col=i%3 + 1   # Determine column index
    #     )
        
    #     fig1.add_trace(
    #         go.Scatter(x=df_line2['X'], y=df_line2['Z'], mode='lines', name=str(line_id), line=dict(color='grey',dash='dash'), showlegend=False),
    #         row=i//3 + 1,  # Determine row index
    #         col=i%3 + 1   # Determine column index
    #     )
    
    # Update x axes visibility
    for i in fig1['layout']:
        if 'xaxis' in i:
            fig1['layout'][i]['showticklabels'] = False

    fig1.update_yaxes(tickfont=dict(size=8))

    fig1.update_layout(title='Tværsnit 2d graf',
    annotations=[
        dict(
            text=f"Line ID: {line_id}", 
            showarrow=False,
            font=dict(
                size=9  
            )
        ) for line_id in line_ids
    ],
)
    #fig.show()
        
    # Create a GeoDataFrame 
    geometry = [Point(xy) for xy in zip(profile['X'], profile['Y'])]
    gdf = gpd.GeoDataFrame(profile, geometry=geometry)

    # Specify the original CRS
    gdf.crs = 'EPSG:25832'  # UTM zone 32N

    # Convert to the CRS to EPSG:4326 (WGS84)
    gdf = gdf.to_crs('EPSG:4326')

    # Add latitude and longitude columns to the dataframe
    gdf['latitude'] = gdf.geometry.y
    gdf['longitude'] = gdf.geometry.x
        
    # Mapbox access token
    px.set_mapbox_access_token("mapbox_token") # replace with your mapbox token

    fig2 = go.Figure(go.Scattermapbox(
        lat=gdf['latitude'],
        lon=gdf['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=2
        ),
        text=gdf['Line_ID'],
        customdata=gdf['Line_ID'],
    ))

    fig2.update_layout(title='Kortlægning af tværsnit',
        autosize=True,
        hovermode='closest',
        mapbox=dict( 
            bearing=0,
            center=dict(
                lat=55.2639,  # Center latitude 
                lon=12.1701   # Center longitude
            ),
            pitch=0,
            zoom=16,
            style='open-street-map'  # OpenStreetMap view
        ),
    )
    
    downsampled_array = new_band_array[::10, ::10]
    fig3 = go.Figure(data=[go.Surface(z=new_band_array, colorscale='Viridis', opacity=0.5)])
    
    fig3.update_layout(title='Raster data over vandløb', autosize=False,
                  width=800, height=500,
                  #scene=dict(zaxis=dict(range=[50,10])),
                  margin=dict(l=65, r=50, b=65, t=90))

    fig3.update_layout(scene_aspectmode='manual',scene_aspectratio=dict(x=10, y=8, z = 1))

    app.layout = html.Div([
        html.H1("Tværsnitsdata henover vandløbsmidte"),
        html.Div([
            dcc.Graph(id='map', figure=fig2, style={'height': '90vh'}),
        ], style={'width': '49%', 'display': 'inline-block'}),

         html.Div([
            dcc.Graph(id='dynamic-graph', style={'height': '40vh'}),
            #dcc.Graph(id='3dmap', figure=fig3, style={'height': '30vh'}),
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ])

    @app.callback(
        Output('dynamic-graph', 'figure'),
        Input('map', 'hoverData'))
    def update_graph(hoverData):
        line_id = hoverData['points'][0]['customdata']  # assumes 'Line_ID' is in 'customdata'
        df_line = profile[profile['Line_ID'] == line_id]
        df_line2 = profile2[profile2['Line_ID'] == line_id]
        fig = go.Figure()
        # Calculate the difference between current and previous 'X' value for both dataframes
        df_line['X_diff'] = df_line['X'].diff().fillna(0).cumsum()
        df_line2['X_diff'] = df_line2['X'].diff().fillna(0).cumsum()
        df_line['Z_diff'] = df_line['Z'].diff().fillna(0).cumsum()
        df_line2['Z_diff'] = df_line2['Z'].diff().fillna(0).cumsum()

        fig.add_trace(
            go.Scatter(x=df_line['X_diff'], y=df_line['Z_diff'], mode='lines', name=str("Tværsnit efter indbrænding"),line=dict(color='lightblue'))
        )
        fig.add_trace(
            go.Scatter(x=df_line2['X_diff'], y=df_line2['Z_diff'], mode='lines', name=str("Tværsnit før indbrænding"), line=dict(color='grey',dash='dash'))
        )
        # Add titles to the axes
        fig.update_xaxes(title_text='[m]')
        fig.update_yaxes(title_text='[m]')
        
        return fig
    return app

if __name__ == '__main__':
    app = main()
    app.run_server(debug=True)
    #main()