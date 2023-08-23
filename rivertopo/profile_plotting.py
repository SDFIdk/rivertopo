import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from osgeo import gdal, ogr
import argparse
from shapely.geometry import Point
import geopandas as gpd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import os

def main():
    app = dash.Dash(__name__)
    argument_parser = argparse.ArgumentParser()
    #argument parser for the 3 examples without cross section burning
    argument_parser.add_argument('ex_profiles1', type= str)
    argument_parser.add_argument('ex_profiles2', type= str)
    argument_parser.add_argument('ex_profiles3', type= str)

    #argument parser for the 1 example with cross section burning and without
    argument_parser.add_argument('ex_profiles4_line1', type= str)
    argument_parser.add_argument('ex_profiles4_line2', type= str)

    input_arguments = argument_parser.parse_args()

    ex_profiles_path1 = input_arguments.ex_profiles1
    ex_profiles_path2 = input_arguments.ex_profiles2
    ex_profiles_path3 = input_arguments.ex_profiles3
    ex_profiles_path4_line1 = input_arguments.ex_profiles4_line1
    ex_profiles_path4_line2 = input_arguments.ex_profiles4_line2
    
    def get_file_label(file_path):
        return os.path.basename(file_path).replace('.csv', '')

    app.layout = html.Div([
        html.H1("Tværsnitsdata henover vandløbsmidte"),
        dcc.Dropdown(
            id='csv-dropdown',
            options=[
                {'label': get_file_label(ex_profiles_path1), 'value': 'ex_profiles1'},
                {'label': get_file_label(ex_profiles_path2), 'value': 'ex_profiles2'},
                {'label': get_file_label(ex_profiles_path3), 'value': 'ex_profiles3'},
                {'label': get_file_label(ex_profiles_path4_line1), 'value': 'ex_profiles4_line1'},
            ],
            value='ex_profiles1',
            clearable=False
        ),
        html.Div([
            dcc.Graph(id='map', style={'height': '90vh'}),
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='dynamic-graph', style={'height': '40vh'}),
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ])

    @app.callback(
        Output('map', 'figure'),
        Input('csv-dropdown', 'value')
    )
    def update_map_figure(selected_csv):
        # select the correct path based on the dropdown
        if selected_csv == 'ex_profiles1':
            profile_path = ex_profiles_path1
        elif selected_csv == 'ex_profiles2':
            profile_path = ex_profiles_path2
        elif selected_csv == 'ex_profiles3':
            profile_path = ex_profiles_path3
        elif selected_csv == 'ex_profiles4_line1':
            profile_path = ex_profiles_path4_line1
        else:
            raise ValueError(f"Unknown CSV file: {selected_csv}")
       
        # Load the profile data csv
        profile = pd.read_csv(profile_path)
                  
        # Get unique Line_IDs
        line_ids = profile['Line_ID'].unique()

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
        # Compute mean latitude and longitude
        mean_latitude = gdf['latitude'].mean()
        mean_longitude = gdf['longitude'].mean()

        # Mapbox access token
        px.set_mapbox_access_token("mapbox_token") #replace with mapbox token 

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
                    lat= mean_latitude,  # Center latitude 
                    lon= mean_longitude  # Center longitude
                ),
                pitch=0,
                zoom=12,
                style='open-street-map'  # OpenStreetMap view
            ),
        )
        
        return fig2

    @app.callback(
        Output('dynamic-graph', 'figure'),
        [Input('map', 'hoverData'),
         Input('csv-dropdown', 'value')]
    )
    def update_graph(hoverData, selected_csv):
        # Load the profile data based on dropdown value
        if selected_csv in ['ex_profiles4_line1', 'ex_profiles4_line2']:
            profile1 = pd.read_csv(ex_profiles_path4_line1)
            profile2 = pd.read_csv(ex_profiles_path4_line2)
        elif selected_csv == 'ex_profiles1':
            profile1 = pd.read_csv(ex_profiles_path1)
            profile2 = None
        elif selected_csv == 'ex_profiles2':
            profile1 = pd.read_csv(ex_profiles_path2)
            profile2 = None
        elif selected_csv == 'ex_profiles3':
            profile1 = pd.read_csv(ex_profiles_path3)
            profile2 = None
        
        line_id = hoverData['points'][0]['customdata']
        df_line1 = profile1[profile1['Line_ID'] == line_id]
        df_line2 = profile2[profile2['Line_ID'] == line_id] if profile2 is not None else None
    
        #df_line = profile[profile['Line_ID'] == line_id]
        fig = go.Figure()

        # Calculate the difference between current and previous 'X' value for both dataframes
        df_line1['X_diff'] = df_line1['X'].diff().fillna(0).cumsum()

        df_line1['Z_diff'] = df_line1['Z'].diff().fillna(0).cumsum()
    
        fig.add_trace(
            go.Scatter(x=df_line1['X_diff'], y=df_line1['Z_diff'], mode='lines', name=str("Tværsnit udfra DHM"),line=dict(color='lightblue'))
        )

        if df_line2 is not None:
            df_line2['X_diff'] = df_line2['X'].diff().fillna(0).cumsum()
            df_line2['Z_diff'] = df_line2['Z'].diff().fillna(0).cumsum()
            fig.add_trace(
                go.Scatter(x=df_line2['X_diff'], y=df_line2['Z_diff'], mode='lines', name=str("Tværsnit efter indbrænding"), line=dict(color='grey', dash='dash'))
            )
       
        # Add titles to the axes
        fig.update_xaxes(title_text='[m]')
        fig.update_yaxes(title_text='[m]')
        
        return fig

    return app

if __name__ == '__main__':
    app = main()
    app.run_server(debug=True)