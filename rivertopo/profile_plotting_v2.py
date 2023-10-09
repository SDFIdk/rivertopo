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


"""
This script is used to plot the .npz files found in tests\data 

"""

def main():
    app = dash.Dash(__name__)
    # argument_parser = argparse.ArgumentParser()
    # #argument parser for the 3 examples without cross section burning
    # argument_parser.add_argument('ex_profiles1', type= str)
    # argument_parser.add_argument('ex_profiles2', type= str)
    # argument_parser.add_argument('ex_profiles3', type= str)
    # argument_parser.add_argument('ex_profiles4', type= str)

    # #argument parser for the 1 example with cross section burning and without
    # argument_parser.add_argument('ex_profiles4_line1', type= str)
    # argument_parser.add_argument('ex_profiles4_line2', type= str)

    # input_arguments = argument_parser.parse_args()

    # ex_profiles_path1 = input_arguments.ex_profiles1
    # ex_profiles_path2 = input_arguments.ex_profiles2
    # ex_profiles_path3 = input_arguments.ex_profiles3
    # ex_profiles_path4 = input_arguments.ex_profiles4
    # ex_profiles_path4_line1 = input_arguments.ex_profiles4_line1
    # ex_profiles_path4_line2 = input_arguments.ex_profiles4_line2
    
    # def get_file_label(file_path):
    #     return os.path.basename(file_path).replace('.csv', '')

    datafolder = os.path.relpath(r'tests\data')

    skive_data = np.load(os.path.join(datafolder,"skive.npz"))
    fiskbaek_data = np.load(os.path.join(datafolder,"fiskbaek.npz"))
    karup_data = np.load(os.path.join(datafolder,"karup.npz"))

    # Convert the npz data to pandas DataFrame
    skive_df = pd.DataFrame({
        'Line_ID': skive_data['line_ids'],
        'X': skive_data['x_coords'],
        'Y': skive_data['y_coords'],
        'Z': skive_data['z_values'],
        'Distance': skive_data['distances']
    })

    fiskbaek_df = pd.DataFrame({
        'Line_ID': fiskbaek_data['line_ids'],
        'X': fiskbaek_data['x_coords'],
        'Y': fiskbaek_data['y_coords'],
        'Z': fiskbaek_data['z_values'],
        'Distance': fiskbaek_data['distances']
    })

    karup_df = pd.DataFrame({
        'Line_ID': karup_data['line_ids'],
        'X': karup_data['x_coords'],
        'Y': karup_data['y_coords'],
        'Z': karup_data['z_values'],
        'Distance': karup_data['distances']
    })

    app.layout = html.Div([
        html.H1("Tværsnitsdata henover vandløbsmidte"),
        dcc.Store(id='selected-lines', data=[]),
        dcc.Dropdown(
            id='csv-dropdown',
            options=[
                {'label': 'Skive', 'value': 'ex_profiles1'},
                {'label': 'Karup', 'value': 'ex_profiles2'},
                {'label': 'Fiskbaek', 'value': 'ex_profiles3'},
                #{'label': get_file_label(ex_profiles_path4), 'value': 'ex_profiles4'},
                #{'label': get_file_label(ex_profiles_path4_line1), 'value': 'ex_profiles4_line1'},
            ],
            value='ex_profiles1',
            clearable=False
        ),
        html.Div([
            dcc.Graph(id='map', style={'height': '90vh'}),
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='dynamic-graph', style={'height': '60vh'}),
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ])

    
    @app.callback(
        Output('map', 'figure'),
        Input('csv-dropdown', 'value')
    )
    def update_map_figure(selected_csv):
        # select the correct path based on the dropdown
        if selected_csv == 'ex_profiles1':
            profile_path = skive_df
        elif selected_csv == 'ex_profiles2':
            profile_path = karup_df
        elif selected_csv == 'ex_profiles3':
            profile_path = fiskbaek_df
        # elif selected_csv == 'ex_profiles4':
        #     profile_path = ex_profiles_path4
        # elif selected_csv == 'ex_profiles4_line1':
        #     profile_path = ex_profiles_path4_line1
        else:
            raise ValueError(f"Unknown CSV file: {selected_csv}")
       
        # Load the profile data csv
        profile = profile_path
                  
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
        if hoverData is None:
            return go.Figure()
        # Load the profile data based on dropdown value
        # if selected_csv in ['ex_profiles4_line1', 'ex_profiles4_line2']:
        #     profile1 = pd.read_csv(ex_profiles_path4_line1)
        #     profile2 = pd.read_csv(ex_profiles_path4_line2)
        if selected_csv == 'ex_profiles1':
            profile1 = skive_df
            profile2 = None
        elif selected_csv == 'ex_profiles2':
            profile1 = karup_df
            profile2 = None
        elif selected_csv == 'ex_profiles3':
            profile1 = fiskbaek_df
            profile2 = None
        # elif selected_csv == 'ex_profiles4':
        #     profile1 = pd.read_csv(ex_profiles_path4)
        #     profile2 = None
        
        line_id = hoverData['points'][0]['customdata']
        df_line1 = profile1[profile1['Line_ID'] == line_id].copy()
        df_line2 = profile2[profile2['Line_ID'] == line_id].copy() if profile2 is not None else None

        fig = go.Figure()

        # Calculate the new X values so that the center of the segment is 0
        x_mean = df_line1['Distance'].mean()
        x_range = df_line1['Distance'].max() - df_line1['Distance'].min()
        scaling_factor = 30 / x_range
        df_line1['X_adjusted'] = (df_line1['Distance'] - x_mean) * scaling_factor

        df_line1.loc[:,'Z_diff'] = df_line1['Z'].diff().fillna(0).cumsum()
    
        fig.add_trace(
            go.Scatter(x=df_line1['X_adjusted'], y=df_line1['Z_diff'], mode='lines', name=str("Tværsnit udfra DHM"),line=dict(color='lightblue'))
        )

        if df_line2 is not None:
            # Calculate the new X values so that the center of the segment is 0
            x_mean = df_line2['X'].mean()
            x_range = df_line2['X'].max() - df_line2['X'].min()
            scaling_factor = 30 / x_range
            df_line2['X_adjusted'] = (df_line2['X'] - x_mean) * scaling_factor
            df_line2.loc[:,'Z_diff'] = df_line2['Z'].diff().fillna(0).cumsum()
            fig.add_trace(
                go.Scatter(x=df_line2['X_adjusted'], y=df_line2['Z_diff'], mode='lines', name=str("Tværsnit efter indbrænding"), line=dict(color='grey', dash='dash'))
            )
       
        # Add titles to the axes
        fig.update_xaxes(title_text='[m]', tick0=-15, dtick=4)
        fig.update_yaxes(title_text='[m]')
        fig.update_layout(title= 'Tværsnits profil:')
        
        return fig

    return app

if __name__ == '__main__':
    app = main()
    app.run_server(debug=True)
