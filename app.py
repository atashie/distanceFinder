import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic

# Set page configuration
st.set_page_config(
    page_title="Distance Finder Dashboard",
    page_icon="📍",
    layout="wide"
)

st.title('📍 Distance Finder Dashboard')
st.markdown('**Two-stage search:** Simple buffer filtering → Accurate network analysis')

# Initialize session state for results caching
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Sidebar for search parameters
with st.sidebar:
    st.header("Search Parameters")
    
    # Location input
    center_lat = st.number_input("Center Latitude", value=40.7128, format="%.4f")
    center_lon = st.number_input("Center Longitude", value=-74.0060, format="%.4f")
    
    # Search radius
    search_radius_km = st.slider("Search Radius (km)", 1, 50, 10)
    
    # Network distance threshold (smaller than search radius)
    network_threshold_km = st.slider(
        "Accurate Distance Threshold (km)", 
        1, 
        search_radius_km, 
        min(5, search_radius_km),
        help="Maximum network distance for final results (must be ≤ search radius)"
    )
    
    st.info(f"""
    **Two-Stage Search Process:**
    1. **Stage 1:** Find all locations within {search_radius_km}km buffer
    2. **Stage 2:** Calculate accurate network distances for subset
    """)

# Two-stage search function
@st.cache_data(show_spinner=False)
def two_stage_location_search(center_lat, center_lon, buffer_radius_km, network_threshold_km):
    """
    Performs efficient two-stage search:
    Stage 1: Simple geometric buffer to subset data
    Stage 2: Accurate network analysis on subset only
    """
    
    # Create center point
    center_point = Point(center_lon, center_lat)
    
    # STAGE 1: Simple buffer search
    with st.spinner('Stage 1: Applying geometric buffer...'):
        # Convert radius to degrees (approximate)
        buffer_degrees = buffer_radius_km / 111.0  # 1 degree ≈ 111 km
        
        # Create buffer polygon
        buffer_polygon = center_point.buffer(buffer_degrees)
        
        # Generate sample locations for demo (in real app, load your data here)
        np.random.seed(42)
        n_locations = 1000
        
        # Generate random points around center
        angles = np.random.uniform(0, 2*np.pi, n_locations)
        distances = np.random.uniform(0, buffer_radius_km * 2, n_locations)
        
        lats = center_lat + (distances / 111.0) * np.sin(angles)
        lons = center_lon + (distances / 111.0) * np.cos(angles)
        
        locations_df = pd.DataFrame({
            'id': range(n_locations),
            'name': [f'Location {i}' for i in range(n_locations)],
            'lat': lats,
            'lon': lons
        })
        
        # Create GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(locations_df['lon'], locations_df['lat'])]
        locations_gdf = gpd.GeoDataFrame(locations_df, geometry=geometry)
        
        # Apply buffer filter
        within_buffer = locations_gdf[locations_gdf.geometry.within(buffer_polygon)]
        
        stage1_count = len(within_buffer)
        st.success(f"Stage 1 complete: {stage1_count} locations within {buffer_radius_km}km buffer")
    
    # STAGE 2: Network analysis on subset
    with st.spinner(f'Stage 2: Calculating network distances for {stage1_count} locations...'):
        try:
            # Download street network for the area
            G = ox.graph_from_point(
                (center_lat, center_lon), 
                dist=buffer_radius_km * 1000,
                network_type='drive'
            )
            
            # Get nearest network node to center
            center_node = ox.nearest_nodes(G, center_lon, center_lat)
            
            # Calculate network distances for subset only
            network_results = []
            
            for idx, row in within_buffer.iterrows():
                try:
                    # Find nearest node for this location
                    dest_node = ox.nearest_nodes(G, row['lon'], row['lat'])
                    
                    # Calculate shortest path
                    distance_m = nx.shortest_path_length(
                        G, center_node, dest_node, weight='length'
                    )
                    distance_km = distance_m / 1000
                    
                    # Only include if within network threshold
                    if distance_km <= network_threshold_km:
                        network_results.append({
                            'id': row['id'],
                            'name': row['name'],
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'buffer_distance_km': geodesic(
                                (center_lat, center_lon), 
                                (row['lat'], row['lon'])
                            ).km,
                            'network_distance_km': distance_km,
                            'distance_difference': distance_km - geodesic(
                                (center_lat, center_lon), 
                                (row['lat'], row['lon'])
                            ).km
                        })
                except:
                    # Skip if network path not found
                    continue
            
            results_df = pd.DataFrame(network_results)
            results_df = results_df.sort_values('network_distance_km')
            
            st.success(f"Stage 2 complete: {len(results_df)} locations within {network_threshold_km}km network distance")
            
            return results_df, within_buffer, G
            
        except Exception as e:
            st.error(f"Network analysis error: {str(e)}")
            # Fallback to simple distance
            within_buffer['network_distance_km'] = within_buffer.apply(
                lambda row: geodesic(
                    (center_lat, center_lon), 
                    (row['lat'], row['lon'])
                ).km, 
                axis=1
            )
            within_buffer = within_buffer[within_buffer['network_distance_km'] <= network_threshold_km]
            return within_buffer, within_buffer, None

# Main search button
if st.button("🔍 Run Two-Stage Search", type="primary"):
    results_df, stage1_subset, network_graph = two_stage_location_search(
        center_lat, center_lon, search_radius_km, network_threshold_km
    )
    st.session_state.search_results = (results_df, stage1_subset, network_graph)

# Display results
if st.session_state.search_results:
    results_df, stage1_subset, network_graph = st.session_state.search_results
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Locations", 1000)
    with col2:
        st.metric("After Stage 1 (Buffer)", len(stage1_subset))
    with col3:
        st.metric("After Stage 2 (Network)", len(results_df))
    with col4:
        if len(results_df) > 0:
            efficiency = (1 - len(results_df)/1000) * 100
            st.metric("Computation Saved", f"{efficiency:.1f}%")
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Results Table", "🗺️ Map View", "📈 Analysis"])
    
    with tab1:
        if not results_df.empty:
            st.subheader("Locations Within Network Distance")
            
            # Format the dataframe for display
            display_df = results_df[[
                'name', 'buffer_distance_km', 'network_distance_km', 'distance_difference'
            ]].round(2)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "distance_finder_results.csv",
                "text/csv"
            )
    
    with tab2:
        st.subheader("Interactive Map")
        
        # Create folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add center marker
        folium.Marker(
            [center_lat, center_lon],
            popup="Search Center",
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)
        
        # Add buffer circle
        folium.Circle(
            location=[center_lat, center_lon],
            radius=search_radius_km * 1000,
            popup=f"Stage 1 Buffer: {search_radius_km}km",
            color='blue',
            fill=True,
            fillOpacity=0.1
        ).add_to(m)
        
        # Add network threshold circle
        folium.Circle(
            location=[center_lat, center_lon],
            radius=network_threshold_km * 1000,
            popup=f"Network Threshold: {network_threshold_km}km",
            color='green',
            fill=True,
            fillOpacity=0.1
        ).add_to(m)
        
        # Add result markers
        if not results_df.empty:
            for idx, row in results_df.iterrows():
                folium.Marker(
                    [row['lat'], row['lon']],
                    popup=f"{row['name']}<br>Network: {row['network_distance_km']:.2f}km",
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)
        
        # Display map
        st_folium(m, height=500, use_container_width=True)
    
    with tab3:
        st.subheader("Search Efficiency Analysis")
        
        if not results_df.empty:
            # Efficiency metrics
            total_locations = 1000
            stage1_filtered = len(stage1_subset)
            stage2_filtered = len(results_df)
            
            efficiency_data = pd.DataFrame({
                'Stage': ['Original', 'After Buffer', 'After Network'],
                'Count': [total_locations, stage1_filtered, stage2_filtered],
                'Computation': ['100%', f'{(stage1_filtered/total_locations)*100:.1f}%', 
                               f'{(stage2_filtered/total_locations)*100:.1f}%']
            })
            
            # Bar chart
            st.bar_chart(efficiency_data.set_index('Stage')['Count'])
            
            # Distance comparison
            st.subheader("Distance Accuracy Improvement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_diff = results_df['distance_difference'].mean()
                st.metric(
                    "Average Distance Difference",
                    f"{abs(avg_diff):.2f} km",
                    help="Difference between straight-line and actual road distance"
                )
            
            with col2:
                max_diff = results_df['distance_difference'].abs().max()
                st.metric(
                    "Maximum Distance Difference",
                    f"{max_diff:.2f} km",
                    help="Largest difference found between methods"
                )
