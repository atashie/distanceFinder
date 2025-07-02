import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import osmnx as ox
import folium
import networkx as nx
from typing import List, Dict, Union, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 300

# Page config
st.set_page_config(
    page_title="Location Analyzer",
    page_icon="📍",
    layout="wide"
)

# Title and description
st.title("📍 Location Analyzer")
st.markdown("Find areas that meet multiple location criteria using progressive filtering")

# Define POI types for dropdown
POI_TYPES = {
    "Grocery Store": {'shop': 'supermarket'},
    "School": {'amenity': 'school'},
    "Park": {'leisure': 'park'},
    "Restaurant": {'amenity': 'restaurant'},
    "Coffee Shop": {'amenity': 'cafe'},
    "Hospital": {'amenity': 'hospital'},
    "Pharmacy": {'amenity': 'pharmacy'},
    "Library": {'amenity': 'library'},
    "Gym/Fitness": {'leisure': 'fitness_centre'},
    "Bus Stop": {'highway': 'bus_stop'},
    "Bank": {'amenity': 'bank'},
    "Bar": {'amenity': 'bar'},
    "Gas Station": {'amenity': 'fuel'},
    "Shopping Mall": {'shop': 'mall'},
    "Post Office": {'amenity': 'post_office'},
    "Fire Station": {'amenity': 'fire_station'},
    "Police Station": {'amenity': 'police'},
    "Playground": {'leisure': 'playground'},
    "Swimming Pool": {'leisure': 'swimming_pool'},
    "Museum": {'tourism': 'museum'}
}

# LocationAnalyzer class (from previous implementation)
class LocationAnalyzer:
    """Location analysis tool with progressive filtering and proper travel mode support."""
    
    TRAVEL_SPEEDS = {
        'walk': 3.0,
        'bike': 12.0,
        'drive': 55.0
    }
    
    BUFFER_ADJUSTMENTS = {
        'walk': 1.2,
        'bike': 1.3,
        'drive': 1.5
    }
    
    def __init__(self, center_location: str, max_radius_miles: float = 50):
        """Initialize with a center point and maximum search radius."""
        st.info(f"Initializing analyzer centered on: {center_location}")
        
        try:
            location_gdf = ox.geocode_to_gdf(center_location)
            self.center_point = location_gdf.geometry.iloc[0].centroid
            lat = self.center_point.y
            lon = self.center_point.x
        except:
            try:
                lat, lon = ox.geocode(center_location)
                self.center_point = Point(lon, lat)
            except:
                raise ValueError(f"Could not geocode location: {center_location}")
        
        self.center_location = center_location
        self.max_radius_miles = max_radius_miles
        self.crs = 'EPSG:4326'
        
        utm_crs = self._estimate_utm_crs(lon, lat)
        center_gdf = gpd.GeoDataFrame([{'geometry': self.center_point}], crs=self.crs)
        center_projected = center_gdf.to_crs(utm_crs)
        
        buffer_meters = max_radius_miles * 1609.34
        search_boundary = center_projected.buffer(buffer_meters)
        
        self.search_boundary = search_boundary.to_crs(self.crs).iloc[0]
        self.current_search_area = self.search_boundary
        
        self.criteria_results = []
        self.criteria_descriptions = []
        
        st.success(f"✅ Search area initialized: {max_radius_miles} mile radius")
    
    def _estimate_utm_crs(self, lon: float, lat: float) -> str:
        """Estimate appropriate UTM CRS for given coordinates."""
        utm_zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            return f'EPSG:326{utm_zone:02d}'
        else:
            return f'EPSG:327{utm_zone:02d}'
    
    def _calculate_area_sq_miles(self, geometry) -> float:
        """Calculate area in square miles."""
        utm_crs = self._estimate_utm_crs(self.center_point.x, self.center_point.y)
        gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs=self.crs)
        gdf_projected = gdf.to_crs(utm_crs)
        area_sq_m = gdf_projected.geometry.area.iloc[0]
        return area_sq_m / 2589988.11
    
    def add_simple_buffer_criterion(self, 
                                   poi_type: Dict = None,
                                   specific_location: str = None,
                                   max_distance_miles: float = 1.0,
                                   criterion_name: str = None) -> gpd.GeoDataFrame:
        """Add a simple distance buffer criterion."""
        pois = self._get_pois(poi_type, specific_location)
        if pois is None or len(pois) == 0:
            return None
        
        utm_crs = self._estimate_utm_crs(self.center_point.x, self.center_point.y)
        pois_projected = pois.to_crs(utm_crs)
        buffer_meters = max_distance_miles * 1609.34
        buffers = pois_projected.geometry.buffer(buffer_meters)
        
        combined_buffer = buffers.unary_union
        result_gdf = gpd.GeoDataFrame([{'geometry': combined_buffer}], crs=utm_crs).to_crs(self.crs)
        result_geometry = result_gdf.geometry.iloc[0].intersection(self.current_search_area)
        
        self.current_search_area = result_geometry
        self.criteria_results.append({
            'name': criterion_name,
            'geometry': result_geometry,
            'description': f"Within {max_distance_miles} miles (straight-line)"
        })
        
        area_sq_miles = self._calculate_area_sq_miles(result_geometry)
        st.info(f"✅ Applied {criterion_name}. New area: {area_sq_miles:.1f} sq miles")
        
        return gpd.GeoDataFrame([{'geometry': result_geometry}], crs=self.crs)
    
    def add_travel_time_criterion(self,
                                 poi_type: Dict = None,
                                 specific_location: str = None,
                                 max_time_minutes: int = 10,
                                 travel_mode: str = 'walk',
                                 criterion_name: str = None,
                                 use_network: bool = None) -> gpd.GeoDataFrame:
        """Add travel time criterion with proper mode-specific calculations."""
        speed_mph = self.TRAVEL_SPEEDS[travel_mode]
        max_distance_miles = (max_time_minutes / 60) * speed_mph
        
        area_sq_miles = self._calculate_area_sq_miles(self.current_search_area)
        
        if use_network is None:
            use_network = area_sq_miles < 50
        
        if use_network and area_sq_miles > 100:
            use_network = False
        
        if use_network:
            return self._network_travel_analysis(
                poi_type, specific_location, max_time_minutes, 
                travel_mode, criterion_name
            )
        else:
            return self._simple_travel_buffer(
                poi_type, specific_location, max_distance_miles,
                travel_mode, criterion_name
            )
    
    def add_two_stage_location_criterion(self,
                                    specific_location: str,
                                    max_time_minutes: int,
                                    travel_mode: str,
                                    criterion_name: str) -> gpd.GeoDataFrame:
        """Two-stage search for specific locations: buffer first, then network analysis."""
        st.info(f"🔍 Two-stage search for {specific_location}")
        
        # Stage 1: Apply simple buffer
        speed_mph = self.TRAVEL_SPEEDS[travel_mode]
        max_distance_miles = (max_time_minutes / 60) * speed_mph
        adjustment = self.BUFFER_ADJUSTMENTS[travel_mode]
        buffer_distance = max_distance_miles * adjustment
        
        # Get the specific location
        pois = self._get_pois(None, specific_location)
        if pois is None or len(pois) == 0:
            return None
        
        # Create initial buffer
        utm_crs = self._estimate_utm_crs(self.center_point.x, self.center_point.y)
        pois_projected = pois.to_crs(utm_crs)
        buffer_meters = buffer_distance * 1609.34
        buffer_geom = pois_projected.geometry.buffer(buffer_meters).iloc[0]
        
        # Convert back to original CRS and intersect with current search area
        buffer_gdf = gpd.GeoDataFrame([{'geometry': buffer_geom}], crs=utm_crs).to_crs(self.crs)
        stage1_area = buffer_gdf.geometry.iloc[0].intersection(self.current_search_area)
        
        st.success(f"✅ Stage 1: Buffer subset created ({self._calculate_area_sq_miles(stage1_area):.1f} sq miles)")
        
        # Stage 2: Network analysis on the subset
        try:
            # Get the location coordinates
            location_point = pois.geometry.iloc[0]
            
            # Download street network for the buffer area only
            # IMPORTANT: Use correct network type for each mode
            network_type = 'walk' if travel_mode == 'walk' else 'drive'
            
            G = ox.graph_from_polygon(
                stage1_area,  # Use the stage 1 area, not the full buffer
                network_type=network_type,
                simplify=True
            )
            
            # Add travel time to edges (this is crucial for correct calculations)
            # Convert edge lengths to travel time in seconds
            for u, v, data in G.edges(data=True):
                # length is in meters
                length_meters = data['length']
                
                # Convert to travel time based on mode
                if travel_mode == 'walk':
                    # Walking: 3 mph = 1.34 m/s
                    travel_time_seconds = length_meters / 1.34
                elif travel_mode == 'bike':
                    # Biking: 12 mph = 5.36 m/s
                    travel_time_seconds = length_meters / 5.36
                else:  # drive
                    # Driving: Need to consider road type for accurate speeds
                    # Default to 55 mph  for now
                    # In reality, should use road class (highway vs residential)
                    if 'highway' in data:
                        # Adjust speed based on road type
                        road_type = data['highway']
                        if isinstance(road_type, list):
                            road_type = road_type[0]
                        
                        # Speed limits by road type (mph)
                        speed_limits = {
                            'motorway': 65,
                            'trunk': 55,
                            'primary': 45,
                            'secondary': 35,
                            'tertiary': 30,
                            'residential': 25,
                            'living_street': 15,
                            'service': 15
                        }
                        
                        speed_mph = speed_limits.get(road_type, 55)
                        speed_mps = speed_mph * 0.44704  # Convert mph to m/s
                    else:
                        speed_mps = 55 * 0.44704  # Default 55 mph
                    
                    travel_time_seconds = length_meters / speed_mps
                
                data['travel_time'] = travel_time_seconds
            
            # Find nearest node to the specific location
            location_node = ox.nearest_nodes(G, location_point.x, location_point.y)
            
            # Calculate travel times from this node
            # Use travel_time as weight, not length!
            max_time_seconds = max_time_minutes * 60
            
            travel_times = nx.single_source_dijkstra_path_length(
                G, 
                location_node, 
                cutoff=max_time_seconds,
                weight='travel_time'  # Use travel time, not distance!
            )
            
            # Create isochrone from reachable nodes
            if travel_times:
                # Get coordinates of all reachable nodes
                node_coords = []
                for node, time_seconds in travel_times.items():
                    if time_seconds <= max_time_seconds:
                        node_data = G.nodes[node]
                        node_coords.append([node_data['x'], node_data['y']])
                
                if len(node_coords) > 2:
                    # Create a more accurate isochrone using alpha shape or concave hull
                    from shapely.geometry import MultiPoint
                    from shapely.ops import unary_union
                    
                    points = MultiPoint([Point(coord) for coord in node_coords])
                    
                    # For driving, use convex hull (simpler but still accurate)
                    # For walking/biking, could use alpha shape for more detail
                    if travel_mode == 'drive':
                        isochrone = points.convex_hull
                    else:
                        # Use buffer + union for more organic shape
                        buffer_size = 0.001 if travel_mode == 'walk' else 0.002
                        buffered_points = [Point(coord).buffer(buffer_size) for coord in node_coords]
                        isochrone = unary_union(buffered_points).convex_hull
                    
                    # Intersect with current search area
                    result_geometry = isochrone.intersection(self.current_search_area)
                    
                    st.success(f"✅ Stage 2: Network analysis complete ({len(node_coords)} reachable nodes)")
                else:
                    # Not enough reachable nodes
                    st.warning("⚠️ Too few reachable nodes found, using buffer method")
                    result_geometry = stage1_area
            else:
                # No reachable nodes found
                st.warning("⚠️ No reachable areas found via network, using buffer method")
                result_geometry = stage1_area
                
        except Exception as e:
            # If network analysis fails, use the buffer result
            st.warning(f"⚠️ Network analysis failed: {str(e)[:100]}... Using buffer method.")
            result_geometry = stage1_area
        
        # Update search area and record results
        self.current_search_area = result_geometry
        self.criteria_results.append({
            'name': criterion_name,
            'geometry': result_geometry,
            'description': f"{travel_mode}: {max_time_minutes} min (two-stage search)"
        })
        
        area_sq_miles = self._calculate_area_sq_miles(result_geometry)
        st.info(f"✅ Applied {criterion_name}. New area: {area_sq_miles:.1f} sq miles")
        
        return gpd.GeoDataFrame([{'geometry': result_geometry}], crs=self.crs)


    def _simple_travel_buffer(self, poi_type, specific_location, 
                             max_distance_miles, travel_mode, criterion_name):
        """Simple buffer adjusted for travel mode."""
        pois = self._get_pois(poi_type, specific_location)
        if pois is None or len(pois) == 0:
            return None
        
        adjustment = self.BUFFER_ADJUSTMENTS[travel_mode]
        adjusted_distance = max_distance_miles / adjustment
        
        utm_crs = self._estimate_utm_crs(self.center_point.x, self.center_point.y)
        pois_projected = pois.to_crs(utm_crs)
        buffer_meters = adjusted_distance * 1609.34
        buffers = pois_projected.geometry.buffer(buffer_meters)
        
        combined_buffer = buffers.unary_union
        result_gdf = gpd.GeoDataFrame([{'geometry': combined_buffer}], crs=utm_crs).to_crs(self.crs)
        result_geometry = result_gdf.geometry.iloc[0].intersection(self.current_search_area)
        
        self.current_search_area = result_geometry
        self.criteria_results.append({
            'name': criterion_name,
            'geometry': result_geometry,
            'description': f"{travel_mode}: {max_distance_miles:.1f} mi / {int(max_distance_miles/self.TRAVEL_SPEEDS[travel_mode]*60)} min"
        })
        
        area_sq_miles = self._calculate_area_sq_miles(result_geometry)
        st.info(f"✅ Applied {criterion_name}. New area: {area_sq_miles:.1f} sq miles")
        
        return gpd.GeoDataFrame([{'geometry': result_geometry}], crs=self.crs)
    
    def _network_travel_analysis(self, poi_type, specific_location,
                                max_time_minutes, travel_mode, criterion_name):
        """Network-based travel time analysis."""
        st.warning("Network analysis requested but not implemented in demo. Using simple buffer method.")
        
        speed_mph = self.TRAVEL_SPEEDS[travel_mode]
        max_distance_miles = (max_time_minutes / 60) * speed_mph
        return self._simple_travel_buffer(
            poi_type, specific_location, max_distance_miles,
            travel_mode, criterion_name
        )
    
    def _get_pois(self, poi_type: Dict = None, 
                  specific_location: str = None) -> gpd.GeoDataFrame:
        """Get POIs from either type search or specific location."""
        if specific_location:
            try:
                location_gdf = ox.geocode_to_gdf(specific_location)
                location_point = location_gdf.geometry.iloc[0].centroid
            except:
                try:
                    lat, lon = ox.geocode(specific_location)
                    location_point = Point(lon, lat)
                except:
                    st.error(f"Could not geocode: {specific_location}")
                    return None
            
            return gpd.GeoDataFrame(
                [{'name': specific_location, 'geometry': location_point}],
                crs=self.crs
            )
        else:
            try:
                pois = ox.features_from_polygon(
                    self.current_search_area,
                    tags=poi_type
                )
                return pois
            except Exception as e:
                st.error(f"No POIs found: {e}")
                return None
    
    def get_current_result(self) -> gpd.GeoDataFrame:
        """Get the current filtered area."""
        return gpd.GeoDataFrame(
            [{'geometry': self.current_search_area,
              'criteria_count': len(self.criteria_results),
              'area_sq_miles': self._calculate_area_sq_miles(self.current_search_area)}],
            crs=self.crs
        )
    
    def visualize_results(self) -> folium.Map:
        """Create interactive map of results."""
        center = [self.center_point.y, self.center_point.x]
        m = folium.Map(location=center, zoom_start=11)
        
        folium.Marker(
            center,
            popup=self.center_location,
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
        
        folium.GeoJson(
            gpd.GeoDataFrame([{'geometry': self.search_boundary}], crs=self.crs).to_json(),
            name='Initial Search Area',
            style_function=lambda x: {
                'fillColor': 'lightgray',
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.1
            }
        ).add_to(m)
        
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightblue']
        for i, result in enumerate(self.criteria_results):
            color = colors[i % len(colors)]
            folium.GeoJson(
                gpd.GeoDataFrame([{'geometry': result['geometry']}], crs=self.crs).to_json(),
                name=f"{result['name']}: {result['description']}",
                style_function=lambda x, c=color: {
                    'fillColor': c,
                    'color': c,
                    'weight': 2,
                    'fillOpacity': 0.15
                }
            ).add_to(m)
        
        folium.GeoJson(
            self.get_current_result().to_json(),
            name='Final Result Area',
            style_function=lambda x: {
                'fillColor': 'darkgreen',
                'color': 'darkgreen',
                'weight': 3,
                'fillOpacity': 0.25,
                'dashArray': '5, 5'
            }
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# Sidebar for inputs
with st.sidebar:
    st.header("🎯 Search Parameters")
    
    # Row 1: Location and radius
    st.subheader("1. Search Area")
    col1, col2 = st.columns([2, 1])
    with col1:
        center_location = st.text_input("City or Location", value="Durham, NC")
    with col2:
        max_radius = st.selectbox("Radius (miles)", options=list(range(1, 26)), index=9)
    
    st.divider()
    
    # Rows 2-5: Amenity criteria
    st.subheader("2. Amenity Criteria")
    amenity_criteria = []
    
    for i in range(4):
        with st.expander(f"Amenity Criterion {i+1}", expanded=(i==0)):
            poi_name = st.selectbox(
                "Amenity Type",
                options=["None"] + list(POI_TYPES.keys()),
                key=f"poi_{i}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                mode = st.selectbox(
                    "Mode",
                    options=["distance", "walk", "bike", "drive"],
                    key=f"mode_{i}"
                )
            with col2:
                if mode == "distance":
                    value = st.number_input("Miles", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"value_{i}")
                else:
                    value = st.number_input("Minutes", min_value=1, max_value=60, value=10, step=1, key=f"value_{i}")
            
            if poi_name != "None":
                amenity_criteria.append({
                    'poi_type': POI_TYPES[poi_name],
                    'poi_name': poi_name,
                    'mode': mode,
                    'value': value
                })
    
    st.divider()
    
    # Rows 6-9: Specific location criteria
    st.subheader("3. Specific Location Criteria")
    location_criteria = []
    
    for i in range(4):
        with st.expander(f"Location Criterion {i+1}", expanded=(i==0)):
            specific_loc = st.text_input("Location/Address", key=f"loc_{i}")
            
            col1, col2 = st.columns(2)
            with col1:
                mode = st.selectbox(
                    "Mode",
                    options=["distance", "walk", "bike", "drive"],
                    key=f"loc_mode_{i}"
                )
            with col2:
                if mode == "distance":
                    value = st.number_input("Miles", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"loc_value_{i}")
                else:
                    value = st.number_input("Minutes", min_value=1, max_value=60, value=10, step=1, key=f"loc_value_{i}")
            
            if specific_loc:
                location_criteria.append({
                    'location': specific_loc,
                    'mode': mode,
                    'value': value
                })
    
    st.divider()
    
    # Analyze button
    analyze_button = st.button("🎯 Analyze Location", type="primary", use_container_width=True)

# Main content area
if analyze_button:
    try:
        # Initialize analyzer
        with st.spinner("Initializing search area..."):
            st.session_state.analyzer = LocationAnalyzer(center_location, max_radius)
        
        # Apply amenity criteria
        for idx, criterion in enumerate(amenity_criteria):
            with st.spinner(f"Analyzing {criterion['poi_name']}..."):
                criterion_name = f"{criterion['poi_name']}_{criterion['mode']}"
                
                if criterion['mode'] == 'distance':
                    st.session_state.analyzer.add_simple_buffer_criterion(
                        poi_type=criterion['poi_type'],
                        max_distance_miles=criterion['value'],
                        criterion_name=criterion_name
                    )
                else:
                    st.session_state.analyzer.add_travel_time_criterion(
                        poi_type=criterion['poi_type'],
                        max_time_minutes=int(criterion['value']),
                        travel_mode=criterion['mode'],
                        criterion_name=criterion_name,
                        use_network=False  # Always use simple method for speed
                    )
        
        # Apply location criteria with TWO-STAGE SEARCH
        for idx, criterion in enumerate(location_criteria):
            with st.spinner(f"Analyzing {criterion['location']}..."):
                criterion_name = f"{criterion['location'][:20]}_{criterion['mode']}"
                
                if criterion['mode'] == 'distance':
                    # For distance mode, use simple buffer as before
                    st.session_state.analyzer.add_simple_buffer_criterion(
                        specific_location=criterion['location'],
                        max_distance_miles=criterion['value'],
                        criterion_name=criterion_name
                    )
                else:
                    # For travel modes, use two-stage search
                    st.session_state.analyzer.add_two_stage_location_criterion(
                        specific_location=criterion['location'],
                        max_time_minutes=int(criterion['value']),
                        travel_mode=criterion['mode'],
                        criterion_name=criterion_name
                    )
        
        st.success("✅ Analysis complete!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.session_state.analyzer = None

# Display results
if st.session_state.analyzer:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Results Map")
        m = st.session_state.analyzer.visualize_results()
        
        # Display map using HTML
        from streamlit.components.v1 import html
        map_html = m._repr_html_()
        html(map_html, height=600)
    
    with col2:
        st.subheader("📊 Analysis Summary")
        
        result = st.session_state.analyzer.get_current_result()
        final_area = result['area_sq_miles'].iloc[0]
        initial_area = st.session_state.analyzer._calculate_area_sq_miles(st.session_state.analyzer.search_boundary)
        reduction = (1 - final_area/initial_area) * 100
        
        st.metric("Final Area", f"{final_area:.1f} sq miles")
        st.metric("Area Reduction", f"{reduction:.1f}%")
        st.metric("Criteria Applied", len(st.session_state.analyzer.criteria_results))
        
        st.divider()
        
        st.subheader("Applied Criteria")
        for criterion in st.session_state.analyzer.criteria_results:
            st.write(f"• **{criterion['name']}**: {criterion['description']}")
        
        # Export button
        st.divider()
        
        if st.button("💾 Export Results", use_container_width=True):
            # Export to GeoJSON
            geojson = result.to_json()
            st.download_button(
                label="Download GeoJSON",
                data=geojson,
                file_name="location_analysis_results.geojson",
                mime="application/json"
            )
else:
    # Welcome message
    st.info("👆 Configure your search parameters in the sidebar and click 'Analyze Location' to begin.")
    
    # Example configurations
    st.subheader("Example Use Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Family-Friendly Neighborhood**
        - Near elementary schools (walking distance)
        - Close to parks (biking distance)
        - Grocery stores within 5-minute drive
        - Near specific daycare center
        """)
    
    with col2:
        st.markdown("""
        **Urban Professional**
        - Walking distance to coffee shops
        - Near public transit stops
        - Gym within biking distance
        - Close to specific workplace address
        """)
