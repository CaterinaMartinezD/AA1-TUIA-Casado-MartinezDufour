import geopandas as gpd
from shapely.geometry import Point

# --- 1. Leer shapefile ---
zonas_gdf = gpd.read_file("Climate Zone Maps/Climate zones AU.shp")

# --- 2. Crear puntos (ejemplo) ---
coordenadas = {
    "Albury": (-36.073730, 146.913544), "Badgerys Creek": (-33.887421, 150.740509), "Cobar": (-31.494930, 145.840164), "Coffs Harbour": (-30.298613, 153.109390),
    "Moree": (-29.463551, 149.841721), "Newcastle": (-32.926670, 151.780014), "Norah Head": (-33.283340, 151.566116), "Norfolk Island": (-29.040834, 167.954712),
    "Penrith": (-33.752918, 150.690674), "Richmond": (-42.735809, 147.437088), "Sydney": (-33.868820, 151.209290), "Sydney Airport": (-33.939922, 151.175278),
    "Wagga Wagga": (-35.114750, 147.369614), "Williamtown": (-32.814999, 151.842773), "Wollongong": (-34.427811, 150.893066), "Canberra": (-35.280937, 149.130005),
    "Tuggeranong": (-35.424400, 149.088806), "Mount Ginini": (-37.828410, 140.780656), "Ballarat": (-37.562160, 143.850250), "Bendigo": (-36.759338, 144.283997),
    "Sale": (-38.107250, 147.067291), "Melbourne Airport": (-37.670528, 144.848938), "Melbourne": (-37.813629, 144.963058), "Mildura": (-34.210468, 142.142044),
    "Nhill": (-36.332472, 141.649494), "Portland": (-38.342281, 141.603958), "Watsonia": (-37.711699, 145.082002), "Dartmoor": (-37.919090, 141.274673),
    "Brisbane": (-27.470030, 153.022980), "Cairns": (-16.918550, 145.778061), "Gold Coast": (-28.001499, 153.428467), "Townsville": (-19.258965, 146.816956),
    "Adelaide": (-34.927170, 138.599533), "Mount Gambier": (-37.828411, 140.780655), "Nuriootpa": (-34.471859, 138.996216), "Woomera": (-31.200684, 136.825919),
    "Albany": (-35.023819, 117.884727), "Witchcliffe": (-34.025699, 115.100107), "RAFF Base Pearce": (-17.591089, 123.777382), "Perth Airport": (-31.932739, 115.960258),
    "Perth": (-31.950527, 115.860458), "Salmon Gums": (-32.982075, 121.644170), "Walpole": (-34.976129, 116.731910), "Hobart": (-42.881901, 147.323807),
    "Launceston": (-41.437019, 147.139389), "Alice Springs": (-23.700680, 133.880707), "Darwin": (-12.463440, 130.845642), "Katherine": (-14.464970, 132.264267),
    "Uluru": (-25.344427, 131.036880)
}

# --- 3. Crear GeoDataFrame con CRS correcto ---
puntos_gdf = gpd.GeoDataFrame(
    {"city": list(coordenadas.keys())},
    geometry=[Point(lon, lat) for lat, lon in coordenadas.values()],
    crs="EPSG:4326"  # lat/lon
)

# --- 4. Reproyectar puntos al CRS del shapefile ---
puntos_gdf = puntos_gdf.to_crs(zonas_gdf.crs)

# --- 5. Hacer join espacial ---
joined = gpd.sjoin_nearest(puntos_gdf, zonas_gdf, how="left", max_distance=50000)

# --- 6. Mostrar resultados ---
for idx, row in joined.iterrows():
    print(f"{row['city']}: {row['clim_zone']}")

# --- 7. Guardar resultados ---
joined[['city', 'clim_zone']].to_csv("zonas_climaticas.csv", index=False)
