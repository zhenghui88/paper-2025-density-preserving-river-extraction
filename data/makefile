
china_10km2.gpkg: ~/work/river/data/merit_hydro.h5
	extract_geometry ~/work/river/data/merit_hydro.h5 ~/work/river/data/merit_hydro.h5 china_10km2.gpkg -c 10.0

china_10km2_inchina_urn_list.txt: china_10km2.gpkg major_basin_in_china.geojson
	select_bypolygon china_10km2.gpkg  major_basin_in_china.geojson china_10km2_inchina_urn_list.txt

china_10km2_urn_list.txt: china_10km2.gpkg china_10km2_inchina_urn_list.txt
	select_bydownstream china_10km2.gpkg china_10km2_inchina_urn_list.txt china_10km2_urn_list.txt

china_10km2_subnet.gpkg: china_10km2.gpkg china_10km2_urn_list.txt
	retain_subnet china_10km2.gpkg china_10km2_urn_list.txt china_10km2_subnet.gpkg

china_10km2_subnet_density.h5: china_10km2_subnet.gpkg
	calculate_density china_10km2_subnet.gpkg china_10km2_subnet_density.h5

drainage_density_merged.h5: china_10km2_subnet_density.h5 drainage_density_survey.h5
	python3 ../tools/drainage_density_merge.py china_10km2_subnet_density.h5 drainage_density_survey.h5 drainage_density_merged.h5

merit_hydro.h5: ~/work/river/data/merit_hydro.h5 drainage_density_merged.h5
	cp ~/work/river/data/merit_hydro.h5 merit_hydro.h5
	python3 ../tools/mask_flow_direction.py merit_hydro.h5 drainage_density_merged.h5

upstream_drainage_length.h5: ~/work/river/data/merit_hydro.h5 drainage_density_merged.h5
	calculate_upstream_length merit_hydro.h5 drainage_density_merged upstream_drainage_length.h5

river_network.gpkg:  merit_hydro.h5 upstream_drainage_length.h5
	extract_geometry_by_drainage_length merit_hydro.h5 merit_hydro.h5 upstream_drainage_length.h5 -c 0.2

china_10km2_subnet_1km.gpkg: china_10km2_subnet.gpkg
    split_by_length china_10km2_subnet.gpkg 1.0 china_10km2_subnet_1km.gpkg

river_network_1km.gpkg: river_network.gpkg
    split_by_length river_network.gpkg 1.0 river_network_1km.gpkg
