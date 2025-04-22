import folium
from folium.plugins import MarkerCluster

def create_map(store_list, store_location, map_center, map_zoom, selected_store=None):
    """지도를 생성하고, 선택된 상권이 있을 경우 해당 위치로 이동하고 마커를 강조합니다."""
    m = folium.Map(location=map_center, zoom_start=map_zoom)
    mc = MarkerCluster().add_to(m)

    for store in store_list:
        location = store_location.get(store)
        if location:
            lat, lot = location['latitude'], location['longitude']
            folium.Marker([lat, lot], popup=None, tooltip=f"상권 이름: {store}",
                          icon=folium.Icon(color='blue', icon='star')).add_to(mc)
    
    if selected_store:
        location = store_location.get(selected_store)
        if location:
            lat, lot = location['latitude'], location['longitude']
            m.location = [lat, lot]
            m.zoom_start = 17
            folium.Marker([lat, lot], popup=None, tooltip=f"상권 이름: {selected_store}",
                          icon=folium.Icon(color='red', icon='star')).add_to(m)
    return m