import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Leaflet's default icon issue with Webpack/Vite
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

interface LocationPickerProps {
    initialLat: number;
    initialLon: number;
    onLocationSelect: (lat: number, lon: number) => void;
}

const LocationMarker: React.FC<{ onSelect: (lat: number, lon: number) => void }> = ({ onSelect }) => {
    const [position, setPosition] = useState<L.LatLng | null>(null);

    useMapEvents({
        click(e) {
            setPosition(e.latlng);
            onSelect(e.latlng.lat, e.latlng.lng);
        },
    });

    return position ? <Marker position={position} /> : null;
};

const RecenterMap: React.FC<{ lat: number; lon: number }> = ({ lat, lon }) => {
    const map = useMap();

    useEffect(() => {
        map.setView([lat, lon], map.getZoom(), { animate: true });
    }, [lat, lon, map]);

    return null;
};

const LocationPicker: React.FC<LocationPickerProps> = ({ initialLat, initialLon, onLocationSelect }) => {
    return (
        <div style={{ height: '220px', width: '100%', borderRadius: '8px', overflow: 'hidden', marginTop: '8px' }}>
            <MapContainer center={[initialLat, initialLon]} zoom={13} style={{ height: '100%', width: '100%' }}>
                <RecenterMap lat={initialLat} lon={initialLon} />
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                <LocationMarker onSelect={onLocationSelect} />
            </MapContainer>
        </div>
    );
};

export default LocationPicker;
