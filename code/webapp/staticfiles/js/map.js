function toQueryString(params) {
    const query = new URLSearchParams();
    for (const key in params) {
        if (Array.isArray(params[key])) {
            params[key].forEach((val) => query.append(key, val));
        } else {
            query.append(key, params[key]);
        }
    }
    return query.toString();
}

async function getMeteoData(lat, lon) {
    const url = "https://api.open-meteo.com/v1/forecast";

    const params = {
        latitude: lat,
        longitude: lon,
        current: ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
        timezone: "auto",
    };

    const fullUrl = `${url}?${toQueryString(params)}`;

    try {
        const response = await fetch(fullUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        const result = data.current;
        if (!result) {
            console.error("No current data found in the open-meteo response.");
            return null;
        }

        return result;
    } catch (err) {
        console.error(`Error fetching meteo data: ${err.message}`);
        return null;
    }
}

async function getAirData(lat, lon) {
    const url = "https://air-quality-api.open-meteo.com/v1/air-quality";

    const params = {
        latitude: lat,
        longitude: lon,
        current: ["european_aqi", "pm2_5", "pm10", "nitrogen_dioxide", "ozone"],
        timezone: "auto",
        domains: "cams_europe",
    };

    const fullUrl = `${url}?${toQueryString(params)}`;

    try {
        const response = await fetch(fullUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        const result = data.current;
        if (!result) {
            console.error("No current data found in the open-meteo response.");
            return null;
        }

        return result;
    } catch (err) {
        console.error(`Error fetching air data: ${err.message}`);
        return null;
    }
}

async function getMapInfo(lat, lon) {
    const meteoData = await getMeteoData(lat, lon);

    const airData = await getAirData(lat, lon);

    const temperature = meteoData.temperature_2m;
    const humidity = meteoData.relative_humidity_2m;
    const windSpeed = meteoData.wind_speed_10m;
    const precipitation = meteoData.precipitation;

    const aqi = airData.european_aqi;
    const pm25 = airData.pm2_5;
    const pm10 = airData.pm10;
    const no2 = airData.nitrogen_dioxide;
    const o3 = airData.ozone;

    return {
        Temperature: `${temperature}°C`,
        Humidity: `${Math.round(humidity)}%`,
        "Wind Speed": `${windSpeed} km/h`,
        Precipitation: `${precipitation} mm`,
        AQI: `${aqi}`,
        "PM2.5": `${pm25} µg/m³`,
        PM10: `${pm10} µg/m³`,
        NO2: `${no2} µg/m³`,
        O3: `${o3} µg/m³`,
    };
}

async function initMapInfo(mapData) {
    if (mapInfo === null) {
        return;
    }

    mapInfo.innerHTML = "";
    const option = document.createElement("li");
    option.classList.add("list-group-item");
    option.textContent = "Loading map info...";
    mapInfo.appendChild(option);

    try {
        const [lat, lon] = mapData.center;

        const data = await getMapInfo(lat, lon);
        mapInfo.innerHTML = "";

        if (Object.keys(data).length === 0) {
            const option = document.createElement("li");
            option.classList.add("list-group-item");
            option.textContent = "No data available.";
            mapInfo.appendChild(option);
            return;
        }

        for (const [key, value] of Object.entries(data)) {
            const option = document.createElement("li");
            option.classList.add("list-group-item");
            option.textContent = `${key}: ${value}`;
            mapInfo.appendChild(option);
        }
    } catch (error) {
        console.error("Error fetching map info:", error);
        mapInfo.innerHTML = "";
        const option = document.createElement("li");
        option.classList.add("list-group-item");
        option.textContent = "No data available.";
        mapInfo.appendChild(option);
    }
}

function resetMapContainer(mapContainerId) {
    const container = L.DomUtil.get(mapContainerId);

    if (container._leaflet_map) {
        container._leaflet_map.remove();
    }

    container.innerHTML = "";
    const newMap = L.map(mapContainerId, { zoomSnap: 0.25 });
    container._leaflet_map = newMap;

    return newMap;
}

function createRoundedRectangle(bounds, radius, options) {
    let lat1 = bounds[0][0], lng1 = bounds[0][1];
    let lat2 = bounds[1][0], lng2 = bounds[1][1];

    let minLat = Math.min(lat1, lat2);
    let maxLat = Math.max(lat1, lat2);
    let minLng = Math.min(lng1, lng2);
    let maxLng = Math.max(lng1, lng2);

    const pathData = [
        "M", [minLat, minLng + radius],
        "Q", [minLat, minLng], [minLat + radius, minLng],
        "L", [maxLat - radius, minLng],
        "Q", [maxLat, minLng], [maxLat, minLng + radius],
        "L", [maxLat, maxLng - radius],
        "Q", [maxLat, maxLng], [maxLat - radius, maxLng],
        "L", [minLat + radius, maxLng],
        "Q", [minLat, maxLng], [minLat, maxLng - radius],
        "Z",
    ];

    return L.curve(pathData, { ...options, fill: true });
}

function addLegend(map, titleText) {
    const legend = L.control({ position: "bottomright" });
    legend.onAdd = function () {
        const div = L.DomUtil.create("div", "legend");
        div.innerHTML = `
          <div class="legend-title">${titleText}</div>
          <div class="legend-item">
            <span class="legend-color" style="background-color:crimson;"></span>
            <span>Parking Meters</span>
          </div>
          <div class="legend-item">
            <span class="legend-color square" style="background-color:steelblue;"></span>
            <span>Parking Slots</span>
          </div>
        `;
        return div;
    };
    legend.addTo(map);
}

function drawMap(data, mapContentId) {
    const zones = data.zones;

    const parkingSlotIcon = L.divIcon({
        className: "",
        html: `<div class="parking-slot-icon"></div>`,
        iconSize: [8, 8],
    });

    const mapContent = resetMapContainer(mapContentId);
    const tileLayer = createTileLayer(14.5).addTo(mapContent);

    let allZonesBounds = L.latLngBounds([]);

    let allZoneBoundsArray = [];

    Object.entries(zones).forEach(([zoneName, data], idx) => {
        const rectColor = mapZoneColors[idx % mapZoneColors.length];
        const label = data.label;

        allZonesBounds.extend(data.bounds);
        allZoneBoundsArray.push(data.bounds);

        createRoundedRectangle(data.bounds, 0.0003, {
            color: rectColor,
            fillColor: rectColor,
            weight: 3,
            opacity: 0.15,
            fillOpacity: 0.15,
        })
            .addTo(mapContent)
            .bindPopup(`${label}`);
    });

    mapContent.fitBounds(allZonesBounds);

    mapContent.createPane("metersPane");
    mapContent.getPane("metersPane").style.zIndex = 650;

    mapContent.createPane("slotsPane");
    mapContent.getPane("slotsPane").style.zIndex = 630;

    mapContent.createPane("roadsPane");
    mapContent.getPane("roadsPane").style.zIndex = 610;

    const parkingMeters = filterByMultipleBounds(data.parkingMeters, allZoneBoundsArray);
    const parkingSlots = filterByMultipleBounds(data.parkingSlots, allZoneBoundsArray);

    parkingSlots.forEach((slot) => {
        L.marker([slot.lat, slot.lng], {
            pane: "slotsPane",
            icon: parkingSlotIcon
        })
            .addTo(mapContent)
            .bindPopup(`Parking slot ${slot.id}`);
    });

    parkingMeters.forEach((meter) => {
        L.circleMarker([meter.lat, meter.lng], {
            pane: "metersPane",
            color: "crimson",
            fillColor: "crimson",
            radius: 5,
            weight: 2,
            fillOpacity: 1,
        })
            .addTo(mapContent)
            .bindPopup(`Parking meter ${meter.id}`);
    });

    data.roads.forEach((roadObj) => {
        const roadName = roadObj.road_name;
        const geojson = roadObj.geometry;

        L.geoJSON(geojson, {
            pane: "roadsPane",
            style: {
                color: "yellow",
                weight: 4,
                opacity: 0.7,
            }
        })
            .addTo(mapContent)
            .bindPopup(`${roadName}`);
    });

    addLegend(mapContent, "All Zones Legend");

    tileLayer.on("load", () => {
        if (!isMapLoaded) {
            mapContent.fitBounds(allZonesBounds);
            mapContent.invalidateSize();
            isMapLoaded = true;
        }
    });
}

function initMap(data, mapContentId) {
    drawMap(data, mapContentId);

    const mapTabButton = document.getElementById("map-tab");
    if (mapTabButton) {
        mapTabButton.addEventListener("shown.bs.tab", function (e) {
            setTimeout(function () {
                const container = L.DomUtil.get("map-context");
                if (container && container._leaflet_map) {
                    container._leaflet_map.invalidateSize();
                }
                isMapLoaded = false;
            }, 100);
        });
    }
}

async function initMapTabContent() {
    console.log("Initializing map tab content...");
    isMapLoaded = false;
    initMap(globalData.mapData, "map-context");
    await initMapInfo(globalData.mapData);
    console.log("Map tab content initialized.");
}
