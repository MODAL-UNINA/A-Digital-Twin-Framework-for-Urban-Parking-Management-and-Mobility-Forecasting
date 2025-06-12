function resetMiniMapContainer(mapContainerId) {
    const container = L.DomUtil.get(mapContainerId);

    if (container._leaflet_map) {
        container._leaflet_map.remove();
    }

    container.innerHTML = "";
    const newMap = L.map(mapContainerId, { zoomSnap: 0.25 });
    container._leaflet_map = newMap;

    return newMap;
}

function drawMiniMap(tabname, data, mapContentId, showRoads = true) {
    const zones = data.zones;

    const parkingSlotIcon = L.divIcon({
        className: "",
        html: `<div class="parking-slot-icon"></div>`,
        iconSize: [8, 8],
    });

    const mapContent = resetMiniMapContainer(mapContentId);
    const tileLayer = createTileLayer(13.75).addTo(mapContent);

    mapContent._zoneLayers = [];
    mapContent._parkingSlotMarkers = [];
    mapContent._parkingMeterMarkers = [];
    mapContent._roadLayers = [];
    mapContent._selectedZones = 0;

    let allZonesBounds = L.latLngBounds([]);
    let allZoneBoundsArray = [];

    Object.entries(zones).forEach(([zoneName, data], idx) => {
        const rectColor = mapZoneColors[idx % mapZoneColors.length];
        const label = data.label;
        const zoneBounds = data.bounds;

        allZonesBounds.extend(zoneBounds);
        allZoneBoundsArray.push(zoneBounds);

        const rectLayer = createRoundedRectangle(zoneBounds, 0.0003, {
            color: rectColor,
            fillColor: rectColor,
            weight: 3,
            opacity: 0.15,
            fillOpacity: 0.15,
        })
            .addTo(mapContent)
            .bindPopup(`${label}`);

        mapContent._zoneLayers.push({
            zoneId: Number(zoneName),
            layer: rectLayer,
            bounds: L.latLngBounds(zoneBounds),
        });
    });

    mapContent.fitBounds(allZonesBounds);

    mapContent.createPane("metersPane");
    mapContent.getPane("metersPane").style.zIndex = 650;

    mapContent.createPane("slotsPane");
    mapContent.getPane("slotsPane").style.zIndex = 630;

    if (showRoads) {
        mapContent.createPane("roadsPane");
        mapContent.getPane("roadsPane").style.zIndex = 610;
    }

    const parkingMeters = filterByMultipleBounds(data.parkingMeters, allZoneBoundsArray);
    const parkingSlots = filterByMultipleBounds(data.parkingSlots, allZoneBoundsArray);

    parkingSlots.forEach((slot) => {
        const marker = L.marker([slot.lat, slot.lng], {
            pane: "slotsPane",
            icon: parkingSlotIcon
        })
            .addTo(mapContent)
            .bindPopup(`Parking slot ${slot.id}`);

        mapContent._parkingSlotMarkers.push({
            slotId: Number(slot.id),
            roadId: Number(slot.road_id),
            zoneId: Number(slot.zone_id),
            marker: marker,
        });
    });

    parkingMeters.forEach((meter) => {
        const circleMarker = L.circleMarker([meter.lat, meter.lng], {
            pane: "metersPane",
            color: "crimson",
            fillColor: "crimson",
            radius: 5,
            weight: 2,
            fillOpacity: 1,
        })
            .addTo(mapContent)
            .bindPopup(`Parking meter ${meter.id}`);

        mapContent._parkingMeterMarkers.push({
            meterId: Number(meter.id),
            roadId: Number(meter.road_id),
            zoneId: Number(meter.zone_id),
            marker: circleMarker,
        });
    });

    if (showRoads) {
        data.roads.forEach((roadObj) => {
            const geojson = roadObj.geometry;

            const marker = L.geoJSON(geojson, {
                pane: "roadsPane",
                style: {
                    color: "yellow",
                    weight: 4,
                    opacity: 0.7,
                }
            })
                .addTo(mapContent)
                .bindPopup(`${roadObj.road_name}`);

            mapContent._roadLayers.push({
                roadId: Number(roadObj.road_id),
                zoneId: Number(roadObj.zone_id),
                marker: marker,
            });
        });
    }

    tileLayer.on("load", () => {
        if (!isMiniMapLoaded[tabname]) {
            mapContent.fitBounds(allZonesBounds);
            mapContent.invalidateSize();
            isMiniMapLoaded[tabname] = true;
        }
    });
}

function updateMiniMap(tabname, selectedKind, selectedValue) {
    const mapContentId = `${tabname}-minimap-context`;
    const container = L.DomUtil.get(mapContentId);

    if (!container || !container._leaflet_map) {
        return;
    }

    const mapContent = container._leaflet_map;

    function showLayer(layer) {
        if (!mapContent.hasLayer(layer)) {
            layer.addTo(mapContent);
        }
    }
    function hideLayer(layer) {
        if (mapContent.hasLayer(layer)) {
            mapContent.removeLayer(layer);
        }
    }

    function parseZoneIds(value) {
        if (Array.isArray(value)) {
            if (value.length === 0) {
                throw new Error("Invalid zone ID array: empty array");
            }
            if (value.includes(0) || value.includes("0")) {
                return [0];
            }

            const numericVals = value.map((v) => {
                const parsed = Number(v);
                if (isNaN(parsed)) {
                    throw new Error(`Invalid zone ID: ${v} is not a valid number`);
                }
                return parsed;
            });

            return numericVals;
        } else {
            const parsed = Number(value);
            if (isNaN(parsed)) {
                throw new Error(`Invalid zone ID: ${value} is not a valid number`);
            }
            return [parsed];
        }
    }

    if (selectedKind === "zone") {
        const zoneIds = parseZoneIds(selectedValue);

        mapContent._selectedZones = zoneIds;

        const allZero = zoneIds.every((zid) => zid === 0);
        if (allZero) {
            mapContent._zoneLayers.forEach((z) => {
                showLayer(z.layer);
            });
            mapContent._parkingSlotMarkers.forEach((s) => showLayer(s.marker));
            mapContent._parkingMeterMarkers.forEach((m) => showLayer(m.marker));

            mapContent._roadLayers.forEach((r) => {
                showLayer(r.marker);
            });
        } else {
            mapContent._zoneLayers.forEach((z) => {
                if (zoneIds.includes(z.zoneId)) {
                    showLayer(z.layer);
                } else {
                    hideLayer(z.layer);
                }
            });

            mapContent._roadLayers.forEach((roadObj) => {
                if (zoneIds.includes(roadObj.zoneId)) {
                    showLayer(roadObj.marker);
                } else {
                    hideLayer(roadObj.marker);
                }
            });

            mapContent._parkingSlotMarkers.forEach((slotObj) => {
                if (zoneIds.includes(slotObj.zoneId)) {
                    showLayer(slotObj.marker);
                } else {
                    hideLayer(slotObj.marker);
                }
            });

            mapContent._parkingMeterMarkers.forEach((meterObj) => {
                if (zoneIds.includes(meterObj.zoneId)) {
                    showLayer(meterObj.marker);
                } else {
                    hideLayer(meterObj.marker);
                }
            });
        }
    }

    const zoneIds = mapContent._selectedZones || [];
    const allZero = zoneIds.every((zid) => zid === 0) || zoneIds.length === 0;

    if (selectedKind === "road") {
        const roadId = Number(selectedValue);
        if (roadId === 0) {
            if (!allZero) {
                mapContent._roadLayers.forEach((roadObj) => {
                    if (zoneIds.includes(roadObj.zoneId)) {
                        showLayer(roadObj.marker);
                    } else {
                        hideLayer(roadObj.marker);
                    }
                });
                mapContent._parkingSlotMarkers.forEach((slotObj) => {
                    if (zoneIds.includes(slotObj.zoneId)) {
                        showLayer(slotObj.marker);
                    } else {
                        hideLayer(slotObj.marker);
                    }
                });
            } else {
                mapContent._roadLayers.forEach((roadObj) => {
                    showLayer(roadObj.marker);
                });
                mapContent._parkingSlotMarkers.forEach((slotObj) => {
                    showLayer(slotObj.marker);
                });
            }
        } else {
            mapContent._roadLayers.forEach((roadObj) => {
                if (roadObj.roadId === roadId) {
                    showLayer(roadObj.marker);
                } else {
                    hideLayer(roadObj.marker);
                }
            });
            mapContent._parkingSlotMarkers.forEach((slotObj) => {
                if (slotObj.roadId === roadId) {
                    showLayer(slotObj.marker);
                } else {
                    hideLayer(slotObj.marker);
                }
            });
        }
    }

    if (selectedKind === "parkingSlot") {
        const slotId = Number(selectedValue);
        if (slotId === 0) {
            if (!allZero) {
                mapContent._parkingSlotMarkers.forEach((slotObj) => {
                    if (zoneIds.includes(slotObj.zoneId)) {
                        showLayer(slotObj.marker);
                    } else {
                        hideLayer(slotObj.marker);
                    }
                });
            } else {
                mapContent._parkingSlotMarkers.forEach((slotObj) => {
                    showLayer(slotObj.marker);
                });
            }
        } else {
            mapContent._parkingSlotMarkers.forEach((slotObj) => {
                if (slotObj.slotId === slotId) {
                    showLayer(slotObj.marker);
                } else {
                    hideLayer(slotObj.marker);
                }
            });
        }
    }

    if (selectedKind === "parkingMeter") {
        const meterId = Number(selectedValue);
        if (meterId === 0) {
            if (!allZero) {
                mapContent._parkingMeterMarkers.forEach((meterObj) => {
                    if (zoneIds.includes(meterObj.zoneId)) {
                        showLayer(meterObj.marker);
                    } else {
                        hideLayer(meterObj.marker);
                    }
                });
            } else {
                mapContent._parkingMeterMarkers.forEach((meterObj) => {
                    showLayer(meterObj.marker);
                });
            }
        } else {
            mapContent._parkingMeterMarkers.forEach((meterObj) => {
                if (meterObj.meterId === meterId) {
                    showLayer(meterObj.marker);
                } else {
                    hideLayer(meterObj.marker);
                }
            });
        }
    }

    mapContent.invalidateSize();
}

function setMiniMapTabUpdateButton(tabname, tabId) {
    const mapContentId = `${tabname}-minimap-context`;

    const mapTabButton = document.getElementById(tabId);
    if (!mapTabButton) return;

    mapTabButton.addEventListener("shown.bs.tab", function (e) {
        setTimeout(function () {
            const container = L.DomUtil.get(mapContentId);
            if (container && container._leaflet_map) {
                container._leaflet_map.invalidateSize();
            }
            isMiniMapLoaded[tabname] = false;
        }, 100);
    });
}

function initMiniMap(tabname, data, showRoads = true) {
    if (!(tabname in isMiniMapLoaded)) {
        console.error(`Mini map for tab ${tabname} is not correctly set up.`);
        return;
    }

    const mapContentId = `${tabname}-minimap-context`;

    drawMiniMap(tabname, data, mapContentId, showRoads);
    setMiniMapTabUpdateButton(tabname, `${tabname}-tab`);
}