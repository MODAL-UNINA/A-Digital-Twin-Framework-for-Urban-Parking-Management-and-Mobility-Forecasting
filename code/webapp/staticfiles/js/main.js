async function initAllTabs() {
    console.log("Initializing all tabs ...");
    await Promise.all(
        Object.values(initTabFunctionMap).map(initFunction => initFunction())
    );
    console.log("All tabs initialized");
}

function initTabMap() {
    tabIds.forEach(id => {
        const cleanId = id.replace("-tab", "");

        const upperCaseId = cleanId === "whatif"
            ? "WhatIf" : cleanId.charAt(0).toUpperCase() + cleanId.slice(1);

        const functionName = `init${upperCaseId}TabContent`;

        if (typeof window[functionName] === "function") {
            initTabFunctionMap[cleanId] = window[functionName];
        }
    });
}

async function initZones() {
    try {
        const data = await fetchData("get_zone_names", {});
        globalData.zones = data;
    } catch (error) {
        handleError(
            `Error fetching zone names: ${error}`,
            "Failed to load zone names. Please try again later."
        );
    }
}

async function initHourslots() {
    try {
        const data = await fetchData("get_hour_slots");
        globalData.hourSlots = data.slots;
    } catch (error) {
        handleError(
            `Error fetching zone hour slots: ${error}`,
            "Failed to load zone names. Please try again."
        );
    }
}

async function initMapData() {
    try {
        const data = await fetchData("get_map_data");
        globalData.mapData = data;
    } catch (error) {
        handleError(
            `Error fetching map data: ${error}`,
            "Failed to load map data. Please try again."
        );
    }
}

async function initCommonData() {
    await Promise.all([
        initZones(),
        initHourslots(),
        initMapData(),
    ]);
}

document.addEventListener("DOMContentLoaded", async () => {
    await checkIfServerIsRunning();

    initTabMap();

    await initCommonData();

    await initAllTabs();
});