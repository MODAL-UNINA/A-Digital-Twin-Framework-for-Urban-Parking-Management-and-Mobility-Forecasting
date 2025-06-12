const csrfToken = document.querySelector("[name=csrfmiddlewaretoken]")?.value;

const tabIds = Array.from(
    document.getElementById("mainTab").getElementsByClassName("nav-link")
).map(link => link.id);

const initTabFunctionMap = {};

const globalData = {
    hourSlots: null,
    mapData: null,
    zones: null
}

const mapZoneColors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#1abc9c", "#e74c3c",
];

const minimapIds = Array.from(
    document.getElementsByClassName("minimap-context")).map(minimap => minimap.id);

const minimapTabNames = minimapIds.map(id => id.replace("-minimap-context", ""));

const isMiniMapLoaded = minimapTabNames.reduce((acc, tabname) => {
    acc[tabname] = false;
    return acc;
}, {});

const mapInfo = document.getElementById("map-info");
let isMapLoaded = false;

const distribData = {
    dates: null,
    parkingMeters: null,
    parkingSlots: null
};

const selectedDistribData = {
    zone: "",
    date: "",
    hourSlot: "",
    parkingMeter: "",
    parkingSlot: "",
    legalityStatus: "",
};

const statsData = {
    dates: null
};

const selectedStatsData = {
    zone: "",
    date: "",
    hourSlot: ""
};

const calendarData = {
    dates: null
};

const selectedCalendarData = {
    date: ""
};

const forecastData = {
    dates: null,
    parkingMeters: null,
    roads: null,
};

const selectedForecastData = {
    zone: "",
    date: "",
    parkingMeter: "",
    road: "",
};

const whatIfScenarios = ["1st", "2nd", "3rd"];

const whatIfScenarioData = {};
const selectedWhatIfScenarioData = {};