async function updateForecastTransactionsCountData(zone, selected_date, parkingmeter, imageId) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    try {
        const data = await fetchData("get_forecast_transactions", {
            "zone_id": zone,
            "parkingmeter": parkingmeter,
            "selected_date": selected_date
        });

        if (!data) {
            throw new Error("No data available.");
        }

        if (!data.forecast_transactions) {
            imageElement.src = "";
            imageElement.style.display = "none";
            return;
        }

        imageElement.src =
            `data:image/png;base64,${data.forecast_transactions}`;
        imageElement.style.display = "block";
    } catch (error) {
        handleError(
            `Error fetching forecast transactions count: ${error}`,
            "Failed to load forecast transactions count. Please try again."
        );
    }
}

async function updateForecastTransactionsAmountData(zone, selected_date, parkingmeter, imageId) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    try {
        const data = await fetchData("get_forecast_amount", {
            "zone_id": zone,
            "parkingmeter": parkingmeter,
            "selected_date": selected_date
        });

        if (!data) {
            throw new Error("No data available.");
        }

        if (!data.forecast_amount) {
            imageElement.src = "";
            imageElement.style.display = "none";
            return;
        }

        imageElement.src =
            `data:image/png;base64,${data.forecast_amount}`;
        imageElement.style.display = "block";
    } catch (error) {
        handleError(
            `Error fetching forecast transactions amount: ${error}`,
            "Failed to load forecast transactions amount. Please try again."
        );
    }
}

async function updateForecastOccupancyData(zone, selected_date, road, imageId) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    try {
        const data = await fetchData("get_forecast_road", {
            "zone_id": zone,
            "road": road,
            "selected_date": selected_date
        });

        if (!data) {
            throw new Error("No data available.");
        }

        if (!data.forecast_road) {
            imageElement.src = "";
            imageElement.style.display = "none";
            return;
        }

        imageElement.src =
            `data:image/png;base64,${data.forecast_road}`;
        imageElement.style.display = "block";
    } catch (error) {
        handleError(
            `Error fetching forecast occupancy: ${error}`,
            "Failed to load forecast occupancy. Please try again."
        );
    }
}

function populateForecastTransactionsParkingmeterSelect(zone) {
    const select = document.getElementById("forecast-parkingmeter-select");
    if (!select) {
        return;
    }

    const data = forecastData.parkingMeters[zone];
    if (!data) {
        alert("No parking meters available for the selected zone.");
        return;
    }

    populateSelectElement(select, data);
}

function populateForecastOccupancyRoadSelect(zone) {
    const select = document.getElementById("forecast-road-select");
    if (!select) {
        return;
    }

    const data = forecastData.roads[zone];
    if (!data) {
        alert("No roads available for the selected zone.");
        return;
    }

    populateSelectElement(select, data);
}

async function updateForecastOccupancyContent(zone, date, road) {
    await updateForecastOccupancyData(zone, date, road, "forecast-occupancy-context");
}

async function updateForecastTransactionContent(zone, date, parkingmeter) {
    await Promise.all([
        await updateForecastTransactionsCountData(
            zone,
            date,
            parkingmeter,
            "forecast-transactions_count-context"
        ),
        await updateForecastTransactionsAmountData(
            zone,
            date,
            parkingmeter,
            "forecast-transactions_amount-context"
        )
    ]);
}

async function updateForecastTransactionSection(zone, date) {
    populateForecastTransactionsParkingmeterSelect(zone);
    selectedForecastData.parkingMeter = "0";
    await updateForecastTransactionContent(zone, date, selectedForecastData.parkingMeter);
}

async function updateForecastOccupancySection(zone, date) {
    populateForecastOccupancyRoadSelect(zone);
    selectedForecastData.road = "0";
    await updateForecastOccupancyContent(zone, date, selectedForecastData.road);
}

async function updateForecastAllSections(zone, date) {
    await Promise.all([
        updateForecastTransactionSection(zone, date),
        updateForecastOccupancySection(zone, date),
    ]);

    const imageIds = [
        "forecast-transactions_count-context",
        "forecast-transactions_amount-context",
        "forecast-occupancy-context"
    ];

    let tobedisabled = false;
    for (const imageId of imageIds) {
        const imageElement = document.getElementById(imageId);
        if (imageElement.src === "") {
            tobedisabled = true;
            break;
        }
    }

    const forecastHeader = document.getElementById("forecast-header-context");
    const forecastBody = document.getElementById("forecast-body-context");

    if (tobedisabled) {
        if (forecastHeader !== null) forecastHeader.style.display = "block";
        if (forecastBody !== null) forecastBody.style.display = "none";
    } else {
        if (forecastHeader !== null) forecastHeader.style.display = "none";
        if (forecastBody !== null) forecastBody.style.display = "block";
    }
}

function initForecastOccupancyButton() {
    const submitButton = document.getElementById("submit-forecast-occupancy-button");
    if (!submitButton) {
        return false;
    }

    const roadSelect = document.getElementById("forecast-road-select");
    if (!roadSelect) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const road = roadSelect.value;
        if (!road || road === "default") {
            alert("Please select a road.");
            return;
        }

        selectedForecastData.road = road;

        toggleInnerContent("forecast-occupancy", false);
        toggleButtonElements("forecast", false);
        toggleInnerContentAfterImages(
            "forecast-occupancy",
            ["forecast-occupancy-context"]
        );
        updateMiniMap("forecast", "road", [selectedForecastData.road]);
        await updateForecastOccupancyContent(
            selectedForecastData.zone,
            selectedForecastData.date,
            selectedForecastData.road,
        );
        toggleButtonElements("forecast", true);
    });
    return true;
}

function initForecastTransactionsButton() {
    const submitButton = document.getElementById("submit-forecast-transactions-button");
    if (!submitButton) {
        return false;
    }

    const parkingmeterSelect = document.getElementById("forecast-parkingmeter-select");
    if (!parkingmeterSelect) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const parkingmeter = parkingmeterSelect.value;
        if (!parkingmeter || parkingmeter === "default") {
            alert("Please select a parking meter.");
            return;
        }
        selectedForecastData.parkingMeter = parkingmeter;

        toggleInnerContent("forecast-transactions", false);
        toggleButtonElements("forecast", false);
        toggleInnerContentAfterImages(
            "forecast-transactions",
            ["forecast-transactions_count-context", "forecast-transactions_amount-context"]
        );
        updateMiniMap("forecast", "parkingMeter", [selectedForecastData.parkingMeter]);
        await updateForecastTransactionContent(
            selectedForecastData.zone,
            selectedForecastData.date,
            selectedForecastData.parkingMeter
        );
        toggleButtonElements("forecast", true);
    });
    return true;
}

function initForecastMainButton() {
    const submitButton = document.getElementById("submit-forecast-main-button");
    if (!submitButton) {
        return false;
    }

    const zoneSelect = document.getElementById("forecast-zone-select");
    if (!zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById("forecast-date-input");
    if (!dateInput) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const zone = zoneSelect.value;
        if (!zone || zone === "default") {
            alert("Please select a zone.");
            return;
        }

        const date = dateInput.value;
        if (!date) {
            alert("Please select a date.");
            return;
        }

        selectedForecastData.zone = zoneSelect.value;
        selectedForecastData.date = dateInput.value;
        toggleInnerContent("forecast", false);
        updateMiniMap("forecast", "zone", [selectedForecastData.zone]);
        toggleInnerForecastAfterImages();
        await updateForecastAllSections(selectedForecastData.zone, selectedForecastData.date);
    });
    return true;
}

function initForecastVars() {
    selectedForecastData.zone = "0";
    selectedForecastData.date = getLastAvailableDate("forecast-date-input");
}

function initForecastMainSelection() {
    const zoneSelect = document.getElementById("forecast-zone-select");
    if (!zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById("forecast-date-input");
    if (!dateInput) {
        return false;
    }

    const submitButton = document.getElementById("submit-forecast-main-button");
    if (!submitButton) {
        return false;
    }

    populateSelectElement(zoneSelect, globalData.zones);
    populateDateInputElement(dateInput, forecastData.dates);
}

async function getAvailableForecastRoads() {
    try {
        const data = await fetchData("get_available_forecast_roads");
        forecastData.roads = data;
    } catch (error) {
        handleError(
            `Error fetching forecasting parking meters: ${error}`,
            "Failed to load forecasting parking meters. Please try again."
        );
    }
}

async function getAvailableForecastParkingmeters() {
    try {
        const data = await fetchData("get_available_forecast_parkingmeters");
        forecastData.parkingMeters = data;
    } catch (error) {
        handleError(
            `Error fetching forecasting parking meters: ${error}`,
            "Failed to load forecasting parking meters. Please try again."
        );
    }
}

async function getAvailableForecastDates() {
    try {
        const data = await fetchData("get_available_forecast_dates");
        forecastData.dates = data;
        return true;
    } catch (error) {
        handleError(
            `Error fetching available forecasting dates: ${error}`,
            "Failed to load available forecasting dates. Please try again."
        );
    }
    return false;
}

function toggleInnerForecastAfterImages() {
    const imageIds = [
        "forecast-transactions_count-context",
        "forecast-transactions_amount-context",
        "forecast-occupancy-context"
    ];

    toggleInnerContentAfterImages("forecast", imageIds);
}

async function initForecastTabContent() {
    console.log("Initializing forecast tab content...");
    toggleInnerContent("forecast", false);
    toggleInnerForecastAfterImages();
    await getAvailableForecastDates();
    await getAvailableForecastParkingmeters();
    await getAvailableForecastRoads();

    initForecastMainSelection();

    initMiniMap("forecast", globalData.mapData);

    initForecastVars();
    initForecastMainButton();

    initForecastTransactionsButton();
    initForecastOccupancyButton();

    await updateForecastAllSections(
        selectedForecastData.zone,
        selectedForecastData.date
    );
    console.log("Forecast tab content initialized.");
}