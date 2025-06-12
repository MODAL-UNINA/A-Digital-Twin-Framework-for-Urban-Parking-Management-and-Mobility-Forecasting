async function updateDistribTransactionsCountData(zone, date, hourSlot, parkingmeter, imageId) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    try {
        const data = await fetchData(
            "get_distr_transactions_count", {
            "zone_id": zone,
            "hourslot": hourSlot,
            "selected_date": date,
            "parkingmeter": parkingmeter
        });

        if (!data) {
            throw new Error("No data available.");
        }

        imageElement.src =
            `data:image/png;base64,${data.transactions_count}`;
        imageElement.style.display = "block";
        return true;
    } catch (error) {
        handleError(
            `Error fetching distribution transactions count: ${error}`,
            "Failed to load distribution transactions count. Please try again."
        );
    }
    return false;
}

async function updateDistribTransactionsAmountData(zone, date, hourSlot, parkingmeter, imageId) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    try {
        const data = await fetchData(
            "get_distr_transactions_amount", {
            "zone_id": zone,
            "hourslot": hourSlot,
            "selected_date": date,
            "parkingmeter": parkingmeter
        });

        if (!data) {
            throw new Error("No data available.");
        }

        imageElement.src =
            `data:image/png;base64,${data.transactions_amount}`;
        imageElement.style.display = "block";
        return true;
    } catch (error) {
        handleError(
            `Error fetching distribution transactions amount: ${error}`,
            "Failed to load distribution transactions amount. Please try again."
        );
    }
    return false;
}

async function updateDistribOccupancyData(
    zone,
    date,
    hourSlot,
    parkingSlot,
    legalityStatus,
    imageId
) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    try {
        const data = await fetchData(
            "get_distr_occupancy", {
            "zone_id": zone,
            "hourslot": hourSlot,
            "legalitystatus": legalityStatus,
            "selected_date": date,
            "parkingslot": parkingSlot
        });

        if (!data) {
            throw new Error("No data available.");
        }

        imageElement.src =
            `data:image/png;base64,${data.occupancy}`;
        imageElement.style.display = "block";
    } catch (error) {
        handleError(
            `Error fetching distribution parking occupancy: ${error}`,
            "Failed to load distribution parking occupancy. Please try again."
        );
    }
}

function areFinesAvailable(zone, date, hourSlot) {
    if (hourSlot === "1" || hourSlot === "2") {
        return false;
    }

    return true;
}

async function updateDistribFinesData(zone, date, hourSlot, imageId, headerId) {
    const imageElement = document.getElementById(imageId);
    if (!imageElement) {
        return;
    }

    const imageHeader = document.getElementById(headerId);
    if (!imageHeader) {
        return;
    }

    if (!areFinesAvailable(zone, date, hourSlot)) {
        imageElement.src = "";
        imageElement.style.display = "none";
        if (imageHeader !== null) {
            imageHeader.innerHTML = "<h5>Fines (not available)</h5>";
        }
        return;
    }

    try {
        const data = await fetchData(
            "get_distr_fines", {
            "zone_id": zone,
            "hourslot": hourSlot,
            "selected_date": date
        });

        if (!data) {
            throw new Error("No data available.");
        }

        imageElement.src = `data:image/png;base64,${data.fines}`;
        imageElement.style.display = "block";
        if (imageHeader !== null) {
            imageHeader.innerHTML = "<h5>Fines</h5>";
        }
    } catch (error) {
        handleError(
            `Error fetching fines: ${error}`,
            "Failed to load fines. Please try again."
        );
    }
}

function populateDistribTransactionsParkingMeterSelect(zone) {
    const select = document.getElementById("distrib-parkingmeter-select");
    if (!select) {
        return;
    }

    const data = distribData.parkingMeters[zone];
    if (!data) {
        alert("No parking meters available for the selected zone.");
        return;
    }

    populateSelectElement(select, data);
}

function populateDistribOccupancyParkingSlotSelect(zone) {
    const select = document.getElementById("distrib-parkingslot-select");
    if (!select) {
        return;
    }

    const data = distribData.parkingSlots[zone];
    if (!data) {
        alert("No parking slots available for the selected zone.");
        return;
    }

    populateSelectElement(select, data);
}

async function updateDistribOccupancyContent(zone, date, hourSlot, parkingSlot, legalityStatus) {
    await updateDistribOccupancyData(
        zone,
        date,
        hourSlot,
        parkingSlot,
        legalityStatus,
        "distrib-occupancy-context"
    );
}

async function updateDistribTransactionContent(zone, date, hourSlot, parkingmeter) {
    await Promise.all([
        await updateDistribTransactionsCountData(
            zone,
            date,
            hourSlot,
            parkingmeter,
            "distrib-transactions_count-context"
        ),
        await updateDistribTransactionsAmountData(
            zone,
            date,
            hourSlot,
            parkingmeter,
            "distrib-transactions_amount-context"
        )
    ]);
}

async function updateDistribTransactionSection(zone, date, hourSlot) {
    populateDistribTransactionsParkingMeterSelect(zone);
    selectedDistribData.parkingMeter = "0";
    await updateDistribTransactionContent(zone, date, hourSlot, selectedDistribData.parkingMeter);
}

async function updateDistribOccupancySection(zone, date, hourSlot) {
    populateDistribOccupancyParkingSlotSelect(zone);
    selectedDistribData.parkingSlot = "0";
    selectedDistribData.legalityStatus = "both";
    await updateDistribOccupancyContent(
        zone,
        date,
        hourSlot,
        selectedDistribData.parkingSlot,
        selectedDistribData.legalityStatus
    );
}

async function updateDistribFinesSection(zone, date, hourSlot) {
    await updateDistribFinesData(
        zone,
        date,
        hourSlot,
        "distrib-fines-context",
        "distrib-fines-header-context"
    );
}

async function updateDistribAllSections(zone, date, hourSlot) {
    await Promise.all([
        updateDistribTransactionSection(zone, date, hourSlot),
        updateDistribOccupancySection(zone, date, hourSlot),
        updateDistribFinesSection(zone, date, hourSlot)
    ]);
}

function initDistribOccupancyButton() {
    const submitButton = document.getElementById("submit-distrib-occupancy-button");
    if (!submitButton) {
        return false;
    }

    const parkingSlotSelect = document.getElementById("distrib-parkingslot-select");
    if (!parkingSlotSelect) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const parkingSlot = parkingSlotSelect.value;
        if (!parkingSlot || parkingSlot === "default") {
            alert("Please select a parking slot.");
            return;
        }

        const legalityelement = document.querySelector(
            "input[name=\"occupancyoptions\"]:checked"
        );

        selectedDistribData.parkingSlot = parkingSlot;
        selectedDistribData.legalityStatus = (
            !legalityelement ? "both" : legalityelement.value
        );

        toggleInnerContent("distrib-occupancy", false);
        toggleButtonElements("distrib", false);
        updateMiniMap("distrib", "parkingSlot", [selectedDistribData.parkingSlot]);
        toggleInnerContentAfterImages(
            "distrib-occupancy",
            ["distrib-occupancy-context"]
        );
        await updateDistribOccupancyContent(
            selectedDistribData.zone,
            selectedDistribData.date,
            selectedDistribData.hourSlot,
            selectedDistribData.parkingSlot,
            selectedDistribData.legalityStatus
        );
        toggleButtonElements("distrib", true);
    });
    return true;
}

function initDistribTransactionsButton() {
    const submitButton = document.getElementById("submit-distrib-transactions-button");
    if (!submitButton) {
        return false;
    }

    const parkingmeterSelect = document.getElementById("distrib-parkingmeter-select");
    if (!parkingmeterSelect) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const parkingmeter = parkingmeterSelect.value;
        if (!parkingmeter || parkingmeter === "default") {
            alert("Please select a parking meter.");
            return;
        }
        selectedDistribData.parkingMeter = parkingmeter;

        toggleInnerContent("distrib-transactions", false);
        toggleButtonElements("distrib", false);
        updateMiniMap("distrib", "parkingMeter", [selectedDistribData.parkingMeter]);
        toggleInnerContentAfterImages(
            "distrib-transactions",
            ["distrib-transactions_count-context", "distrib-transactions_amount-context"]
        );
        await updateDistribTransactionContent(
            selectedDistribData.zone,
            selectedDistribData.date,
            selectedDistribData.hourSlot,
            selectedDistribData.parkingMeter
        );
        toggleButtonElements("distrib", true);
    });
    return true;
}

function initDistribMainButton() {
    const submitButton = document.getElementById("submit-distrib-main-button");
    if (!submitButton) {
        return false;
    }

    const zoneSelect = document.getElementById("distrib-zone-select");
    if (!zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById("distrib-date-input");
    if (!dateInput) {
        return false;
    }

    const hourSlotSelect = document.getElementById("distrib-hourslot-select");
    if (!hourSlotSelect) {
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

        const hourSlot = hourSlotSelect.value;
        if (!hourSlot) {
            alert("Please select an hour slot.");
            return;
        }

        selectedDistribData.zone = zoneSelect.value;
        selectedDistribData.date = dateInput.value;
        selectedDistribData.hourSlot = hourSlotSelect.value;
        toggleInnerContent("distrib", false);
        updateMiniMap("distrib", "zone", [selectedDistribData.zone]);
        toggleInnerDistribAfterImages(
            areFinesAvailable(
                selectedDistribData.zone,
                selectedDistribData.date,
                selectedDistribData.hourSlot
            ));
        await updateDistribAllSections(
            selectedDistribData.zone,
            selectedDistribData.date,
            selectedDistribData.hourSlot
        );
    });
    return true;
}

function initDistribVars() {
    selectedDistribData.zone = "0";
    selectedDistribData.date = getLastAvailableDate("distrib-date-input");
    selectedDistribData.hourSlot = "0";
}

function initDistribMainSelection() {
    const zoneSelect = document.getElementById("distrib-zone-select");
    if (!zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById("distrib-date-input");
    if (!dateInput) {
        return false;
    }

    const hourSlotSelect = document.getElementById("distrib-hourslot-select");
    if (!hourSlotSelect) {
        return false;
    }

    const submitButton = document.getElementById("submit-distrib-main-button");
    if (!submitButton) {
        return false;
    }

    populateSelectElement(zoneSelect, globalData.zones);
    populateDateInputElement(dateInput, distribData.dates);
    populateSelectElement(hourSlotSelect, globalData.hourSlots);
}

async function getAvailableDistribParkingSlots() {
    try {
        const data = await fetchData("get_available_distrib_parkingslots");
        distribData.parkingSlots = data;
    } catch (error) {
        handleError(
            `Error fetching distribution parking meters: ${error}`,
            "Failed to load distribution parking meters. Please try again."
        );
    }
}

async function getAvailableDistribParkingMeters() {
    try {
        const data = await fetchData("get_available_distrib_parkingmeters");
        distribData.parkingMeters = data;
    } catch (error) {
        handleError(
            `Error fetching distribution parking meters: ${error}`,
            "Failed to load distribution parking meters. Please try again."
        );
    }
}

async function getAvailableDistribDates() {
    try {
        const data = await fetchData("get_available_distrib_dates");
        distribData.dates = data;
        return true;
    } catch (error) {
        handleError(
            `Error fetching available distribution dates: ${error}`,
            "Failed to load available distribution dates. Please try again."
        );
    }
    return false;
}

function toggleInnerDistribAfterImages(loadFines) {
    let imageIds = [
        "distrib-transactions_count-context",
        "distrib-transactions_amount-context",
        "distrib-occupancy-context"
    ];

    if (loadFines) {
        imageIds.push("distrib-fines-context");
    }

    toggleInnerContentAfterImages("distrib", imageIds);
}

async function initDistribTabContent() {
    console.log("Initializing distribution tab content...");
    toggleInnerContent("distrib", false);
    toggleInnerDistribAfterImages(
        areFinesAvailable(
            selectedDistribData.zone,
            selectedDistribData.date,
            selectedDistribData.hourSlot
        )
    );
    await getAvailableDistribDates();
    await getAvailableDistribParkingMeters();
    await getAvailableDistribParkingSlots();

    initDistribMainSelection();

    initMiniMap("distrib", globalData.mapData, false);

    initDistribVars();
    initDistribMainButton();
    initDistribTransactionsButton();
    initDistribOccupancyButton();

    await updateDistribAllSections(
        selectedDistribData.zone,
        selectedDistribData.date,
        selectedDistribData.hourSlot
    );
    console.log("Distribution tab content initialized.");
}