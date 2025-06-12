async function runGeneration(scenario, zone, date, quantity) {
    try {
        const data = await fetchData("run_whatif_generation", {
            "quantity": quantity,
            "selected_date": date,
            "scenario": scenario,
            "zone_id": zone
        });

        if (data.error) {
            return [null, data.error];
        }

        return [data, null];
    } catch (error) {
        handleError(
            `Error running generation for scenario ${scenario}: ${error}`,
            "Failed to run generation. Please try again."
        );
    }
    return [null, "Failed to run generation"];
}

async function getWhatIfHeatmapsData(zone, date, scenario, quantity, kind, weekDay) {
    try {
        const data = await fetchData("get_whatif_heatmaps", {
            "zone_id": zone,
            "selected_date": date,
            "scenario": scenario,
            "quantity": quantity,
            "kind": kind,
            "selected_weekday": weekDay,
        });

        if (data.error) {
            throw new Error(data.error);
        }

        return data;
    } catch (error) {
        handleError(
            `Error fetching the simulation data: ${error}`,
            "Failed to load the simulation data. Please try again."
        );
    }
    return null;
}

async function getWhatIfDistributionsData(zone, selected_date, scenario, quantity, kind) {
    try {
        const data = await fetchData("get_whatif_distributions", {
            "zone_id": zone,
            "selected_date": selected_date,
            "scenario": scenario,
            "quantity": quantity,
            "kind": kind,
        });

        if (data.error) {
            throw new Error(data.error);
        }

        return data;
    } catch (error) {
        handleError(
            `Error fetching the simulation data: ${error}`,
            "Failed to load the simulation data. Please try again."
        );
    }
    return null;
}

async function getWhatIfCumulativePlotData(
    zone,
    selected_date,
    scenario,
    quantity,
    kind,
    localZone
) {
    try {
        const data = await fetchData("get_whatif_cumulative_plot", {
            "zone_id": zone,
            "selected_date": selected_date,
            "scenario": scenario,
            "quantity": quantity,
            "adjacent_zone_id": localZone,
            "kind": kind,
        });

        if (data.error) {
            throw new Error(data.error);
        }

        return data;
    } catch (error) {
        handleError(
            `Error fetching the simulation data: ${error}`,
            "Failed to load the simulation data. Please try again."
        );
    }
    return null;
}

async function getWhatIfGeneratedData(
    zone,
    date,
    scenario,
    quantity,
    section,
    kind,
    weekDay,
    localZone
) {
    if (section === "heatmap") {
        return await getWhatIfHeatmapsData(zone, date, scenario, quantity, kind, weekDay);
    }
    if (section === "distrib") {
        return await getWhatIfDistributionsData(zone, date, scenario, quantity, kind);
    }
    if (section === "plot") {
        return await getWhatIfCumulativePlotData(zone, date, scenario, quantity, kind, localZone);
    }

    throw new Error("Invalid section.");
}

async function updateWhatIfImages(
    zone,
    date,
    scenario,
    quantity,
    section,
    kind,
    weekDay,
    localZone,
    imageElemData
) {
    if (imageElemData === null) return;

    for (const [key, imageElem] of Object.entries(imageElemData)) {
        if (imageElem === null) return;

        imageElem.style.display = "none";
        imageElem.src = "";
    }

    const imageFormat = section === "heatmap" ? "gif" : "png";

    const outdata = await getWhatIfGeneratedData(
        zone,
        date,
        scenario,
        quantity,
        section,
        kind,
        weekDay,
        localZone
    );
    if (outdata === null) throw new Error("What-If image data is missing.");

    const promises = [];
    for (const key of Object.keys(imageElemData)) {
        promises.push(loadImageAsync(
            imageElemData[key],
            `data:image/${imageFormat};base64,${outdata[key]}`
        ));
    }

    await Promise.all(promises);

    for (const [key, imageElem] of Object.entries(imageElemData)) {
        if (imageElem === null) return;
        imageElem.style.display = "block";
    }
}

function populateWhatIfScenarioSelectors(scenario, data) {
    populateDateInputElement(
        document.getElementById(`whatif_${scenario}scenario-heatmap-date-input`),
        whatIfScenarioData[scenario].inner.heatmap.weekDays, false
    );

    const localZoneSelect = document.getElementById(
        `whatif_${scenario}scenario-plot-localzone-select`);
    if (localZoneSelect != null) {
        const localZoneSelectData = {};
        for (const val of data.selected_zones) {
            localZoneSelectData[val] = (
                Number(val) === 0 ? `All zones` : `Zone ${Number(val) - 1}`
            );
        }
        populateSelectElement(
            localZoneSelect,
            localZoneSelectData,
            selectedWhatIfScenarioData[scenario].inner.plot.localZone
        );
    }
}

function resetSelectedWhatIfScenarioInnerData(scenario, zone, innerData, data) {
    innerData.heatmap.weekDay = data.start_date;
    innerData.plot.localZone = scenario === "2nd" ? zone : data.selected_zones[0];
}

function updateWhatIfScenarioInnerData(innerData, data) {
    innerData.heatmap.weekDays = { min_date: data.start_date, max_date: data.end_date };
    innerData.plot.localZones = data.selected_zones;
}

function resetWhatIfScenarioInnerData(innerData) {
    for (const [key, elem] of Object.entries(innerData)) {
        for (const elemkey of Object.keys(elem)) {
            innerData[key][elemkey] = null;
        }
    }
}

async function updateWhatIfScenarioSection(scenario, zone, date, quantity, loadingDescr) {
    const [data, headerMsg] = await runGeneration(scenario, zone, date, quantity);
    if (headerMsg !== null) {
        return headerMsg;
    }

    if (loadingDescr) {
        loadingDescr.innerHTML = "Generation done. Loading results";
    }

    resetWhatIfScenarioInnerData(whatIfScenarioData[scenario].inner);
    updateWhatIfScenarioInnerData(whatIfScenarioData[scenario].inner, data);
    resetSelectedWhatIfScenarioInnerData(
        scenario,
        zone,
        selectedWhatIfScenarioData[scenario].inner,
        data
    );
    populateWhatIfScenarioSelectors(scenario, data);

    const imageMap = {
        heatmap: ["real", "gen"],
        distrib: ["hist", "radar"],
        plot: ["plot"],
    }

    const scenarioDataKind = whatIfScenarioData[scenario].main.dataKind;

    const dataKinds = scenarioDataKind === "both"
        ? ["parkingmeter", "parkingslot"] : [`parking${scenarioDataKind.slice(0, -1)}`];

    const imageElemMap = {}
    for (const [key, elem] of Object.entries(imageMap)) {
        imageElemMap[key] = {};
        for (const dataKind of dataKinds) {
            imageElemMap[key][dataKind] = {};
            for (const elemval of elem) {
                imageElemMap[key][dataKind][elemval] = document.getElementById(
                    `whatif_${scenario}scenario-${key}-${elemval}-${dataKind}s-context`
                );
            }
        }
    }

    const selectedDay = selectedWhatIfScenarioData[scenario].inner.heatmap.weekDay;
    const localZone = selectedWhatIfScenarioData[scenario].inner.plot.localZone;

    const promises = [];

    for (const [section, imageElemDataMap] of Object.entries(imageElemMap)) {
        for (const dataKind of dataKinds) {
            promises.push(updateWhatIfImages(
                zone,
                date,
                scenario,
                quantity,
                section,
                dataKind,
                selectedDay,
                localZone,
                imageElemDataMap[dataKind]
            ));
        }
    }

    await Promise.all(promises);
    return null;
}

function updateWhatIfSectionHeaderBody(scenario, headerMsg) {
    const header = document.getElementById(`whatif_${scenario}scenario-header-content`);
    const body = document.getElementById(`whatif_${scenario}scenario-body-content`);

    if (headerMsg !== null) {
        if (header !== null) {
            header.style.display = "block";
            header.innerHTML = headerMsg;
        }
        if (body !== null) body.style.display = "none";
    } else {
        if (header !== null) {
            header.innerHTML = "";
            header.style.display = "none";
        }
        if (body !== null) body.style.display = "block";
    }
}

function initWhatIfScenarioInnerButton(scenario, section) {
    const imageMap = {
        heatmap: ["real", "gen"],
        distrib: ["hist", "radar"],
        plot: ["plot"],
    }

    const scenarioDataKind = whatIfScenarioData[scenario].main.dataKind;

    const dataKinds = scenarioDataKind === "both"
        ? ["parkingmeter", "parkingslot"] : [`parking${scenarioDataKind.slice(0, -1)}`];

    const scenarioId = `whatif_${scenario}scenario`;
    const sectionId = `${scenarioId}-${section}`;

    const imageElemDataMap = {};
    const elem = imageMap[section];
    for (const dataKind of dataKinds) {
        imageElemDataMap[dataKind] = {};
        for (const elemval of elem) {
            imageElemDataMap[dataKind][elemval] = document.getElementById(
                `${sectionId}-${elemval}-${dataKind}s-context`
            );
        }
    }

    const buttonCategory = (section === "heatmap") ? "weekdate" : "localzone";
    const inputCategory = (section === "heatmap") ? "date" : "localzone";
    const inputType = (section === "heatmap") ? "input" : "select";
    const label = (section === "heatmap") ? "week day" : "local zone";
    const selectedKey = (section === "heatmap") ? "weekDay" : "localZone";

    const submitButton = document.getElementById(
        `submit-${sectionId}-${buttonCategory}-button`
    );
    if (!submitButton) {
        return false;
    }

    const elemSelect = document.getElementById(
        `${sectionId}-${inputCategory}-${inputType}`
    );
    if (!elemSelect) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const value = elemSelect.value;
        if (value === null || value === "default") {
            alert(`Please select a ${label}.`);
            return;
        }
        selectedWhatIfScenarioData[scenario].inner[section][selectedKey] = value;

        const quantity = "quantity" in selectedWhatIfScenarioData[scenario].main
            ? selectedWhatIfScenarioData[scenario].main.quantity : null;

        const weekDate = section === "heatmap" ? value : null;
        const localZone = section === "plot" ? value : null;

        toggleInnerContent(sectionId, false);
        toggleLoadingScreen(sectionId, true);
        toggleButtonElements(scenarioId, false);
        toggleButtonElements(sectionId, false);

        updateMiniMap(scenarioId, "zone", [localZone]);

        const promises = [];

        for (const dataKind of dataKinds) {
            promises.push(updateWhatIfImages(
                scenario !== "3rd" ? selectedWhatIfScenarioData[scenario].main.zone : "0",
                selectedWhatIfScenarioData[scenario].main.date,
                scenario,
                quantity,
                section,
                dataKind,
                weekDate,
                localZone,
                imageElemDataMap[dataKind]
            ));
        }

        await Promise.all(promises);

        toggleLoadingScreen(sectionId, false);
        toggleInnerContent(sectionId, true);
        toggleButtonElements(sectionId, true);
        toggleButtonElements(scenarioId, true);
    });
    return true;
}

function initWhatIfScenarioInnerButtons(scenario) {
    for (const section of Object.keys(selectedWhatIfScenarioData[scenario].inner)) {

        initWhatIfScenarioInnerButton(scenario, section);
    }
}

function toggleWhatIfScenarioInnerButtons(scenario, enable) {
    const scenarioId = `whatif_${scenario}scenario`;

    toggleButtonElements(`${scenarioId}-heatmap`, enable);
    toggleButtonElements(`${scenarioId}-plot`, enable);
}

function initWhatIfScenarioMainButton(scenario) {
    const scenarioId = `whatif_${scenario}scenario`;

    const submitButton = document.getElementById(`submit-${scenarioId}-main-button`);
    if (!submitButton) {
        return false;
    }

    const zoneSelect = document.getElementById(`${scenarioId}-zone-select`);

    const dateInput = document.getElementById(`${scenarioId}-date-input`);
    if (!dateInput) {
        return false;
    }

    const quantitySelect = document.getElementById(`${scenarioId}-quantity-select`);

    submitButton.addEventListener("click", async () => {
        const date = dateInput.value;
        if (!date) {
            alert("Please select a date.");
            return;
        }

        let quantity = quantitySelect ? quantitySelect.value : null;
        if (quantitySelect) {
            if ((!quantity || quantity === "default")) {
                alert("Please select a quantity.");
                return;
            }
            if (Number(quantity) < 30) {
                alert("Please select a quantity not less than 30.");
                return;
            }
        }

        let zone = zoneSelect ? zoneSelect.value : "0";
        if (zoneSelect && (!zone || zone === "default" || zone === "0")) {
            alert("Please select a specific zone.");
            return;
        }

        if (zoneSelect) {
            selectedWhatIfScenarioData[scenario].main.zone = zone;
        }

        selectedWhatIfScenarioData[scenario].main.date = dateInput.value;

        if (quantitySelect) {
            selectedWhatIfScenarioData[scenario].main.quantity = quantity;
        }

        toggleWhatIfScenarioInnerButtons(scenario, false);
        toggleButtonElements(scenarioId, false);
        toggleInnerContent(scenarioId, false);
        const loadingDescr = document.getElementById(`loading-${scenarioId}-description`);
        const prevDescr = loadingDescr ? loadingDescr.innerHTML : "";
        if (loadingDescr) {
            loadingDescr.innerHTML = "Running generation";
        }
        toggleLoadingScreen(scenarioId, true);
        updateMiniMap(scenarioId, "zone", [zone]);
        const headerMsg = await updateWhatIfScenarioSection(scenario, zone, date, quantity, loadingDescr);
        updateWhatIfSectionHeaderBody(scenario, headerMsg);
        toggleLoadingScreen(scenarioId, false);
        if (loadingDescr) {
            loadingDescr.innerHTML = prevDescr;
        }
        toggleInnerContent(scenarioId, true);
        toggleButtonElements(scenarioId, true);
        toggleWhatIfScenarioInnerButtons(scenario, true);
    });
    return true;
}

function initWhatIfScenarioVars(scenario) {
    if (scenario !== "3rd") {
        selectedWhatIfScenarioData[scenario].main.zone = "0";
    }
    selectedWhatIfScenarioData[scenario].main.date = getLastAvailableDate(
        `whatif_${scenario}scenario-date-input`
    );
}

function initWhatIfScenarioMainSelection(scenario) {
    const zoneSelect = document.getElementById(`whatif_${scenario}scenario-zone-select`);
    if (scenario !== "3rd" && !zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById(`whatif_${scenario}scenario-date-input`);
    if (!dateInput) {
        return false;
    }

    const submitButton = document.getElementById(`submit-whatif_${scenario}scenario-main-button`);
    if (!submitButton) {
        return false;
    }

    var zones = Object.assign({}, globalData.zones);
    delete zones["0"];

    if (zoneSelect) {
        populateSelectElement(zoneSelect, zones);
    }
    populateDateInputElement(dateInput, whatIfScenarioData[scenario].main.dates);
}

async function getAvailableWhatIfScenarioDates(scenario) {
    try {
        const data = await fetchData(`get_available_whatif_${scenario}scenario_dates`);
        whatIfScenarioData[scenario].main.dates = data;
        return true;
    } catch (error) {
        handleError(
            `Error fetching available simulate dates: ${error}`,
            "Failed to load available simulate dates. Please try again."
        );
    }
    return false;
}

function initWhatIfScenarioData(scenario) {
    whatIfScenarioData[scenario] = {
        main: {
            dates: null,
            dataKind: scenario === "1st" ? "both" : "slots",
        },
        inner: {
            heatmap: {
                weekDays: null,
            },
            plot: {
                localZones: null,
            },
        },
    };

    selectedWhatIfScenarioData[scenario] = {
        main: {
            date: "",
            ...(scenario === "2nd" ? { quantity: "" } : {}),
            ...(scenario !== "3rd" ? { zone: "" } : {})
        },
        inner: {
            heatmap: {
                weekDay: "",
            },
            plot: {
                localZone: "",
            },
        },
    };
}


async function initWhatIfScenarioContent(scenario) {
    initWhatIfScenarioData(scenario);
    await getAvailableWhatIfScenarioDates(scenario);
    initWhatIfScenarioMainSelection(scenario);

    initMiniMap(`whatif_${scenario}scenario`, globalData.mapData);
    setMiniMapTabUpdateButton(`whatif_${scenario}scenario`, "whatif-tab");

    initWhatIfScenarioVars(scenario);
    initWhatIfScenarioMainButton(scenario);
    initWhatIfScenarioInnerButtons(scenario);
    toggleButtonElements(`whatif_${scenario}scenario`, true);
}

async function initWhatIfTabContent() {
    console.log("Initializing what-if tab content...");

    await Promise.all(
        whatIfScenarios.map(scenario => initWhatIfScenarioContent(scenario))
    );
    console.log("What-If tab content initialized.");
}