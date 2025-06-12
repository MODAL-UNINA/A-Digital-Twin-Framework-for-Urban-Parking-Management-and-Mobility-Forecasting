function genTableElement(data, labelMap, borderColor, bordered = true, recursiveIndex = 0) {
    if (recursiveIndex > 1) {
        throw new Error("Recursion limit reached.");
    }
    const responsiveTable = document.createElement("div");
    responsiveTable.classList.add("table-responsive");
    const table = document.createElement("table");
    responsiveTable.appendChild(table);
    table.classList.add("table", "align-middle");
    if (bordered) {
        table.classList.add("table-bordered");
    }
    table.classList.add("table-hover");

    for (const [key, value] of Object.entries(data)) {
        const tr = document.createElement("tr");

        const tdKey = document.createElement("td");
        tdKey.style.borderColor = borderColor;
        tdKey.innerHTML = labelMap[key];
        tdKey.classList.add("fw-bold");

        const tdValue = document.createElement("td");
        tdValue.style.borderColor = borderColor;
        if (value.constructor === Object) {
            const innerTab = genTableElement(value, labelMap, borderColor, false, recursiveIndex + 1);
            innerTab.classList.add("mb-0");
            tdValue.appendChild(innerTab);
        } else {
            tdValue.classList.add("text-end");
            tdValue.textContent = value;
        }

        tr.appendChild(tdKey);
        tr.appendChild(tdValue);
        table.appendChild(tr);
    }
    responsiveTable.style.backgroundColor = "transparent";

    return responsiveTable;
}

function genStatContent(data, labelMap, colorMap) {
    const resRow = getRowElement();

    for (const [key, value] of Object.entries(data)) {
        const resKeyCol = getColElement(resRow, [[null, 12], ["xxl", 6]]);
        resKeyCol.classList.add("p-1");

        const [resCard, resCardBody] = getContentContainer();
        resKeyCol.appendChild(resCard);
        resCard.classList.add("h-100");

        const [resHeaderRow, resHeaderCol] = getRowColElement();
        resHeaderCol.innerHTML = `<h5>${labelMap[key]}</h5>`;
        resHeaderRow.classList.add("mb-2");

        const [resContentRow, resContentCol] = getRowColElement();

        resContentCol.appendChild(genTableElement(value, labelMap, colorMap[key].border_color));

        resCardBody.appendChild(resHeaderRow);
        resCardBody.appendChild(resContentRow);
        resCardBody.style.backgroundColor = colorMap[key].background_color;
    }

    return resRow;
}

async function updateStatistics(
    zone,
    date,
    hourslot,
    infoId,
    imageRelId,
    imageAbsId,
    imageRelSectionId,
    imageAbsSectionId
) {
    const statsImageRel = document.getElementById(imageRelId);
    if (!statsImageRel) return;

    const statsImageAbs = document.getElementById(imageAbsId);
    if (!statsImageAbs) return;

    const statsInfoContent = document.getElementById(infoId);
    if (!statsInfoContent) return;

    const statsImageRelSection = document.getElementById(imageRelSectionId);
    if (!statsImageRelSection) return;

    const statsImageAbsSection = document.getElementById(imageAbsSectionId);
    if (!statsImageAbsSection) return;

    statsInfoContent.innerHTML = "";

    try {
        const data = await fetchData("get_stats_info", {
            "zone_id": zone,
            "hourslot": hourslot,
            "selected_date": date
        });

        if (!data || !data.stats_data) {
            throw new Error("Statistics data not found.");
        }

        const statsData = data.stats_data;

        if (statsData.statsImgRel) {
            statsImageRelSection.style.display = "block";
            statsImageRel.src =
                `data:image/png;base64,${statsData.statsImgRel}`;
            statsImageRel.style.display = "block";
        } else {
            statsImageRelSection.style.display = "none";
            statsImageRel.src = "";
            statsImageRel.style.display = "none";
        }

        if (statsData.statsImgAbs) {
            statsImageAbsSection.style.display = "block";
            statsImageAbs.src =
                `data:image/png;base64,${statsData.statsImgAbs}`;
            statsImageAbs.style.display = "block";
        } else {
            statsImageAbsSection.style.display = "none";
            statsImageAbs.src = "";
            statsImageAbs.style.display = "none";
        }


        const generalStats = statsData.generalStats;
        const abusivismStats = statsData.abusivismStats;
        const statsLabelMap = statsData.labelMap;
        const statsColorMap = statsData.colorMap;

        if (generalStats) {
            statsInfoContent.appendChild(genStatContent(
                generalStats, statsLabelMap, statsColorMap));
        }
        if (abusivismStats) {
            statsInfoContent.appendChild(genStatContent(
                abusivismStats, statsLabelMap, statsColorMap));
        }
    } catch (error) {
        handleError(
            `Error fetching statistics: ${error}`,
            "Failed to load statistics. Please try again."
        );
    }
}

async function updateStatsAllSections(zone, date, hourSlot) {
    await updateStatistics(
        zone,
        date,
        hourSlot,
        "stats-info-content",
        "stats-img-rel-context",
        "stats-img-abs-context",
        "stats-img-rel-section",
        "stats-img-abs-section"
    );
}

function initStatsMainButton() {
    const submitButton = document.getElementById("submit-stats-main-button");
    if (!submitButton) {
        return false;
    }

    const zoneSelect = document.getElementById("stats-zone-select");
    if (!zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById("stats-date-input");
    if (!dateInput) {
        return false;
    }

    const hourSlotSelect = document.getElementById("stats-hourslot-select");
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

        selectedStatsData.zone = zoneSelect.value;
        selectedStatsData.date = dateInput.value;
        selectedStatsData.hourSlot = hourSlotSelect.value;
        toggleInnerContent("stats", false);
        toggleInnerStatsAfterImages();
        updateMiniMap("stats", "zone", [selectedStatsData.zone]);
        await updateStatsAllSections(
            selectedStatsData.zone,
            selectedStatsData.date,
            selectedStatsData.hourSlot
        );
    });
    return true;
}

function initStatsVars() {
    selectedStatsData.zone = "0";
    selectedStatsData.date = getLastAvailableDate("stats-date-input");
    selectedStatsData.hourSlot = "0";
}

function initStatsMainSelection() {
    const zoneSelect = document.getElementById("stats-zone-select");
    if (!zoneSelect) {
        return false;
    }

    const dateInput = document.getElementById("stats-date-input");
    if (!dateInput) {
        return false;
    }

    const hourSlotSelect = document.getElementById("stats-hourslot-select");
    if (!hourSlotSelect) {
        return false;
    }

    const submitButton = document.getElementById("submit-stats-main-button");
    if (!submitButton) {
        return false;
    }

    populateSelectElement(zoneSelect, globalData.zones);
    populateDateInputElement(dateInput, statsData.dates);
    populateSelectElement(hourSlotSelect, globalData.hourSlots);
}

async function getAvailableStatsDates() {
    try {
        const data = await fetchData("get_available_stats_dates");
        statsData.dates = data;
        return true;
    } catch (error) {
        handleError(
            `Error fetching available statistics dates: ${error}`,
            "Failed to load available statistics dates. Please try again."
        );
    }
    return false;
}

function toggleInnerStatsAfterImages() {
    const imageIds = [
        "stats-img-rel-context",
        "stats-img-abs-context"
    ];

    toggleInnerContentAfterImages("stats", imageIds);
}

async function initStatsTabContent() {
    console.log("Initializing statistics tab content...");
    toggleInnerContent("stats", false);
    toggleInnerStatsAfterImages();
    await getAvailableStatsDates();

    initStatsMainSelection();

    initMiniMap("stats", globalData.mapData, false);

    initStatsVars();
    initStatsMainButton();

    await updateStatsAllSections(
        selectedStatsData.zone,
        selectedStatsData.date,
        selectedStatsData.hourSlot
    );
    console.log("Statistics tab content initialized.");
}
