function openCalendarImageInNewTab(base64Uri) {
    const newTab = window.open("", "_blank");
    if (!newTab) {
        alert("Unable to open image in a new tab. Please check popup blocker settings.");
        return;
    }

    newTab.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Agents calendar</title>
        </head>
        <body style="margin: 0; padding: 0;">
            <img src="${base64Uri}" alt="Calendar">
        </body>
        </html>
    `);
    newTab.document.close();
}

async function updateCalendarContent(selected_date, headerId, imageContextId) {
    const calendarImageHeader = document.getElementById(headerId);
    if (!calendarImageHeader) return;

    const calendarImage = document.getElementById(imageContextId);
    if (!calendarImage) return;

    try {
        const data = await fetchData("get_calendar", {
            "selected_date": selected_date
        });

        calendarImageHeader.innerHTML = "";

        if (!data.calendar) {
            const error = data.error;
            calendarImageHeader.innerHTML = `<h5>Calendar unavailable. ${error}</h5>`;

            calendarImage.src = "";
            calendarImage.style.display = "none";
            return;
        }

        calendarImageHeader.innerHTML = "<h5>Agents calendar - click to enlarge</h5>";

        const base64Uri = `data:image/png;base64,${data.calendar}`;

        calendarImage.src = base64Uri;
        calendarImage.style.display = "block";

        calendarImage.onclick = null;

        calendarImage.addEventListener("click", () => {
            openCalendarImageInNewTab(base64Uri);
        });

    } catch (error) {
        handleError(
            `Error fetching the calendar: ${error}`,
            "Failed to load calendar. Please try again."
        );
    }
}

async function updateCalendarAllSections(date) {
    await updateCalendarContent(date, "calendar-img-header-context", "calendar-img-context");
}

function initCalendarMainButton() {
    const submitButton = document.getElementById("submit-calendar-main-button");
    if (!submitButton) {
        return false;
    }

    const dateInput = document.getElementById("calendar-date-input");
    if (!dateInput) {
        return false;
    }

    submitButton.addEventListener("click", async () => {
        const date = dateInput.value;
        if (!date) {
            alert("Please select a date.");
            return;
        }

        selectedCalendarData.date = dateInput.value;
        toggleInnerContent("calendar", false);
        toggleInnerCalendarAfterImages();
        await updateCalendarAllSections(selectedCalendarData.date);
    });
    return true;
}

function initCalendarVars() {
    selectedCalendarData.date = getLastAvailableDate("calendar-date-input");
}

function initCalendarMainSelection() {
    const dateInput = document.getElementById("calendar-date-input");
    if (!dateInput) {
        return false;
    }

    const submitButton = document.getElementById("submit-calendar-main-button");
    if (!submitButton) {
        return false;
    }

    populateDateInputElement(dateInput, calendarData.dates);
}

async function getAvailableCalendarDates() {
    try {
        const data = await fetchData("get_available_calendar_dates");
        calendarData.dates = data;
        return true;
    } catch (error) {
        handleError(
            `Error fetching available calendar dates: ${error}`,
            "Failed to load available calendar dates. Please try again."
        );
    }
    return false;
}

function toggleInnerCalendarAfterImages() {
    const imageIds = [
        "calendar-img-context"
    ];

    toggleInnerContentAfterImages("calendar", imageIds);
}

async function initCalendarTabContent() {
    console.log("Initializing calendar tab content...");
    toggleInnerContent("calendar", false);
    toggleInnerCalendarAfterImages();
    await getAvailableCalendarDates();

    initCalendarMainSelection();
    initCalendarVars();
    initCalendarMainButton();

    await updateCalendarAllSections(selectedCalendarData.date);
    console.log("Calendar tab content initialized.");
}