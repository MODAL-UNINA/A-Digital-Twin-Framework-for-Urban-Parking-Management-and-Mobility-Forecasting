function populateSelectElement(selectElement, data, defaultKey = "0") {
    if (!selectElement) return false;

    selectElement.innerHTML = "";

    for (const [key, value] of Object.entries(data)) {
        const option = document.createElement("option");
        if (key === defaultKey) {
            option.selected = true;
        }
        option.value = key;
        option.textContent = value;
        selectElement.appendChild(option);
    }
    selectElement.removeAttribute("disabled");

    return true;
}

function populateDateInputElement(inputElement, data, setValueMax = true) {
    if (!inputElement) return false;

    const min_date = data.min_date;
    const max_date = data.max_date;
    inputElement.min = min_date;
    inputElement.max = max_date;
    inputElement.value = setValueMax ? max_date : min_date;
    inputElement.removeAttribute("disabled");

    return true;
}

function addEventListenerToButtonElement(buttonElement, callback) {
    if (!buttonElement) return false;

    buttonElement.addEventListener("click", callback);

    return true;
}

function toggleLoadingScreen(tabname, show) {
    const loadingScreen = document.getElementById(
        `loading-${tabname}-screen`
    );
    if (loadingScreen === null) {
        return;
    }
    if (show) {
        if (!loadingScreen.classList.contains("visible")) {
            loadingScreen.classList.add("visible");
        }
    } else {
        if (loadingScreen.classList.contains("visible")) {
            loadingScreen.classList.remove("visible");
        }
    }
}

function toggleInnerContent(tabname, show) {
    const innerContent = document.getElementById(
        `inner-${tabname}-content`
    );
    if (innerContent === null) {
        return;
    }

    if (show) {
        if (innerContent.classList.contains("hidden")) {
            innerContent.classList.remove("hidden");
        }
    } else {
        if (!innerContent.classList.contains("hidden")) {
            innerContent.classList.add("hidden");
        }
    }
}

function toggleButtonElements(tabname, enable) {
    const buttonElements = document.getElementsByClassName(`submit-${tabname}-button`);

    if (buttonElements.length === 0) {
        return;
    }

    for (const buttonElement of buttonElements) {
        if (enable) {
            buttonElement.removeAttribute("disabled");
        } else {
            buttonElement.setAttribute("disabled", "");
        }
    }
}


function waitForImagesToLoad(tabname, imageIds, callback) {
    toggleLoadingScreen(tabname, true);

    const imageLoadPromises = imageIds.map((id) => {
        return new Promise((resolve, reject) => {
            const img = document.getElementById(id);
            img.onload = resolve;
            img.onerror = reject;
        });
    });

    Promise.allSettled(
        imageLoadPromises
    ).then(() => {
        toggleLoadingScreen(tabname, false);
        callback();
    }).catch((error) => {
        console.error("Error loading one or more images:", error);
        toggleLoadingScreen(tabname, false);
        callback();
    });
}

function toggleInnerContentAfterImages(tabname, imageIds) {
    toggleInnerContent(tabname, false);

    toggleButtonElements(tabname, false);

    waitForImagesToLoad(tabname, imageIds, () => {
        toggleButtonElements(tabname, true);
        toggleInnerContent(tabname, true);
    });
}

function getLastAvailableDate(elementId) {
    const selectedDateInput = document.getElementById(elementId);
    if (selectedDateInput === null) {
        return null;
    }
    return selectedDateInput.max;
}

async function loadImageAsync(imageElement, src) {
    return await new Promise((resolve, reject) => {
        imageElement.onload = resolve;
        imageElement.onerror = reject;
        imageElement.src = src;
    });
}

function createTileLayer(minZoom) {
    return L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        {
            attribution: `&copy;`
                + ` <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>`
                + ` contributors &copy; <a href="https://carto.com/">CARTO</a>`,
            subdomains: "abcd",
            minZoom: minZoom,
        }
    );
}