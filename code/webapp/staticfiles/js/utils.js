function handleError(console_message, alert_message) {
    if (console_message) console.error(console_message);
    if (alert_message) alert(alert_message);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchData(action, args = {}) {
    if (!csrfToken) {
        throw new Error("CSRF token not found. Ensure the page contains a valid CSRF token.");
    }

    const options = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrfToken,
            "X-Requested-With": "XMLHttpRequest",
        },
        body: JSON.stringify({ action, args }),
    };

    try {
        const response = await fetch("/", options);

        if (response.redirected) {
            console.warn("Session expired. Redirecting to the login page.");
            window.location.href = response.url;
            return;
        }

        if (!response.ok) {
            const errorDetails = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, Details: ${errorDetails}`);
        }

        return response.json();
    } catch (error) {
        console.error(`Error fetching data "[${action}]": ${error}`);
        throw error;
    }
}

async function checkIfServerIsRunning() {
    const data = await fetchData("check_server_status", {});
    if (!data || !data.status || data.status !== "running") {
        throw new Error("Server is not running.");
    }
}

function getFormattedDate(date) {
    const dateObject = new Date(date);
    const formattedDate = dateObject.toISOString().split("T")[0];
    return formattedDate;
}

function filterByMultipleBounds(items, boundsArray) {
    return items.filter(item => {
        return boundsArray.some(bounds => {
            const lat1 = bounds[0][0], lng1 = bounds[0][1];
            const lat2 = bounds[1][0], lng2 = bounds[1][1];

            const minLat = Math.min(lat1, lat2);
            const maxLat = Math.max(lat1, lat2);
            const minLng = Math.min(lng1, lng2);
            const maxLng = Math.max(lng1, lng2);

            return (
                item.lat >= minLat &&
                item.lat <= maxLat &&
                item.lng >= minLng &&
                item.lng <= maxLng
            );
        });
    });
}