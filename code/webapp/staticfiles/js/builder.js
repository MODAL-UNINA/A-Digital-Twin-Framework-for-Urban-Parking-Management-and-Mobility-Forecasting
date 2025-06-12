function getCardBody(cardContainer) {
    const cardBody = document.createElement("div");
    cardBody.classList.add("card-body");

    cardContainer.appendChild(cardBody);
    return cardBody;
}

function getCardContainer(kind) {
    const cardContainer = document.createElement("div");
    cardContainer.classList.add("card", `card-${kind}`);

    const cardBody = getCardBody(cardContainer);
    return [cardContainer, cardBody];
}

function getSubmitContainer() {
    return getCardContainer("well");
}

function getContentContainer() {
    return getCardContainer("container");
}

function getRowElement() {
    const resElement = document.createElement("div");
    resElement.classList.add("row");
    return resElement;
}

function getColElement(rowElement, colClasses = [[null, null]]) {
    const resElement = document.createElement("div");
    for (const cls of colClasses) {
        const [breakpoint, size] = cls;
        let colClass = "col";
        if (breakpoint !== null) {
            colClass = `${colClass}-${breakpoint}`;
        }
        if (size !== null) {
            colClass = `${colClass}-${size}`;
        }
        resElement.classList.add(colClass);
    }

    rowElement.appendChild(resElement);
    return resElement;
}

function getRowColElement(colClasses = [[null, null]]) {
    const resRow = getRowElement();
    const resCol = getColElement(resRow, colClasses);
    return [resRow, resCol];
}
