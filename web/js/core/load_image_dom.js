export function createLoadImageDom({ app, node, openFilePicker }) {
    const imageEl = document.createElement("img");
    Object.assign(imageEl.style, {
        width: "100%",
        maxWidth: "100%",
        height: "100%",
        maxHeight: "100%",
        display: "block",
        objectFit: "contain",
        minHeight: "0",
        pointerEvents: "none",
    });

    const previewViewport = document.createElement("div");
    Object.assign(previewViewport.style, {
        width: "100%",
        flex: "1 1 auto",
        minHeight: "0",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        pointerEvents: "none",
    });
    previewViewport.appendChild(imageEl);

    const infoEl = document.createElement("div");
    Object.assign(infoEl.style, {
        width: "100%",
        height: "20px",
        lineHeight: "20px",
        textAlign: "center",
        fontSize: "12px",
        color: "#aaa",
        fontFamily: "sans-serif",
        flex: "0 0 20px",
        background: "transparent",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        pointerEvents: "none",
    });
    infoEl.innerText = "";

    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "none",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        minHeight: "0",
        overflow: "hidden",
        boxSizing: "border-box",
        pointerEvents: "none",
    });

    container.appendChild(previewViewport);
    container.appendChild(infoEl);

    const fileInputEl = document.createElement("input");
    fileInputEl.type = "file";
    fileInputEl.accept = "image/*";
    fileInputEl.style.display = "none";
    container.appendChild(fileInputEl);

    const uploadWidget = node.addWidget(
        "button",
        "choose file to upload",
        "image",
        () => {
            app.canvas.node_widget = null;
            openFilePicker();
        }
    );
    uploadWidget.serialize = false;

    imageEl.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        if (app.canvas) {
            app.canvas.processContextMenu(node, event);
        }
    });

    return {
        container,
        imageEl,
        infoEl,
        fileInputEl,
        uploadWidget,
    };
}

