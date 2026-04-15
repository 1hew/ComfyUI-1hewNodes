export function createLoadVideoDom({ app, node, openFilePicker }) {
    const videoEl = document.createElement("video");
    Object.assign(videoEl.style, {
        width: "100%",
        maxWidth: "100%",
        height: "auto",
        maxHeight: "calc(100% - 20px)",
        display: "block",
        flex: "0 0 auto",
        objectFit: "contain",
        minHeight: "0",
    });

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
    });

    container.appendChild(videoEl);
    container.appendChild(infoEl);
    container.style.cursor = "pointer";

    const fileInputEl = document.createElement("input");
    fileInputEl.type = "file";
    fileInputEl.accept = ".mp4,.webm,.mkv,.mov,.avi,video/*";
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

    return {
        container,
        videoEl,
        infoEl,
        fileInputEl,
        uploadWidget,
    };
}

