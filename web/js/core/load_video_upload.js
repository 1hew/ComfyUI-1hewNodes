export function createLoadVideoUploader({
    app,
    api,
    node,
    fileWidget,
    indexWidget,
    infoEl,
}) {
    const uploadSingleFile = async (file) => {
        if (!file) {
            return;
        }

        infoEl.innerText = "uploading...";
        const form = new FormData();
        form.append("file", file, file.name);

        const response = await api.fetchApi("/1hew/upload_video", {
            method: "POST",
            body: form,
        });
        if (response.status !== 200) {
            infoEl.innerText = "upload failed";
            return;
        }

        const data = await response.json();
        const newPath = data?.path;
        if (!newPath) {
            infoEl.innerText = "upload failed";
            return;
        }

        if (fileWidget) {
            fileWidget.value = newPath;
            if (typeof fileWidget.callback === "function") {
                fileWidget.callback();
            }
        }
        if (typeof node.updatePreview === "function") {
            await node.updatePreview();
        }

        app.graph.setDirtyCanvas(true, true);
    };

    const uploadFilesAsFolder = async (pairs) => {
        if (!pairs || pairs.length === 0) {
            return;
        }

        infoEl.innerText = "uploading...";
        const form = new FormData();
        for (const pair of pairs) {
            if (!pair?.file) continue;
            const name = pair.relativePath || pair.file.name;
            form.append("files", pair.file, name);
        }

        const response = await api.fetchApi("/1hew/upload_videos", {
            method: "POST",
            body: form,
        });
        if (response.status !== 200) {
            infoEl.innerText = "upload failed";
            return;
        }

        const data = await response.json();
        const folder = data?.folder;
        if (!folder) {
            infoEl.innerText = "upload failed";
            return;
        }

        if (fileWidget) {
            fileWidget.value = folder;
            if (typeof fileWidget.callback === "function") {
                fileWidget.callback();
            }
        }
        if (indexWidget) {
            indexWidget.value = 0;
            if (typeof indexWidget.callback === "function") {
                indexWidget.callback();
            }
        }
        if (typeof node.updatePreview === "function") {
            await node.updatePreview();
        }

        app.graph.setDirtyCanvas(true, true);
    };

    return {
        uploadSingleFile,
        uploadFilesAsFolder,
    };
}

