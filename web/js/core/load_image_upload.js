function isValidImageFile(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith("image/")) return true;
    const name = (file.name || "").toLowerCase();
    return (
        name.endsWith(".png")
        || name.endsWith(".jpg")
        || name.endsWith(".jpeg")
        || name.endsWith(".webp")
        || name.endsWith(".bmp")
        || name.endsWith(".tiff")
        || name.endsWith(".gif")
    );
}

export function createLoadImageUploader({
    app,
    api,
    node,
    fileWidget,
    indexWidget,
    includeSubdirWidget,
    computeStateKey,
}) {
    const uploadFilesAsFolder = async (pairs, hasDirectory) => {
        const files = (pairs || []).filter((pair) => isValidImageFile(pair?.file));
        if (files.length === 0) {
            return;
        }

        const form = new FormData();
        for (const pair of files) {
            const name = pair.relativePath || pair.file.name;
            form.append("files", pair.file, name);
        }

        const response = await api.fetchApi("/1hew/upload_images", {
            method: "POST",
            body: form,
        });
        if (response.status !== 200) {
            return;
        }

        const data = await response.json();
        const uploadedFolder = data?.folder;
        const uploadedFiles = data?.files;
        if (!uploadedFolder && (!uploadedFiles || uploadedFiles.length === 0)) {
            return;
        }

        let finalPath = uploadedFolder;
        if (Array.isArray(uploadedFiles) && uploadedFiles.length === 1) {
            finalPath = uploadedFiles[0];
        }

        if (fileWidget) {
            fileWidget.value = finalPath;
        }
        if (indexWidget) {
            indexWidget.value = 0;
        }
        if (includeSubdirWidget && hasDirectory) {
            includeSubdirWidget.value = true;
        }

        node._comfy1hewLoadImageStateKey = computeStateKey();
        if (typeof node.updatePreview === "function") {
            await node.updatePreview();
        }

        app.graph.setDirtyCanvas(true, true);
    };

    return {
        isValidImageFile,
        uploadFilesAsFolder,
    };
}

