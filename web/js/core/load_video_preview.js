export function createLoadVideoPreviewController({
    app,
    api,
    node,
    videoEl,
    infoEl,
    container,
    videoWidget,
    fileWidget,
    indexWidget,
    includeSubdirWidget,
    updateLayout,
    ensurePreviewLayout,
    resetNodeHeightToBase,
}) {
    const buildPosterUrl = (file, index, includeSubdir) => {
        const params = new URLSearchParams({
            file: String(file || "").trim(),
            index: String(index ?? 0),
            include_subdir: String(includeSubdir),
            t: "0.0",
        });
        return `/1hew/video_frame_from_folder?${params.toString()}`;
    };

    return async function updatePreview() {
        const file = fileWidget.value;
        const index = indexWidget.value;
        const includeSubdir = includeSubdirWidget.value;

        if (videoWidget) {
            videoWidget._comfy1hew_maxPreviewHeight = 220;
        }

        const trimmedFile = String(file || "").trim();
        if (trimmedFile === "") {
            node._comfy1hewVideoInfo = null;
            node._comfy1hewVideoAutoSizeKey = "";
            videoWidget.aspectRatio = undefined;
            infoEl.innerText = "";
            container.style.display = "none";
            videoEl.removeAttribute("poster");
            try {
                videoEl.pause();
            } catch {}
            videoEl.removeAttribute("src");
            try {
                videoEl.load();
            } catch {}
            updateLayout();
            resetNodeHeightToBase?.();
            setTimeout(() => ensurePreviewLayout({ allowShrink: true }), 0);
            return;
        }

        container.style.display = "flex";
        if (infoEl.innerText === "uploading..." || infoEl.innerText === "upload failed") {
            infoEl.innerText = "";
        }
        videoEl.poster = buildPosterUrl(trimmedFile, index, includeSubdir);

        const params = new URLSearchParams({
            file: trimmedFile,
            index,
            include_subdir: includeSubdir,
            audio: "false",
            preview: "true",
            t: Date.now(),
        });

        const url = `/1hew/view_video_from_folder?${params.toString()}`;
        if (videoEl.src.indexOf(url.split("&t=")[0]) === -1) {
            node._comfy1hewVideoAutoSizeKey = "";
            videoEl.src = url;
            setTimeout(
                () => ensurePreviewLayout({ allowShrink: true, forceAutoSize: true }),
                0
            );
            setTimeout(
                () => ensurePreviewLayout({ allowShrink: true, forceAutoSize: true }),
                200
            );
        }

        try {
            const infoParams = new URLSearchParams({
                file: trimmedFile,
                index,
                include_subdir: includeSubdir,
            });
            const infoRes = await api.fetchApi(
                `/1hew/video_info_from_folder?${infoParams.toString()}`,
                { cache: "no-store" }
            );
            if (infoRes && infoRes.status === 200) {
                node._comfy1hewVideoInfo = await infoRes.json();
            } else {
                node._comfy1hewVideoInfo = null;
            }
        } catch {
            node._comfy1hewVideoInfo = null;
        }

        const info = node._comfy1hewVideoInfo;
        const w = Number(info?.width) || 0;
        const h = Number(info?.height) || 0;
        if (w > 0 && h > 0) {
            infoEl.innerText = `${w} x ${h}`;
            videoWidget.aspectRatio = h / w;
            ensurePreviewLayout({ allowShrink: true, forceAutoSize: true });
        }

        if (node.updateVideoPlaybackState) {
            node.updateVideoPlaybackState();
        }

        app.graph.setDirtyCanvas(true, true);
    };
}

