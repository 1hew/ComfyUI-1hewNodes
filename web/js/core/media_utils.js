const managedVideos = new Set();
let resumeHooksInstalled = false;

function installResumeHooks() {
    if (resumeHooksInstalled) {
        return;
    }
    resumeHooksInstalled = true;

    const resumeAll = () => {
        for (const videoEl of managedVideos) {
            if (!videoEl) {
                continue;
            }
            if (videoEl.dataset.comfy1hewUserPaused === "1") {
                continue;
            }
            const p = videoEl.play();
            if (p && typeof p.catch === "function") {
                p.catch(() => {});
            }
        }
    };

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            return;
        }
        resumeAll();
    });
    window.addEventListener("focus", resumeAll);
    window.addEventListener("pageshow", resumeAll);
}

export function applyLoopedHoverAudioPreview(videoEl) {
    if (!videoEl || videoEl.dataset.comfy1hewPreviewApplied === "1") {
        return;
    }
    videoEl.dataset.comfy1hewPreviewApplied = "1";

    installResumeHooks();
    managedVideos.add(videoEl);

    videoEl.autoplay = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.controls = false;
    videoEl.preload = "auto";

    const safePlay = () => {
        if (videoEl.dataset.comfy1hewUserPaused === "1") {
            return;
        }
        if (videoEl.dataset.comfy1hewFrameAccurate === "1") {
            return;
        }
        const p = videoEl.play();
        if (p && typeof p.catch === "function") {
            p.catch(() => {});
        }
    };

    safePlay();
    videoEl.addEventListener("loadeddata", safePlay);
    videoEl.addEventListener("canplay", safePlay);
    videoEl.addEventListener("playing", () => managedVideos.add(videoEl));

    videoEl.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (videoEl.dataset.comfy1hewFrameAccurate === "1") {
            if (videoEl.dataset.comfy1hewUserPaused === "1") {
                videoEl.dataset.comfy1hewUserPaused = "0";
                if (videoEl.dataset.comfy1hewForceMute !== "1") {
                    videoEl.muted = false;
                    videoEl.volume = 1.0;
                }
            } else {
                videoEl.dataset.comfy1hewUserPaused = "1";
                videoEl.muted = true;
            }
            return;
        }
        if (videoEl.paused) {
            videoEl.dataset.comfy1hewUserPaused = "0";
            if (videoEl.dataset.comfy1hewForceMute !== "1") {
                videoEl.muted = false;
                videoEl.volume = 1.0;
            }
            safePlay();
        } else {
            videoEl.dataset.comfy1hewUserPaused = "1";
            videoEl.pause();
        }
    });

    videoEl.addEventListener("pointerenter", () => {
        if (videoEl.dataset.comfy1hewUserPaused === "1") {
            return;
        }
        if (videoEl.dataset.comfy1hewForceMute === "1") {
            videoEl.muted = true;
        } else {
            videoEl.muted = false;
            videoEl.volume = 1.0;
        }
        if (videoEl.dataset.comfy1hewFrameAccurate === "1") {
            return;
        }
        safePlay();
    });

    videoEl.addEventListener("pointerleave", () => {
        videoEl.muted = true;
    });
}

export async function collectDropPayload(e) {
    const readAllEntries = async (dirEntry) => {
        const reader = dirEntry.createReader();
        const entries = [];
        while (true) {
            const batch = await new Promise((resolve) => reader.readEntries(resolve));
            if (!batch || batch.length === 0) break;
            entries.push(...batch);
        }
        return entries;
    };

    const walkEntry = async (entry) => {
        if (!entry) return [];
        if (entry.isFile) {
            const file = await new Promise((resolve) => entry.file(resolve));
            const relativePath = (entry.fullPath || file.name).replace(/^\/+/, "").trim();
            return [{ file, relativePath }];
        }
        if (entry.isDirectory) {
            const entries = await readAllEntries(entry);
            const out = [];
            for (const e2 of entries) {
                const sub = await walkEntry(e2);
                out.push(...sub);
            }
            return out;
        }
        return [];
    };

    const items = e?.dataTransfer?.items;
    if (items && items.length > 0) {
        const out = [];
        let hasDirectory = false;
        for (const item of items) {
            const entry = item?.webkitGetAsEntry?.();
            if (!entry) continue;
            if (entry.isDirectory) hasDirectory = true;
            const sub = await walkEntry(entry);
            out.push(...sub);
        }
        return { pairs: out, hasDirectory };
    }

    const files = Array.from(e?.dataTransfer?.files || []);
    return {
        pairs: files.map((f) => ({ file: f, relativePath: f.name })),
        hasDirectory: false,
    };
}

function getTargetNodesForSave(app, currentNode) {
    const canvas = app?.canvas;
    const selected = canvas?.selected_nodes || {};
    const selection = Object.values(selected);
    if (selection.length > 0 && selection.includes(currentNode)) {
        return selection;
    }
    return [currentNode];
}

function downloadUrl(url, filename) {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

export function addSaveMediaMenuOption(
    options,
    {
        app,
        currentNode,
        content,
        getMediaElFromNode,
        filenamePrefix,
        filenameExt,
    },
) {
    if (!Array.isArray(options) || !app || !currentNode || !getMediaElFromNode) {
        return;
    }
    const mediaEl = getMediaElFromNode(currentNode);
    if (!mediaEl || !mediaEl.src) {
        return;
    }

    options.push({
        content,
        callback: () => {
            const targets = getTargetNodesForSave(app, currentNode);
            for (const node of targets) {
                const el = getMediaElFromNode(node);
                if (el && el.src) {
                    downloadUrl(
                        el.src,
                        `${filenamePrefix}_${node.id}_${Date.now()}.${filenameExt}`,
                    );
                }
            }
        },
    });
}

export function addCopyMediaFrameMenuOption(
    options,
    {
        content,
        getWidth,
        getHeight,
        drawToCanvas,
        copyErrorMessage,
        prepareErrorMessage,
    },
) {
    if (!Array.isArray(options)) {
        return;
    }
    if (
        typeof getWidth !== "function"
        || typeof getHeight !== "function"
        || typeof drawToCanvas !== "function"
    ) {
        return;
    }

    const width = getWidth();
    const height = getHeight();
    if (!width || !height) {
        return;
    }

    options.push({
        content,
        callback: () => {
            try {
                const canvas = document.createElement("canvas");
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext("2d");
                drawToCanvas(ctx, canvas);

                canvas.toBlob((blob) => {
                    if (blob) {
                        try {
                            const item = new ClipboardItem({ "image/png": blob });
                            navigator.clipboard.write([item]);
                        } catch (err) {
                            console.error(copyErrorMessage, err);
                        }
                    }
                }, "image/png");
            } catch (err) {
                console.error(prepareErrorMessage, err);
            }
        },
    });
}

export function installImagePreviewLayout({
    app,
    node,
    imageWidget,
    container,
    imageEl,
    allWidget,
}) {
    const autoSizeToContent = () => {
        if (!imageWidget.aspectRatio) {
            return;
        }

        const maxAspectRatio =
            typeof imageWidget._comfy1hew_maxAspectRatio === "number"
                ? imageWidget._comfy1hew_maxAspectRatio
                : null;
        const aspectRatio =
            maxAspectRatio && isFinite(maxAspectRatio)
                ? Math.min(imageWidget.aspectRatio, maxAspectRatio)
                : imageWidget.aspectRatio;

        const width = node.size[0];
        const desiredWidgetHeight = width * aspectRatio + 20;

        let desiredHeight;
        if (Number.isFinite(imageWidget.last_y)) {
            desiredHeight = imageWidget.last_y + desiredWidgetHeight;
        } else {
            try {
                const computed = node.computeSize?.([
                    node.size[0],
                    node.size[1],
                ]);
                if (
                    Array.isArray(computed) &&
                    computed.length >= 2 &&
                    Number.isFinite(computed[1])
                ) {
                    desiredHeight = computed[1];
                }
            } catch {}
            if (!Number.isFinite(desiredHeight)) {
                const estimatedTop = 130;
                desiredHeight = estimatedTop + desiredWidgetHeight;
            }
        }

        if (allWidget && allWidget.value) {
            desiredHeight = Math.min(desiredHeight, 420);
        }

        if (node.size[1] + 1 < desiredHeight) {
            node.setSize([node.size[0], desiredHeight]);
        }
    };

    const requestAutoSize = () => {
        if (!imageEl.naturalWidth || !imageEl.naturalHeight) {
            return;
        }
        const key = `${imageEl.naturalWidth}x${imageEl.naturalHeight}`;
        if (node._comfy1hewImageAutoSizeKey === key) {
            return;
        }
        node._comfy1hewImageAutoSizeKey = key;
        setTimeout(autoSizeToContent, 0);
    };

    const updateLayout = () => {
        if (container.dataset.comfy1hewForceHidden === "1") {
            container.style.height = "0px";
            container.style.display = "none";
            app.graph.setDirtyCanvas(true, true);
            return;
        }
        if (!imageWidget?.aspectRatio) {
            container.style.height = "0px";
            container.style.display = "none";
            app.graph.setDirtyCanvas(true, true);
            return;
        }
        container.style.display = "flex";

        let availableHeight;
        if (Number.isFinite(imageWidget?.last_y)) {
            availableHeight = node.size[1] - imageWidget.last_y - 15;
        } else {
            const maxAspectRatio =
                typeof imageWidget._comfy1hew_maxAspectRatio === "number"
                    ? imageWidget._comfy1hew_maxAspectRatio
                    : null;
            const aspectRatio =
                maxAspectRatio && isFinite(maxAspectRatio)
                    ? Math.min(imageWidget.aspectRatio, maxAspectRatio)
                    : imageWidget.aspectRatio;
            const width = node.size[0];
            availableHeight = width * aspectRatio + 20;
        }

        if (availableHeight < 0) availableHeight = 0;
        container.style.height = `${availableHeight}px`;

        app.graph.setDirtyCanvas(true, true);
    };

    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        const r2 = originalOnResize
            ? originalOnResize.apply(this, arguments)
            : undefined;
        try {
            updateLayout();
        } catch {}
        return r2;
    };

    return { autoSizeToContent, requestAutoSize, updateLayout };
}

export function installVideoPreviewLayout({
    app,
    node,
    videoWidget,
    container,
    videoEl,
}) {
    const autoSizeToContent = () => {
        if (!videoWidget.aspectRatio) {
            return;
        }

        const width = node.size[0];
        const desiredWidgetHeight = width * videoWidget.aspectRatio + 20;

        let desiredHeight;
        if (Number.isFinite(videoWidget.last_y)) {
            desiredHeight = videoWidget.last_y + desiredWidgetHeight;
        } else {
            try {
                const computed = node.computeSize?.([
                    node.size[0],
                    node.size[1],
                ]);
                if (
                    Array.isArray(computed) &&
                    computed.length >= 2 &&
                    Number.isFinite(computed[1])
                ) {
                    desiredHeight = computed[1];
                }
            } catch {}
            if (!Number.isFinite(desiredHeight)) {
                let estimatedTop = 130;
                if (node.widgets && node.widgets.length > 0) {
                    const widgetCount = node.widgets.filter(
                        (w) => w !== videoWidget && w.type !== "hidden",
                    ).length;
                    estimatedTop = 30 + widgetCount * 26 + 30;
                }
                if (estimatedTop < 130) estimatedTop = 130;
                desiredHeight = estimatedTop + desiredWidgetHeight;
            }
        }

        if (node.size[1] + 1 < desiredHeight) {
            node.setSize([node.size[0], desiredHeight]);
        }
    };

    const requestAutoSize = () => {
        if (!videoEl.videoWidth || !videoEl.videoHeight) {
            return;
        }
        const key = `${videoEl.videoWidth}x${videoEl.videoHeight}`;
        if (node._comfy1hewVideoAutoSizeKey === key) {
            return;
        }
        node._comfy1hewVideoAutoSizeKey = key;
        setTimeout(autoSizeToContent, 0);
    };

    const updateLayout = () => {
        if (container.dataset.comfy1hewForceHidden === "1") {
            container.style.height = "0px";
            container.style.display = "none";
            return;
        }
        if (!videoWidget.aspectRatio) {
            container.style.height = "0px";
            container.style.display = "none";
            return;
        }
        container.style.display = "flex";

        let availableHeight;
        if (Number.isFinite(videoWidget?.last_y)) {
            availableHeight = node.size[1] - videoWidget.last_y - 15;
        } else {
            const width = node.size[0];
            availableHeight = width * videoWidget.aspectRatio + 20;
        }

        if (availableHeight < 0) availableHeight = 0;
        container.style.height = `${availableHeight}px`;

        app.graph.setDirtyCanvas(true, true);
    };

    const ensurePreviewLayout = () => {
        if (container.dataset.comfy1hewForceHidden === "1") {
            updateLayout();
            return;
        }

        if (videoEl.videoWidth && videoEl.videoHeight) {
            videoWidget.aspectRatio = videoEl.videoHeight / videoEl.videoWidth;
        }

        requestAutoSize();
        updateLayout();
        autoSizeToContent();
    };

    node.updateVideoLayout = updateLayout;
    node._comfy1hewEnsurePreviewLayout = ensurePreviewLayout;

    return { autoSizeToContent, requestAutoSize, updateLayout, ensurePreviewLayout };
}

