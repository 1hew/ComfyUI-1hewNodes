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

function estimatePreviewTopOffset(node, widget) {
    try {
        if (node.widgets && node.widgets.length > 0) {
            const widgetCount = node.widgets.filter(
                (currentWidget) =>
                    currentWidget !== widget && currentWidget.type !== "hidden",
            ).length;
            let estimatedTop = 30 + widgetCount * 26 + 30;
            if (estimatedTop < 130) estimatedTop = 130;
            return estimatedTop;
        }
    } catch {}
    return 130;
}

function getDesiredNodeHeight(node, widget, desiredWidgetHeight) {
    if (Number.isFinite(widget?.last_y)) {
        return widget.last_y + desiredWidgetHeight;
    }

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
            return computed[1];
        }
    } catch {}

    return estimatePreviewTopOffset(node, widget) + desiredWidgetHeight;
}

function bindPreviewResizeHandler(node, {
    prevWidthKey,
    userResizedKey,
    internalResizeFlagKey,
    updateLayout,
}) {
    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        const result = originalOnResize
            ? originalOnResize.apply(this, arguments)
            : undefined;
        try {
            const nextWidth = Array.isArray(size) ? size[0] : node.size?.[0];
            const prevWidth = node[prevWidthKey];
            node[prevWidthKey] = nextWidth;

            if (
                !node[internalResizeFlagKey]
                && Number.isFinite(nextWidth)
                && Number.isFinite(prevWidth)
                && Math.abs(nextWidth - prevWidth) > 1
            ) {
                node[userResizedKey] = true;
            }

            updateLayout();
        } catch {}
        return result;
    };
}

function setPreviewVisibility(container, {
    visible,
    height = 0,
    app,
}) {
    container.style.height = `${height}px`;
    container.style.display = visible ? "flex" : "none";
    app?.graph?.setDirtyCanvas?.(true, true);
}

function getAvailablePreviewHeight(node, widget, aspectRatio, maxPreviewHeight) {
    let availableHeight;
    if (Number.isFinite(widget?.last_y)) {
        availableHeight = node.size[1] - widget.last_y - 15;
    } else {
        const width = node.size[0];
        availableHeight = width * aspectRatio + 20;
    }

    if (maxPreviewHeight && isFinite(maxPreviewHeight)) {
        availableHeight = Math.min(availableHeight, maxPreviewHeight);
    }

    if (availableHeight < 0) {
        return 0;
    }
    return availableHeight;
}

function getCappedAspectRatio(aspectRatio, maxAspectRatio = null) {
    if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
        return null;
    }
    if (maxAspectRatio && isFinite(maxAspectRatio)) {
        return Math.min(aspectRatio, maxAspectRatio);
    }
    return aspectRatio;
}

function syncPreviewAspectRatio(widget, {
    width,
    height,
    maxAspectRatio = null,
}) {
    if (!width || !height) {
        return null;
    }
    const nextAspectRatio = getCappedAspectRatio(height / width, maxAspectRatio);
    if (nextAspectRatio) {
        widget.aspectRatio = nextAspectRatio;
    }
    return nextAspectRatio;
}

function resizeNodeToPreviewContent({
    node,
    widget,
    aspectRatio,
    maxPreviewHeight,
    allowShrink = false,
    internalResizeFlagKey,
    adjustDesiredHeight,
}) {
    if (!aspectRatio) {
        return;
    }

    const width = node.size[0];
    let desiredWidgetHeight = width * aspectRatio + 20;
    if (maxPreviewHeight && isFinite(maxPreviewHeight)) {
        desiredWidgetHeight = Math.min(desiredWidgetHeight, maxPreviewHeight);
    }

    let desiredHeight = getDesiredNodeHeight(
        node,
        widget,
        desiredWidgetHeight,
    );

    if (typeof adjustDesiredHeight === "function") {
        desiredHeight = adjustDesiredHeight(desiredHeight);
    }

    if (
        (allowShrink && Math.abs(node.size[1] - desiredHeight) > 1)
        || (!allowShrink && node.size[1] + 1 < desiredHeight)
    ) {
        node[internalResizeFlagKey] = true;
        try {
            node.setSize([node.size[0], desiredHeight]);
        } finally {
            node[internalResizeFlagKey] = false;
        }
    }
}

function requestPreviewAutoSize({
    node,
    width,
    height,
    autoSizeKey,
    autoSizeToContent,
    allowShrink = false,
    force = false,
}) {
    if (!width || !height) {
        return;
    }
    const key = `${width}x${height}`;
    if (!force && node[autoSizeKey] === key) {
        return;
    }
    node[autoSizeKey] = key;
    setTimeout(() => autoSizeToContent({ allowShrink }), 0);
}

function updatePreviewContainerLayout({
    app,
    container,
    previewStrategy,
}) {
    if (container.dataset.comfy1hewForceHidden === "1") {
        setPreviewVisibility(container, { visible: false, app });
        return;
    }

    if (!previewStrategy.getAspectRatio()) {
        previewStrategy.syncAspectRatioFromMedia();
    }

    if (!previewStrategy.getAspectRatio()) {
        const placeholderState =
            typeof previewStrategy.getPlaceholderState === "function"
                ? previewStrategy.getPlaceholderState()
                : null;
        if (placeholderState) {
            setPreviewVisibility(container, { ...placeholderState, app });
            return;
        }
        setPreviewVisibility(container, { visible: false, app });
        return;
    }

    setPreviewVisibility(container, {
        ...previewStrategy.getVisibleState(),
        app,
    });
}

export function installImagePreviewLayout({
    app,
    node,
    imageWidget,
    container,
    imageEl,
    allWidget,
    fileWidget,
}) {
    const getMaxAspectRatio = () =>
        typeof imageWidget._comfy1hew_maxAspectRatio === "number"
            ? imageWidget._comfy1hew_maxAspectRatio
            : null;

    const getMaxPreviewHeight = () => {
        const maxPreviewHeight =
            typeof imageWidget._comfy1hew_maxPreviewHeight === "number"
                ? imageWidget._comfy1hew_maxPreviewHeight
                : null;
        if (
            node._comfy1hewImagePreviewUserResized
            && !(allWidget && allWidget.value)
        ) {
            return null;
        }
        return maxPreviewHeight;
    };

    const previewStrategy = {
        getAspectRatio() {
            return getCappedAspectRatio(
                imageWidget.aspectRatio,
                getMaxAspectRatio(),
            );
        },
        syncAspectRatioFromMedia() {
            return syncPreviewAspectRatio(imageWidget, {
                width: imageEl?.naturalWidth,
                height: imageEl?.naturalHeight,
                maxAspectRatio: getMaxAspectRatio(),
            });
        },
        getPlaceholderState() {
            const hasSrc = Boolean(
                imageEl?.src && String(imageEl.src).trim() !== "",
            );
            const stillDecoding =
                hasSrc
                && (!imageEl.naturalWidth || !imageEl.naturalHeight);
            const filePathTrim = String(fileWidget?.value ?? "").trim();
            const awaitingFileLoad = Boolean(filePathTrim && !hasSrc);
            const holdPlaceholder =
                stillDecoding
                || awaitingFileLoad
                || Number.isFinite(node._comfy1hewPreserveFrameHeightUntilPreview);
            if (!holdPlaceholder) {
                return null;
            }
            return {
                visible: true,
                height: computePlaceholderPreviewHeight(),
            };
        },
        getVisibleState() {
            const availableHeight = getAvailablePreviewHeight(
                node,
                imageWidget,
                this.getAspectRatio(),
                getMaxPreviewHeight(),
            );
            return {
                visible: true,
                height: availableHeight,
            };
        },
    };

    const autoSizeToContent = ({ allowShrink = false } = {}) => {
        const aspectRatio = previewStrategy.getAspectRatio();
        resizeNodeToPreviewContent({
            node,
            widget: imageWidget,
            aspectRatio,
            maxPreviewHeight: getMaxPreviewHeight(),
            allowShrink,
            internalResizeFlagKey: "_comfy1hewInternalPreviewResize",
            adjustDesiredHeight: (desiredHeight) => {
                let nextDesiredHeight = desiredHeight;
                if (allWidget && allWidget.value) {
                    nextDesiredHeight = Math.min(nextDesiredHeight, 420);
                }

                const preserve = node._comfy1hewPreserveFrameHeightUntilPreview;
                if (
                    Number.isFinite(preserve)
                    && Number.isFinite(nextDesiredHeight)
                    && nextDesiredHeight < preserve
                ) {
                    nextDesiredHeight = preserve;
                }
                return nextDesiredHeight;
            },
        });
    };

    const requestAutoSize = ({ allowShrink = false, force = false } = {}) => {
        requestPreviewAutoSize({
            node,
            width: imageEl.naturalWidth,
            height: imageEl.naturalHeight,
            autoSizeKey: "_comfy1hewImageAutoSizeKey",
            autoSizeToContent,
            allowShrink,
            force,
        });
    };

    node._comfy1hewLastPreviewNodeWidth ??= node.size?.[0];

    const computePlaceholderPreviewHeight = () => {
        let placeholderH = 100;
        if (Number.isFinite(imageWidget?.last_y)) {
            placeholderH = Math.max(
                0,
                node.size[1] - imageWidget.last_y - 15,
            );
        } else if (Array.isArray(node.size) && node.size[1] > 140) {
            const baseGuess = 130;
            placeholderH = Math.max(
                40,
                Math.min(220, node.size[1] - baseGuess),
            );
        }
        if (placeholderH < 40) {
            placeholderH = 100;
        }
        return placeholderH;
    };

    const updateLayout = () => {
        // 复制节点 / 刚设置 src 时：DOM 上可能已有 <img> 尺寸，但 widget.aspectRatio 尚未写入。
        // 若此时走「无比例则 display:none」，会出现整段预览被关掉再打开的可见闪烁。
        updatePreviewContainerLayout({
            app,
            container,
            previewStrategy,
        });
    };

    bindPreviewResizeHandler(node, {
        prevWidthKey: "_comfy1hewLastPreviewNodeWidth",
        userResizedKey: "_comfy1hewImagePreviewUserResized",
        internalResizeFlagKey: "_comfy1hewInternalPreviewResize",
        updateLayout,
    });

    const ensurePreviewLayout = ({
        allowShrink = false,
        forceAutoSize = false,
    } = {}) => {
        if (container.dataset.comfy1hewForceHidden === "1") {
            updateLayout();
            return;
        }

        previewStrategy.syncAspectRatioFromMedia();

        requestAutoSize({ allowShrink, force: forceAutoSize });
        updateLayout();
    };

    node.updateImageLayout = updateLayout;
    node._comfy1hewEnsurePreviewLayout = ensurePreviewLayout;

    return { autoSizeToContent, requestAutoSize, updateLayout, ensurePreviewLayout };
}

export function installVideoPreviewLayout({
    app,
    node,
    videoWidget,
    container,
    videoEl,
}) {
    const getMaxPreviewHeight = () => {
        const maxPreviewHeight =
            typeof videoWidget._comfy1hew_maxPreviewHeight === "number"
                ? videoWidget._comfy1hew_maxPreviewHeight
                : null;
        if (node._comfy1hewVideoPreviewUserResized) {
            return null;
        }
        return maxPreviewHeight;
    };

    const autoSizeToContent = ({ allowShrink = false } = {}) => {
        if (!videoWidget.aspectRatio) {
            return;
        }

        const width = node.size[0];
        let desiredWidgetHeight = width * videoWidget.aspectRatio + 20;
        const maxPreviewHeight = getMaxPreviewHeight();
        if (maxPreviewHeight && isFinite(maxPreviewHeight)) {
            desiredWidgetHeight = Math.min(desiredWidgetHeight, maxPreviewHeight);
        }

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
                desiredHeight = estimatePreviewTopOffset(node, videoWidget) + desiredWidgetHeight;
            }
        }

        if (
            (allowShrink && Math.abs(node.size[1] - desiredHeight) > 1)
            || (!allowShrink && node.size[1] + 1 < desiredHeight)
        ) {
            node._comfy1hewInternalVideoPreviewResize = true;
            try {
                node.setSize([node.size[0], desiredHeight]);
            } finally {
                node._comfy1hewInternalVideoPreviewResize = false;
            }
        }
    };

    const requestAutoSize = ({ allowShrink = false, force = false } = {}) => {
        const fallbackWidth = Number(node?._comfy1hewVideoInfo?.width) || 0;
        const fallbackHeight = Number(node?._comfy1hewVideoInfo?.height) || 0;
        const width = videoEl.videoWidth || fallbackWidth;
        const height = videoEl.videoHeight || fallbackHeight;
        if (!width || !height) {
            return;
        }
        const key = `${width}x${height}`;
        if (!force && node._comfy1hewVideoAutoSizeKey === key) {
            return;
        }
        node._comfy1hewVideoAutoSizeKey = key;
        setTimeout(() => autoSizeToContent({ allowShrink }), 0);
    };

    node._comfy1hewLastPreviewVideoNodeWidth ??= node.size?.[0];

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

        const maxPreviewHeight = getMaxPreviewHeight();
        if (maxPreviewHeight && isFinite(maxPreviewHeight)) {
            availableHeight = Math.min(availableHeight, maxPreviewHeight);
        }

        if (availableHeight < 0) {
            availableHeight = 0;
        }
        container.style.height = `${availableHeight}px`;

        app.graph.setDirtyCanvas(true, true);
    };

    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        const r2 = originalOnResize
            ? originalOnResize.apply(this, arguments)
            : undefined;
        try {
            const nextWidth = Array.isArray(size) ? size[0] : node.size?.[0];
            const prevWidth = node._comfy1hewLastPreviewVideoNodeWidth;
            node._comfy1hewLastPreviewVideoNodeWidth = nextWidth;

            if (
                !node._comfy1hewInternalVideoPreviewResize
                && Number.isFinite(nextWidth)
                && Number.isFinite(prevWidth)
                && Math.abs(nextWidth - prevWidth) > 1
            ) {
                node._comfy1hewVideoPreviewUserResized = true;
            }

            updateLayout();
        } catch {}
        return r2;
    };

    const ensurePreviewLayout = ({
        allowShrink = false,
        forceAutoSize = false,
    } = {}) => {
        if (container.dataset.comfy1hewForceHidden === "1") {
            updateLayout();
            return;
        }

        if (videoEl.videoWidth && videoEl.videoHeight) {
            videoWidget.aspectRatio = videoEl.videoHeight / videoEl.videoWidth;
        } else {
            const fallbackWidth = Number(node?._comfy1hewVideoInfo?.width) || 0;
            const fallbackHeight = Number(node?._comfy1hewVideoInfo?.height) || 0;
            if (fallbackWidth > 0 && fallbackHeight > 0) {
                videoWidget.aspectRatio = fallbackHeight / fallbackWidth;
            }
        }

        requestAutoSize({ allowShrink, force: forceAutoSize });
        updateLayout();
    };

    node.updateVideoLayout = updateLayout;
    node._comfy1hewEnsurePreviewLayout = ensurePreviewLayout;

    return { autoSizeToContent, requestAutoSize, updateLayout, ensurePreviewLayout };
}

