const PREVIEW_HIDDEN_PROPERTY = "comfy1hew_preview_hidden";
const FORCE_HIDDEN_DATASET_KEY = "comfy1hewForceHidden";
const PREV_DISPLAY_DATASET_KEY = "comfy1hewPrevDisplay";
const PREV_COMPUTE_SIZE_STATE_KEY = "_comfy1hewPrevComputeSizeState";

function safePlay(videoEl, respectFrameAccurate) {
    if (!videoEl || videoEl.dataset.comfy1hewUserPaused === "1") {
        return;
    }
    if (respectFrameAccurate && videoEl.dataset.comfy1hewFrameAccurate === "1") {
        return;
    }
    const p = videoEl.play();
    if (p && typeof p.catch === "function") {
        p.catch(() => {});
    }
}

function getTargetNodes(app, currentNode) {
    const canvas = app?.canvas;
    const selectedNodes = canvas?.selected_nodes || {};
    const targetNodes = Object.values(selectedNodes);
    if (targetNodes.length > 0) {
        return targetNodes;
    }
    return [currentNode];
}

function listPreviewWidgets(node) {
    const items = [];
    if (!node?.widgets) {
        return items;
    }
    for (const w of node.widgets) {
        const element = w?.element;
        if (!element) {
            continue;
        }
        const hasMedia =
            element.tagName === "VIDEO" ||
            element.tagName === "IMG" ||
            element.querySelector("video") ||
            element.querySelector("img");
        if (!hasMedia) {
            continue;
        }

        const videos =
            element.tagName === "VIDEO"
                ? [element]
                : Array.from(element.querySelectorAll("video"));
        items.push({ widget: w, element, videos });
    }
    return items;
}

function fitNodeHeight(node) {
    try {
        const computed = node?.computeSize?.([node.size?.[0], node.size?.[1]]);
        if (Array.isArray(computed) && computed.length >= 2) {
            node.setSize([node.size[0], computed[1]]);
        }
        node?.graph?.setDirtyCanvas?.(true, true);
    } catch (e) {}
}

function isPreviewHidden(node) {
    return node?.properties?.[PREVIEW_HIDDEN_PROPERTY] === 1;
}

function setPreviewHidden(
    node,
    hidden,
    { respectFrameAccurateOnShow = false } = {},
) {
    if (!node) {
        return;
    }
    node.properties ??= {};
    node.properties[PREVIEW_HIDDEN_PROPERTY] = hidden ? 1 : 0;

    const items = listPreviewWidgets(node);
    for (const { widget, element, videos } of items) {
        element.dataset[FORCE_HIDDEN_DATASET_KEY] = hidden ? "1" : "0";

        if (hidden) {
            if (!widget[PREV_COMPUTE_SIZE_STATE_KEY]) {
                widget[PREV_COMPUTE_SIZE_STATE_KEY] = {
                    hadOwn: Object.prototype.hasOwnProperty.call(
                        widget,
                        "computeSize",
                    ),
                    value: widget.computeSize,
                };
            }
            widget.computeSize = (width) => [width, -4];
        } else if (widget[PREV_COMPUTE_SIZE_STATE_KEY]) {
            const state = widget[PREV_COMPUTE_SIZE_STATE_KEY];
            if (state.hadOwn) {
                widget.computeSize = state.value;
            } else {
                delete widget.computeSize;
            }
            delete widget[PREV_COMPUTE_SIZE_STATE_KEY];
        }

        if (hidden) {
            element.dataset[PREV_DISPLAY_DATASET_KEY] ??= element.style.display ?? "";
            element.style.display = "none";
        } else {
            element.style.display = element.dataset[PREV_DISPLAY_DATASET_KEY] ?? "";
            delete element.dataset[PREV_DISPLAY_DATASET_KEY];
        }

        for (const v of videos) {
            if (hidden) {
                v.pause();
            } else {
                safePlay(v, respectFrameAccurateOnShow);
            }
        }
    }

    try {
        node?.updateVideoLayout?.();
    } catch (e) {}
    fitNodeHeight(node);
}

function syncPreview(nodes) {
    for (const node of nodes) {
        for (const { videos } of listPreviewWidgets(node)) {
            for (const v of videos) {
                v.currentTime = 0;
                v.play();
                if (v.dataset.comfy1hewUserPaused) {
                    v.dataset.comfy1hewUserPaused = "0";
                }
                v.muted = true;
            }
        }
    }
}

function hasAnyVideo(nodes) {
    for (const node of nodes) {
        for (const { videos } of listPreviewWidgets(node)) {
            if (Array.isArray(videos) && videos.length > 0) {
                return true;
            }
        }
    }
    return false;
}

function isMuted(node) {
    for (const { videos } of listPreviewWidgets(node)) {
        const v = videos?.[0];
        if (v && v.dataset.comfy1hewForceMute === "1") {
            return true;
        }
    }
    return false;
}

function setMuted(nodes, forcedMute) {
    for (const node of nodes) {
        for (const { videos } of listPreviewWidgets(node)) {
            for (const v of videos) {
                v.dataset.comfy1hewForceMute = forcedMute ? "1" : "0";
                v.muted = !!forcedMute;
                if (!forcedMute) {
                    v.volume = 1.0;
                }
            }
        }
    }
}

export function applyPreviewHiddenState(
    node,
    { respectFrameAccurateOnShow = false } = {},
) {
    setPreviewHidden(node, isPreviewHidden(node), { respectFrameAccurateOnShow });
}

export function addPreviewMenuOptions(
    options,
    { app, currentNode, respectFrameAccurateOnShow = false } = {},
) {
    if (!options || !app || !currentNode) {
        return;
    }
    const targetNodes = getTargetNodes(app, currentNode);
    const firstNode = targetNodes[0];
    const anyVideo = hasAnyVideo(targetNodes);

    if (anyVideo) {
        options.push({
            content: "Sync Preview",
            callback: () => syncPreview(targetNodes),
        });
    }

    const hiddenDesc = isPreviewHidden(firstNode) ? "Show Preview" : "Hide Preview";
    options.push({
        content: hiddenDesc,
        callback: () => {
            const nextHidden = !isPreviewHidden(firstNode);
            for (const node of targetNodes) {
                setPreviewHidden(node, nextHidden, { respectFrameAccurateOnShow });
            }
        },
    });

    if (!anyVideo) {
        return;
    }

    if (isMuted(firstNode)) {
        options.push({
            content: "Unmute Preview",
            callback: () => setMuted(targetNodes, false),
        });
    } else {
        options.push({
            content: "Mute Preview",
            callback: () => setMuted(targetNodes, true),
        });
    }
}
