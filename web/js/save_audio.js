import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

const SAVE_AUDIO_NODE = "1hew_SaveAudio";
const AUDIO_WIDGET_NAME = "audio_preview";
// Do not use the built-in "audioUI" type here.  Nodes 2.0 maps that type to its
// own Vue AudioPreviewPlayer, which ignores a custom <audio> element and reads
// the widget value instead; the result is an empty/zero-width player.  A custom
// type keeps this DOM element mounted in both the legacy canvas and Nodes 2.0.
const AUDIO_WIDGET_TYPE = "1hewAudioPreview";
const PLAYER_HEIGHT = 58;
const PLAYER_MIN_WIDTH = 180;

function buildAudioUrl(entry) {
    if (!entry || !entry.filename) {
        return "";
    }
    const params = new URLSearchParams({
        filename: String(entry.filename),
        subfolder: String(entry.subfolder ?? ""),
        type: String(entry.type ?? "output"),
        rand: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function extractAudioEntries(message) {
    // null = payload has no "audio" key, so an unrelated executed message must
    // not clear a player from a previous run.  [] = explicit empty result.
    if (!message || typeof message !== "object") {
        return null;
    }
    if (Object.prototype.hasOwnProperty.call(message, "audio")) {
        const value = message.audio;
        if (Array.isArray(value)) {
            return value;
        }
        return value ? [value] : [];
    }
    for (const key of ["output", "result", "ui"]) {
        const nested = message[key];
        if (nested && typeof nested === "object") {
            const found = extractAudioEntries(nested);
            if (found !== null) {
                return found;
            }
        }
    }
    return null;
}

function firstAudioEntry(entries) {
    if (!Array.isArray(entries)) {
        return null;
    }
    return entries.find(
        (entry) =>
            entry
            && typeof entry === "object"
            && typeof entry.filename === "string"
            && entry.filename.length > 0,
    ) ?? null;
}

function syncWidgetWidth(node) {
    const widget = node?._comfy1hewAudioWidget;
    const width = Number(node?.size?.[0]);
    if (!widget || !Number.isFinite(width) || width <= 0) {
        return;
    }
    // Older frontends size DOM widgets from widget.width.  The modern overlay
    // uses node.size, so keeping both in sync is safe.
    widget.width = width;
}

function resizeNodeForPlayer(node, hasAudio) {
    // Nodes 2.0 measures the DOM and resizes the node itself; calling setSize()
    // there fights the ResizeObserver and can cause a grow/shrink loop.
    if (typeof LiteGraph !== "undefined" && LiteGraph.vueNodesMode) {
        return;
    }
    try {
        const width = node?.size?.[0] ?? 240;
        const currentHeight = node?.size?.[1] ?? 0;
        const computed = node?.computeSize?.([width, currentHeight]);
        const nextHeight = Array.isArray(computed) ? computed[1] : null;
        if (!Number.isFinite(nextHeight)) {
            return;
        }
        // Grow to fit the player when audio exists; only shrink back when the
        // node was still at its computed height (the user did not resize it).
        if (hasAudio && nextHeight > currentHeight + 0.5) {
            node.setSize([width, nextHeight]);
        } else if (!hasAudio && currentHeight <= nextHeight + 0.5) {
            node.setSize([width, nextHeight]);
        }
    } catch {}
}

function setPlayerSource(node, url) {
    const audioEl = node?._comfy1hewAudioEl;
    const container = node?._comfy1hewAudioContainer;
    if (!audioEl || !container) {
        return;
    }

    const hasAudio = Boolean(url);
    node._comfy1hewAudioUrl = hasAudio ? url : "";

    if (hasAudio) {
        if (audioEl.getAttribute("src") !== url) {
            audioEl.src = url;
        }
        container.style.display = "flex";
        audioEl.style.display = "block";
    } else {
        try {
            audioEl.pause();
        } catch {}
        audioEl.removeAttribute("src");
        try {
            audioEl.load();
        } catch {}
        container.style.display = "none";
        audioEl.style.display = "none";
    }

    const widget = node._comfy1hewAudioWidget;
    if (widget) {
        widget.computeSize = (width) => [width, hasAudio ? PLAYER_HEIGHT : 0];
        widget.value = node._comfy1hewAudioUrl;
    }

    syncWidgetWidth(node);
    resizeNodeForPlayer(node, hasAudio);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function ensureAudioPlayer(node) {
    if (!node) {
        return null;
    }
    if (node._comfy1hewAudioEl) {
        return node._comfy1hewAudioEl;
    }
    if (typeof node.addDOMWidget !== "function") {
        return null;
    }

    const container = document.createElement("div");
    container.className = "comfy-1hew-save-audio-container";
    Object.assign(container.style, {
        display: "none",
        alignItems: "center",
        width: "100%",
        minWidth: `${PLAYER_MIN_WIDTH}px`,
        height: `${PLAYER_HEIGHT}px`,
        padding: "0 10px",
        boxSizing: "border-box",
    });

    const audioEl = document.createElement("audio");
    audioEl.controls = true;
    audioEl.preload = "metadata";
    audioEl.className = "comfy-1hew-save-audio";
    Object.assign(audioEl.style, {
        display: "block",
        width: "100%",
        minWidth: "0",
        height: `${PLAYER_HEIGHT}px`,
        margin: "0",
    });
    container.appendChild(audioEl);

    const widget = node.addDOMWidget(
        AUDIO_WIDGET_NAME,
        AUDIO_WIDGET_TYPE,
        container,
        { serialize: false, hideOnZoom: false, margin: 0 },
    );
    if (!widget) {
        return null;
    }

    widget.serialize = false;
    widget.options = widget.options || {};
    widget.options.serialize = false;
    // Keep a URL value on the widget as well.  It is what the built-in Vue
    // audio component would read, and it also makes the DOM widget's value
    // observable by other extensions.
    widget.options.getValue = () => node._comfy1hewAudioUrl || "";
    widget.options.setValue = (value) => {
        node._comfy1hewAudioUrl = typeof value === "string" ? value : "";
    };
    widget.computeSize = (width) => [width, 0];
    widget.width = Number(node.size?.[0]) || 0;

    node._comfy1hewAudioEl = audioEl;
    node._comfy1hewAudioContainer = container;
    node._comfy1hewAudioWidget = widget;
    node._comfy1hewAudioUrl = "";
    syncWidgetWidth(node);
    return audioEl;
}

app.registerExtension({
    name: "ComfyUI-1hewNodes.SaveAudio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== SAVE_AUDIO_NODE) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            ensureAudioPlayer(this);
            syncWidgetWidth(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            ensureAudioPlayer(this);
            syncWidgetWidth(this);
            return result;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = onResize?.apply(this, arguments);
            syncWidgetWidth(this);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            ensureAudioPlayer(this);
            const entries = extractAudioEntries(message);
            if (entries !== null) {
                const entry = firstAudioEntry(entries);
                setPlayerSource(this, entry ? buildAudioUrl(entry) : "");
            }
            return result;
        };
    },
});
