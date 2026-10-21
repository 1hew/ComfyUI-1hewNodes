import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

const SAVE_AUDIO_NODE = "1hew_SaveAudio";
const AUDIO_WIDGET_NAME = "audio_preview";
const PLAYER_HEIGHT = 56;

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

function resizeNodeForPlayer(node, hasAudio) {
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
    if (!audioEl) {
        return;
    }

    const hasAudio = Boolean(url);
    if (hasAudio) {
        audioEl.src = url;
    } else {
        try {
            audioEl.pause();
        } catch {}
        audioEl.removeAttribute("src");
    }
    audioEl.style.display = hasAudio ? "block" : "none";
    try {
        audioEl.load();
    } catch {}

    const widget = node._comfy1hewAudioWidget;
    if (widget) {
        widget.computeSize = (width) => [width, hasAudio ? PLAYER_HEIGHT : 0];
    }

    resizeNodeForPlayer(node, hasAudio);
    node.graph?.setDirtyCanvas?.(true, true);
}

function ensureAudioPlayer(node) {
    if (!node || node._comfy1hewAudioEl) {
        return node?._comfy1hewAudioEl ?? null;
    }
    if (typeof node.addDOMWidget !== "function") {
        return null;
    }

    const audioEl = document.createElement("audio");
    audioEl.controls = true;
    audioEl.preload = "metadata";
    audioEl.className = "comfy-1hew-save-audio";
    audioEl.style.width = "100%";
    audioEl.style.margin = "4px 0";
    audioEl.style.display = "none";

    const widget = node.addDOMWidget(
        AUDIO_WIDGET_NAME,
        "audioUI",
        audioEl,
        { serialize: false, hideOnZoom: false },
    );
    if (widget) {
        widget.serialize = false;
        widget.options = widget.options || {};
        widget.options.serialize = false;
        widget.computeSize = (width) => [width, 0];
    }

    node._comfy1hewAudioEl = audioEl;
    node._comfy1hewAudioWidget = widget;
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
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            ensureAudioPlayer(this);
            const audios = Array.isArray(message?.audio) ? message.audio : [];
            const first = audios.find((entry) => entry && entry.filename);
            setPlayerSource(this, buildAudioUrl(first));
            return result;
        };
    },
});
