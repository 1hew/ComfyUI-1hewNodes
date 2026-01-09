import { app } from "../../../scripts/app.js";

function getTextValue(message) {
    if (!message) {
        return null;
    }
    const raw = message.text;
    if (raw == null) {
        return null;
    }
    if (Array.isArray(raw)) {
        return raw.length ? String(raw[0]) : null;
    }
    return String(raw);
}

function applyBaseOutputLabels(node, baseLabels) {
    if (!node?.outputs || !Array.isArray(baseLabels)) {
        return;
    }
    for (let i = 0; i < baseLabels.length; i += 1) {
        if (!node.outputs[i]) {
            continue;
        }
        node.outputs[i].label = baseLabels[i];
    }
}

function applyValueOutputLabels(node, values, slots) {
    if (!node?.outputs || !Array.isArray(slots)) {
        return;
    }
    for (const slot of slots) {
        const outIdx = slot?.out;
        const valueIdx = slot?.value;
        const suffix = slot?.suffix;
        if (
            typeof outIdx !== "number"
            || typeof valueIdx !== "number"
            || typeof suffix !== "string"
        ) {
            continue;
        }
        if (!node.outputs[outIdx]) {
            continue;
        }
        const v = values[valueIdx];
        if (typeof v !== "number" || Number.isNaN(v)) {
            continue;
        }
        node.outputs[outIdx].label = `${v} ${suffix}`;
    }
}

function parseSplitNumbers(text, delimiter) {
    if (typeof text !== "string" || !text.length) {
        return null;
    }
    const parts = text.split(delimiter).map((p) => Number(p));
    if (!parts.length) {
        return null;
    }
    for (const n of parts) {
        if (typeof n !== "number" || Number.isNaN(n)) {
            return null;
        }
    }
    return parts;
}

function parseSingleNumber(text) {
    if (typeof text !== "string" || !text.length) {
        return null;
    }
    const n = Number(text);
    if (typeof n !== "number" || Number.isNaN(n)) {
        return null;
    }
    return [n];
}

const CONFIGS = {
    "1hew_IntImageSize": {
        base: ["width", "height"],
        delimiter: "x",
        slots: [
            { out: 0, value: 0, suffix: "width" },
            { out: 1, value: 1, suffix: "height" },
        ],
    },
    "1hew_IntImageSideLength": {
        base: ["int"],
        delimiter: null,
        slots: [{ out: 0, value: 0, suffix: "int" }],
    },
    "1hew_IntMaskSideLength": {
        base: ["int"],
        delimiter: null,
        slots: [{ out: 0, value: 0, suffix: "int" }],
    },
    "1hew_GetFileCount": {
        base: ["count", "folder", "include_subdir"],
        delimiter: null,
        slots: [{ out: 0, value: 0, suffix: "count" }],
    },
    "1hew_ListCustomFloat": {
        base: ["float_list", "count"],
        delimiter: null,
        slots: [{ out: 1, value: 0, suffix: "count" }],
    },
    "1hew_ListCustomInt": {
        base: ["int_list", "count"],
        delimiter: null,
        slots: [{ out: 1, value: 0, suffix: "count" }],
    },
    "1hew_ListCustomSeed": {
        base: ["seed_list", "count"],
        delimiter: null,
        slots: [{ out: 1, value: 0, suffix: "count" }],
    },
    "1hew_ListCustomString": {
        base: ["string_list", "count"],
        delimiter: null,
        slots: [{ out: 1, value: 0, suffix: "count" }],
    },
};

app.registerExtension({
    name: "ComfyUI-1hewNodesV3.UiOutputLabels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const typeName = nodeData?.name || nodeType?.type || nodeType?.title;
        const cfg = CONFIGS[typeName];
        if (!cfg) {
            return;
        }

        const originalOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function onAddedPatched() {
            const r = originalOnAdded?.apply(this, arguments);
            applyBaseOutputLabels(this, cfg.base);
            return r;
        };

        const originalOnConnectInput = nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function onConnectInputPatched() {
            const r = originalOnConnectInput?.apply(this, arguments);
            applyBaseOutputLabels(this, cfg.base);
            return r;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        const originalOnAfterExecuteNode = nodeType.prototype.onAfterExecuteNode;

        function updateFromMessage(node, message) {
            const text = getTextValue(message);
            if (!text) {
                return;
            }
            if (node._last_ui_output_text === text) {
                return;
            }
            node._last_ui_output_text = text;
            const values = cfg.delimiter
                ? parseSplitNumbers(text, cfg.delimiter)
                : parseSingleNumber(text);
            if (!values) {
                return;
            }
            applyValueOutputLabels(node, values, cfg.slots);
        }

        nodeType.prototype.onExecuted = function onExecutedPatched(message) {
            const r = originalOnExecuted?.apply(this, arguments);
            updateFromMessage(this, message);
            return r;
        };

        nodeType.prototype.onAfterExecuteNode = function onAfterExecuteNodePatched(
            message,
        ) {
            const r = originalOnAfterExecuteNode?.apply(this, arguments);
            updateFromMessage(this, message);
            return r;
        };
    },
});

