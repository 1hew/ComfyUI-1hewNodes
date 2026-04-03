import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const TEXT_PREVIEW_NODES = ["1hew_SaveTxt", "1hew_LoadTxt"];

function getSerializableWidgets(node) {
    return (node?.widgets || []).filter((w) => w && w.serialize !== false);
}

function getWidgetDefaultValue(widget) {
    const optionDefault = widget?.options?.default;
    if (optionDefault !== undefined) {
        return optionDefault;
    }
    if (widget?.type === "toggle") {
        return false;
    }
    if (widget?.type === "combo") {
        const values = widget?.options?.values;
        if (Array.isArray(values) && values.length > 0) {
            return values[0];
        }
    }
    return "";
}

function isValueCompatible(widget, value) {
    if (widget?.type === "toggle") {
        return typeof value === "boolean";
    }
    if (widget?.type === "combo") {
        return typeof value === "string";
    }
    if (widget?.type === "text" || widget?.type === "customtext") {
        return typeof value === "string";
    }
    return value !== undefined;
}

function scoreWindow(widgets, values, start) {
    let score = 0;
    for (let i = 0; i < widgets.length; i += 1) {
        if (isValueCompatible(widgets[i], values[start + i])) {
            score += 1;
        }
    }
    return score;
}

function normalizeWidgetsValues(node, rawValues) {
    const widgets = getSerializableWidgets(node);
    const expectedCount = widgets.length;
    const sourceValues = Array.isArray(rawValues) ? rawValues.slice() : [];
    if (!expectedCount) {
        return sourceValues;
    }

    let candidate = sourceValues.slice(0, expectedCount);
    if (sourceValues.length >= expectedCount) {
        let bestStart = 0;
        let bestScore = -1;
        for (let start = 0; start <= sourceValues.length - expectedCount; start += 1) {
            const score = scoreWindow(widgets, sourceValues, start);
            if (score > bestScore) {
                bestScore = score;
                bestStart = start;
            }
        }
        candidate = sourceValues.slice(bestStart, bestStart + expectedCount);
    }

    while (candidate.length < expectedCount) {
        candidate.push(undefined);
    }

    return widgets.map((widget, index) => {
        const value = candidate[index];
        return isValueCompatible(widget, value)
            ? value
            : getWidgetDefaultValue(widget);
    });
}

function configurePreviewWidget(widget) {
    if (!widget) {
        return widget;
    }
    widget.serialize = false;
    widget.options = widget.options || {};
    widget.options.read_only = true;
    widget.options.hidden = false;
    if (widget.element) {
        widget.element.readOnly = true;
    }
    return widget;
}

function movePreviewWidgetToTop(node, widget) {
    if (!node?.widgets || !widget) {
        return widget;
    }
    const idx = node.widgets.indexOf(widget);
    if (idx > 0) {
        node.widgets.splice(idx, 1);
        node.widgets.splice(0, 0, widget);
    }
    return widget;
}

function ensurePreviewWidget(node) {
    if (!node) {
        return null;
    }
    let previewWidget = node.widgets?.find((w) => w.name === "preview_text") || null;
    if (!previewWidget) {
        previewWidget = ComfyWidgets["STRING"](
            node,
            "preview_text",
            ["STRING", { multiline: true }],
            app,
        ).widget;
    }
    configurePreviewWidget(previewWidget);
    movePreviewWidgetToTop(node, previewWidget);
    return previewWidget;
}

app.registerExtension({
    name: "ComfyUI-1hewNodesV3.TextPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const typeName = nodeData?.name || nodeType?.type || nodeType?.title;
        if (!TEXT_PREVIEW_NODES.includes(typeName)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const previewWidget = ensurePreviewWidget(this);
            if (previewWidget) {
                const savedPreviewText =
                    typeof this.properties?._comfy1hewPreviewText === "string"
                        ? this.properties._comfy1hewPreviewText
                        : "";
                if (savedPreviewText && !previewWidget.value) {
                    previewWidget.value = savedPreviewText;
                }
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const data = arguments[0];
            const previewWidget = this.widgets?.find((w) => w.name === "preview_text");
            const previewValue = previewWidget?.value ?? "";
            const normalizedIncomingValues = Array.isArray(data?.widgets_values)
                ? normalizeWidgetsValues(this, data.widgets_values)
                : null;
            const savedPreviewText =
                (typeof data?.properties?._comfy1hewPreviewText === "string"
                    ? data.properties._comfy1hewPreviewText
                    : null)
                ?? (typeof this.properties?._comfy1hewPreviewText === "string"
                    ? this.properties._comfy1hewPreviewText
                    : "");

            if (data && normalizedIncomingValues) {
                data.widgets_values = normalizedIncomingValues;
            }

            const r = onConfigure?.apply(this, arguments);
            const restoredPreviewWidget = ensurePreviewWidget(this);
            if (restoredPreviewWidget) {
                const nextPreviewValue = previewValue || savedPreviewText;
                if (nextPreviewValue && !restoredPreviewWidget.value) {
                    restoredPreviewWidget.value = nextPreviewValue;
                }
            }
            return r;
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function () {
            const r = onSerialize?.apply(this, arguments);
            const serialized = arguments[0];
            const normalizedSerializedValues = normalizeWidgetsValues(
                this,
                Array.isArray(serialized?.widgets_values)
                    ? serialized.widgets_values
                    : [],
            );
            const previewWidget = this.widgets?.find((w) => w?.name === "preview_text");
            const previewTextToPersist =
                (typeof previewWidget?.value === "string" ? previewWidget.value : "")
                || (typeof this.properties?._comfy1hewPreviewText === "string"
                    ? this.properties._comfy1hewPreviewText
                    : "");
            if (serialized) {
                serialized.widgets_values = normalizedSerializedValues;
                serialized.properties = {
                    ...(serialized.properties || {}),
                    _comfy1hewPreviewText: previewTextToPersist,
                };
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const w = ensurePreviewWidget(this);
            if (!w) {
                return;
            }
            const raw = message?.text;
            if (raw == null) {
                return;
            }
            w.value = Array.isArray(raw) ? raw.join("\n\n") : String(raw);
            this.properties = this.properties || {};
            this.properties._comfy1hewPreviewText = w.value;
        };
    },
});
