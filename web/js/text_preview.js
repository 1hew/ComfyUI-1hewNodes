import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const TEXT_PREVIEW_NODES = ["1hew_SaveTxt", "1hew_LoadTxt"];

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

            const previewWidget = ComfyWidgets["STRING"](
                this,
                "preview_text",
                ["STRING", { multiline: true }],
                app,
            ).widget;

            previewWidget.serialize = false;
            previewWidget.options = previewWidget.options || {};
            previewWidget.options.read_only = true;
            previewWidget.options.hidden = false;
            if (previewWidget.element) {
                previewWidget.element.readOnly = true;
            }

            if (this.widgets) {
                const idx = this.widgets.indexOf(previewWidget);
                if (idx > 0) {
                    this.widgets.splice(idx, 1);
                    this.widgets.splice(0, 0, previewWidget);
                }
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const w = this.widgets?.find((w) => w.name === "preview_text");
            if (!w) {
                return;
            }
            const raw = message?.text;
            if (raw == null) {
                return;
            }
            w.value = Array.isArray(raw) ? raw.join("\n\n") : String(raw);
        };
    },
});
