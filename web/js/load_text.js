import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { setTextPreview } from "./text_preview.js";

const LOAD_TEXT_NODE = "1hew_LoadTxt";
const UPLOAD_SUBFOLDER = "text";

const previewTimers = new WeakMap();

async function refreshTextPreview(node) {
    if (!node || node.type !== LOAD_TEXT_NODE) {
        return;
    }
    const fileWidget = node.widgets?.find((item) => item.name === "file");
    const indexWidget = node.widgets?.find((item) => item.name === "index");
    const includeWidget = node.widgets?.find((item) => item.name === "include_subdir");
    const file = String(fileWidget?.value ?? "").trim();
    if (!file) {
        setTextPreview(node, "", { resize: true });
        return;
    }

    const requestId = (node._comfy1hewTextPreviewReq || 0) + 1;
    node._comfy1hewTextPreviewReq = requestId;
    const params = new URLSearchParams({
        file,
        index: String(indexWidget?.value ?? 0),
        include_subdir: includeWidget?.value ? "true" : "false",
    });

    try {
        const response = await api.fetchApi(
            `/1hew/read_text_preview?${params.toString()}`,
        );
        if (requestId !== node._comfy1hewTextPreviewReq) {
            return;
        }
        if (!response.ok) {
            setTextPreview(node, "", { resize: true });
            return;
        }
        const data = await response.json();
        if (requestId !== node._comfy1hewTextPreviewReq) {
            return;
        }
        setTextPreview(node, data?.text ?? "", { resize: true });
    } catch (error) {
        if (requestId !== node._comfy1hewTextPreviewReq) {
            return;
        }
        console.warn("[1hewNodes] Failed to preview text file:", error);
        setTextPreview(node, "", { resize: true });
    }
}

function scheduleTextPreview(node, delay = 120) {
    if (!node || node.type !== LOAD_TEXT_NODE) {
        return;
    }
    const previous = previewTimers.get(node);
    if (previous) {
        clearTimeout(previous);
    }
    const timer = setTimeout(() => {
        previewTimers.delete(node);
        void refreshTextPreview(node);
    }, delay);
    previewTimers.set(node, timer);
}

function bindTextPreview(node) {
    if (!node || node._comfy1hewTextPreviewBound) {
        return;
    }
    node._comfy1hewTextPreviewBound = true;

    for (const name of ["file", "index", "include_subdir"]) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget) {
            continue;
        }
        const originalCallback = widget.callback;
        widget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            scheduleTextPreview(node, 120);
            return result;
        };
    }

    scheduleTextPreview(node, 0);
}

function setFileWidget(node, value) {
    const widget = node?.widgets?.find((item) => item.name === "file");
    if (!widget) {
        return false;
    }
    widget.value = value;
    widget.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
}

async function uploadTextFile(node, file) {
    if (!/\.txt$/i.test(file.name)) {
        window.alert("Please choose a .txt file.");
        return;
    }

    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    body.append("subfolder", UPLOAD_SUBFOLDER);
    body.append("overwrite", "false");

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const name = String(data?.name || file.name);
    const subfolder = String(data?.subfolder || "").replace(/\\/g, "/").replace(/\/+$/, "");
    const value = subfolder ? `${subfolder}/${name}` : name;
    if (!setFileWidget(node, value)) {
        throw new Error("file widget not found");
    }
}

function pickTextFile(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.id = "comfy-1hew-load-text-upload";
    input.accept = ".txt,text/plain";
    input.style.display = "none";
    document.body.appendChild(input);

    const cleanup = () => {
        input.remove();
    };
    input.addEventListener("cancel", cleanup);
    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        cleanup();
        if (!file) {
            return;
        }
        try {
            await uploadTextFile(node, file);
        } catch (error) {
            console.error("[1hewNodes] Failed to upload text file:", error);
            window.alert(`Upload failed: ${error}`);
        }
    });
    input.click();
}

function installUploadButton(node) {
    if (!node || node._comfy1hewLoadTextUpload) {
        return;
    }
    node._comfy1hewLoadTextUpload = true;

    const button = node.addWidget(
        "button",
        "choose file to upload",
        null,
        () => pickTextFile(node),
    );
    if (!button) {
        return;
    }
    button.serialize = false;
    button.options = button.options || {};
    button.options.serialize = false;
    button.tooltip = "Choose a .txt file to upload to ComfyUI input/text";

    // Keep the upload action next to the file/index controls, before the
    // growable text preview so the preview continues to occupy the bottom.
    const includeIndex = node.widgets.findIndex((item) => item.name === "include_subdir");
    const currentIndex = node.widgets.indexOf(button);
    if (includeIndex >= 0 && currentIndex > includeIndex + 1) {
        node.widgets.splice(currentIndex, 1);
        node.widgets.splice(includeIndex + 1, 0, button);
    }

    const computed = node.computeSize?.();
    if (Array.isArray(computed) && Number.isFinite(computed[1])) {
        const width = Number(node.size?.[0]) || 240;
        const height = Number(node.size?.[1]) || 0;
        if (computed[1] > height) {
            node.setSize?.([width, computed[1]]);
        }
    }
}

app.registerExtension({
    name: "ComfyUI-1hewNodes.LoadTextUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== LOAD_TEXT_NODE) {
            return;
        }
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installUploadButton(this);
            bindTextPreview(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            scheduleTextPreview(this, 50);
            return result;
        };
    },
});
