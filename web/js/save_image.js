import { app } from "../../../scripts/app.js";
import { chainOnRemoved, registerExtensionOnce } from "./core/runtime.js";

const SAVE_IMAGE_NODE_NAMES = new Set(["1hew_SaveImage"]);
const PREVIEW_PROPERTY_KEY = "_comfy1hewSavedImagePreview";

function normalizePreviewEntry(entry) {
    if (!entry || typeof entry !== "object") {
        return null;
    }

    const filename = typeof entry.filename === "string" ? entry.filename.trim() : "";
    if (!filename) {
        return null;
    }

    const subfolder = typeof entry.subfolder === "string" ? entry.subfolder.trim() : "";
    const rawType = typeof entry.type === "string" ? entry.type.trim().toLowerCase() : "";
    const type = rawType === "temp" ? "temp" : "output";

    return { filename, subfolder, type };
}

function normalizePreviewEntries(value) {
    if (Array.isArray(value)) {
        return value.map(normalizePreviewEntry).filter(Boolean);
    }

    const entry = normalizePreviewEntry(value);
    return entry ? [entry] : [];
}

function extractPreviewEntries(message) {
    const visit = (value, depth) => {
        if (depth <= 0 || value == null) {
            return [];
        }

        if (Array.isArray(value)) {
            const directEntries = normalizePreviewEntries(value);
            if (directEntries.length) {
                return directEntries;
            }

            for (const item of value) {
                const result = visit(item, depth - 1);
                if (result.length) {
                    return result;
                }
            }
            return [];
        }

        if (typeof value !== "object") {
            return [];
        }

        const imageEntries = normalizePreviewEntries(value.images);
        if (imageEntries.length) {
            return imageEntries;
        }

        const directEntry = normalizePreviewEntry(value);
        if (directEntry) {
            return [directEntry];
        }

        for (const key of Object.keys(value)) {
            const result = visit(value[key], depth - 1);
            if (result.length) {
                return result;
            }
        }
        return [];
    };

    return visit(message, 6);
}

function buildPreviewUrl(entry) {
    const params = new URLSearchParams({
        filename: entry.filename,
        type: entry.type || "output",
        t: String(Date.now()),
    });
    if (entry.subfolder) {
        params.set("subfolder", entry.subfolder);
    }
    return `/view?${params.toString()}`;
}

function previewKey(entry) {
    return entry ? `${entry.type}::${entry.subfolder}::${entry.filename}` : "";
}

function previewEntriesKey(entries) {
    return entries.map(previewKey).join("||");
}

function loadPreviewImage(entry) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.decoding = "async";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = buildPreviewUrl(entry);
    });
}

function clearPreview(node, { clearPersisted = false } = {}) {
    if (!node) {
        return;
    }

    node._comfy1hewSaveImagePreviewKey = "";
    node._comfy1hewSaveImagePreview = null;
    node._comfy1hewPreviewImageEl = null;
    node.imgs = null;

    if (clearPersisted) {
        node.properties = node.properties || {};
        delete node.properties[PREVIEW_PROPERTY_KEY];
    }

    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function applyPreview(node, entries) {
    const normalizedEntries = normalizePreviewEntries(entries);
    if (!normalizedEntries.length) {
        clearPreview(node, { clearPersisted: false });
        return;
    }

    const nextPreviewKey = previewEntriesKey(normalizedEntries);
    if (
        node._comfy1hewSaveImagePreviewKey === nextPreviewKey
        && Array.isArray(node.imgs)
        && node.imgs[0]
    ) {
        node._comfy1hewSaveImagePreview = normalizedEntries;
        node.properties = node.properties || {};
        node.properties[PREVIEW_PROPERTY_KEY] = normalizedEntries;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        return;
    }

    node._comfy1hewSaveImageReqId = (node._comfy1hewSaveImageReqId || 0) + 1;
    const reqId = node._comfy1hewSaveImageReqId;

    Promise.all(normalizedEntries.map(loadPreviewImage)).then((imgs) => {
        if (reqId !== node._comfy1hewSaveImageReqId) {
            return;
        }

        node._comfy1hewSaveImagePreview = normalizedEntries;
        node._comfy1hewSaveImagePreviewKey = nextPreviewKey;
        node._comfy1hewPreviewImageEl = imgs[0] || null;
        node.imgs = imgs;
        node.properties = node.properties || {};
        node.properties[PREVIEW_PROPERTY_KEY] = normalizedEntries;

        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }).catch(() => {
        if (reqId !== node._comfy1hewSaveImageReqId) {
            return;
        }
        clearPreview(node, { clearPersisted: false });
    });
}

registerExtensionOnce("__comfy1hewSaveImageExtensionRegistered", () => app.registerExtension({
    name: "ComfyUI-1hewNodes.save_image",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const typeName = nodeData?.name || nodeType?.type || nodeType?.title;
        if (!SAVE_IMAGE_NODE_NAMES.has(typeName)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            this._comfy1hewSaveImageReqId = 0;
            this._comfy1hewSaveImagePreview = null;
            this._comfy1hewSaveImagePreviewKey = "";
            this._comfy1hewPreviewImageEl = null;
            chainOnRemoved(this, function () {
                this._comfy1hewSaveImageReqId = 0;
                this._comfy1hewSaveImagePreview = null;
                this._comfy1hewSaveImagePreviewKey = "";
                this._comfy1hewPreviewImageEl = null;
                this.imgs = null;
            });
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            const previewEntries = extractPreviewEntries(message);
            if (previewEntries.length) {
                applyPreview(this, previewEntries);
            } else {
                clearPreview(this, { clearPersisted: true });
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const data = arguments[0];
            const savedPreviewEntries = normalizePreviewEntries(
                data?.properties?.[PREVIEW_PROPERTY_KEY]
                ?? this.properties?.[PREVIEW_PROPERTY_KEY],
            );
            const r = onConfigure?.apply(this, arguments);
            if (savedPreviewEntries.length) {
                applyPreview(this, savedPreviewEntries);
            }
            return r;
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function () {
            const r = onSerialize?.apply(this, arguments);
            const serialized = arguments[0];
            const previewEntries = normalizePreviewEntries(
                this._comfy1hewSaveImagePreview
                ?? this.properties?.[PREVIEW_PROPERTY_KEY],
            );
            if (serialized) {
                serialized.properties = {
                    ...(serialized.properties || {}),
                    [PREVIEW_PROPERTY_KEY]: previewEntries.length ? previewEntries : null,
                };
            }
            return r;
        };

    },
}));
