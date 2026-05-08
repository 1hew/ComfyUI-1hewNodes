import { app } from "../../../scripts/app.js";
import { collectDropPayload } from "./core/media_utils.js";
import { registerExtensionOnce } from "./core/runtime.js";
import { attachCanvasUploadHandlers } from "./core/upload_drop_runtime.js";

function isValidPSFile(file) {
    if (!file) return false;
    const name = String(file.name || file.path || file || "").toLowerCase();
    return name.endsWith(".psd") || name.endsWith(".psb");
}

function getDroppedPath(file) {
    if (!file || typeof file !== "object") return "";
    return String(file.path || file.fullPath || file.webkitRelativePath || "").trim();
}

function isUploadablePSFile(file) {
    return Boolean(
        file
        && typeof file === "object"
        && isValidPSFile(file)
        && typeof file.name === "string"
        && (typeof file.arrayBuffer === "function" || typeof file.stream === "function" || typeof file.size === "number")
    );
}

function normalizeUrl(src) {
    if (!src) return "";
    try {
        const url = new URL(src, window.location.href);
        url.searchParams.delete("t");
        return url.toString();
    } catch {
        return String(src);
    }
}

function uploadPSFileWithProgress(file, onProgress) {
    return new Promise((resolve, reject) => {
        const form = new FormData();
        form.append("files", file, file.name || "file.psd");

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/1hew/upload_ps", true);
        xhr.upload.onprogress = (event) => {
            if (!event.lengthComputable) {
                onProgress?.(null);
                return;
            }
            const percent = Math.max(0, Math.min(100, Math.round((event.loaded / event.total) * 100)));
            onProgress?.(percent);
        };
        xhr.onload = () => {
            if (xhr.status !== 200) {
                reject(new Error(`upload failed: ${xhr.status}`));
                return;
            }
            try {
                resolve(JSON.parse(xhr.responseText || "{}"));
            } catch (error) {
                reject(error);
            }
        };
        xhr.onerror = () => reject(new Error("upload failed"));
        xhr.onabort = () => reject(new Error("upload aborted"));
        xhr.send(form);
    });
}

const loadPSNodes = new Set();
let globalDropBridgeInstalled = false;
const LOAD_PS_GLOBAL_DROP_HANDLED = "__comfy1hewLoadPSGlobalHandled";

function hasPSDFile(event) {
    const files = Array.from(event?.dataTransfer?.files || []);
    return files.some(isValidPSFile);
}

function firstPSDFile(event) {
    return Array.from(event?.dataTransfer?.files || []).find(isValidPSFile) || null;
}

function firstPSDTextPath(event) {
    const dataTransfer = event?.dataTransfer;
    if (!dataTransfer) return "";
    const types = Array.from(dataTransfer.types || []);
    for (const type of types) {
        if (!type || type === "Files") continue;
        try {
            const text = String(dataTransfer.getData(type) || "").trim();
            const path = text.split(/\r?\n/).find((line) => isValidPSFile(line));
            if (path) return path.replace(/^file:\/+/i, "");
        } catch {}
    }
    return "";
}

function hasPotentialFileDrop(event) {
    if (hasPSDFile(event)) return true;
    const items = Array.from(event?.dataTransfer?.items || []);
    return items.some((item) => item?.kind === "file");
}

async function getPSDFromDropEvent(event) {
    const textPath = firstPSDTextPath(event);
    if (textPath) return textPath;

    const file = firstPSDFile(event);
    if (file) return file;

    try {
        const payload = await collectDropPayload(event);
        const pair = (payload?.pairs || []).find((item) => isValidPSFile(item?.file));
        if (pair?.file) return pair.file;
    } catch {}

    return null;
}

function setWidgetValue(widget, value) {
    if (!widget) return;
    widget.value = value;
    try {
        widget.callback?.(value);
    } catch {}
}

function clientToGraph(event) {
    const canvas = app?.canvas;
    const canvasEl = canvas?.canvas;
    const ds = canvas?.ds;
    if (!canvasEl || !ds) return null;
    const rect = canvasEl.getBoundingClientRect();
    return [
        (event.clientX - rect.left) / ds.scale - ds.offset[0],
        (event.clientY - rect.top) / ds.scale - ds.offset[1],
    ];
}

function nodeContainsPoint(node, point) {
    if (!node || !point) return false;
    const pos = node.pos || [0, 0];
    const size = node.size || node._size || [0, 0];
    return (
        point[0] >= pos[0]
        && point[1] >= pos[1]
        && point[0] <= pos[0] + size[0]
        && point[1] <= pos[1] + size[1]
    );
}

function getLoadPSNodeAtEvent(event) {
    const point = clientToGraph(event);
    const nodes = Array.from(loadPSNodes).filter((node) => node?.graph);
    if (point) {
        for (let i = nodes.length - 1; i >= 0; i -= 1) {
            if (nodeContainsPoint(nodes[i], point)) {
                return nodes[i];
            }
        }
    }
    return nodes.length === 1 ? nodes[0] : null;
}

function installGlobalDropBridge() {
    if (globalDropBridgeInstalled) return;
    globalDropBridgeInstalled = true;

    window.addEventListener("dragover", (event) => {
        if (!hasPotentialFileDrop(event)) return;
        const node = getLoadPSNodeAtEvent(event);
        if (!node) return;
        event.preventDefault();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
    }, true);

    window.addEventListener("drop", (event) => {
        if (!hasPotentialFileDrop(event)) return;
        const node = getLoadPSNodeAtEvent(event);
        if (!node || typeof node._comfy1hewUploadPSFile !== "function") return;
        event.preventDefault();
        if (event[LOAD_PS_GLOBAL_DROP_HANDLED]) return;
        event[LOAD_PS_GLOBAL_DROP_HANDLED] = true;
        void (async () => {
            const file = await getPSDFromDropEvent(event);
            if (file) {
                await node._comfy1hewUploadPSFile(file);
            }
        })();
    }, true);
}

registerExtensionOnce("__comfy1hewLoadPSExtensionRegistered", () => app.registerExtension({
    name: "ComfyUI-1hewNodes.load_ps",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadPS") {
            return;
        }
        installGlobalDropBridge();

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this._comfy1hewLoadPSPendingPreview = true;
            this._comfy1hewSchedulePSPreviewUpdate?.(0);
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const fileWidget = this.widgets.find((w) => w.name === "file");
            const indexWidget = this.widgets.find((w) => w.name === "index");
            const includeHiddenWidget = this.widgets.find((w) => w.name === "include_hidden");
            const groupModeWidget = this.widgets.find((w) => w.name === "group_mode");
            const outputModeWidget = this.widgets.find((w) => w.name === "output_mode");
            const previewWidget = this.widgets.find((w) => w.name === "preview");

            this._comfy1hewLoadPSUpdateTimer = null;
            this._comfy1hewLoadPSLastImgSrc = undefined;
            this._comfy1hewPreviewImageEl = null;
            let uploadWidget = null;
            let uploading = false;

            const computeStateKey = () => {
                const p = String(fileWidget ? fileWidget.value : "");
                const i = String(indexWidget ? indexWidget.value : 0);
                const h = String(includeHiddenWidget ? includeHiddenWidget.value : false);
                const g = String(groupModeWidget ? groupModeWidget.value : "layer");
                const o = String(outputModeWidget ? outputModeWidget.value : "all_layers");
                const v = String(previewWidget ? previewWidget.value : false);
                return `${p}||${i}||${h}||${g}||${o}||${v}`;
            };

            const clearPreview = () => {
                this._comfy1hewPreviewImageEl = null;
                this._comfy1hewLoadPSLastImgSrc = undefined;
                this.imgs = null;
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            const updatePreview = async () => {
                this._comfy1hewLoadPSReqId = (this._comfy1hewLoadPSReqId || 0) + 1;
                const reqId = this._comfy1hewLoadPSReqId;
                const file = String(fileWidget?.value || "").trim();
                if (!previewWidget?.value) {
                    clearPreview();
                    return;
                }
                if (!file) {
                    clearPreview();
                    return;
                }

                const params = new URLSearchParams({
                    file,
                    index: String(indexWidget?.value ?? 0),
                    include_hidden: includeHiddenWidget?.value ? "true" : "false",
                    group_mode: String(groupModeWidget?.value || "layer"),
                    output_mode: String(outputModeWidget?.value || "all_layers"),
                    t: String(Date.now()),
                });
                const url = `/1hew/view_ps?${params.toString()}`;
                const desiredNorm = normalizeUrl(url);
                const currentNorm = normalizeUrl(this._comfy1hewLoadPSLastImgSrc);
                if (currentNorm === desiredNorm && Array.isArray(this.imgs) && this.imgs[0]) {
                    app?.graph?.setDirtyCanvas?.(true, true);
                    return;
                }

                const img = new Image();
                img.crossOrigin = "anonymous";
                img.decoding = "async";
                img.onload = () => {
                    if (reqId !== this._comfy1hewLoadPSReqId) return;
                    this._comfy1hewPreviewImageEl = img;
                    this.imgs = [img];
                    try {
                        const imageUrl = new URL(img.src);
                        imageUrl.searchParams.delete("t");
                        this._comfy1hewLoadPSLastImgSrc = imageUrl.toString();
                    } catch {
                        this._comfy1hewLoadPSLastImgSrc = img.src;
                    }
                    try {
                        this.setSizeForImage?.();
                    } catch {}
                    app?.graph?.setDirtyCanvas?.(true, true);
                };
                img.onerror = () => {
                    if (reqId !== this._comfy1hewLoadPSReqId) return;
                    this._comfy1hewPreviewImageEl = null;
                    this._comfy1hewLoadPSLastImgSrc = undefined;
                    this.imgs = null;
                    app?.graph?.setDirtyCanvas?.(true, true);
                };
                img.src = url;
            };

            const schedulePreviewUpdate = (delay = 0) => {
                try {
                    if (this._comfy1hewLoadPSUpdateTimer) {
                        clearTimeout(this._comfy1hewLoadPSUpdateTimer);
                    }
                } catch {}
                this._comfy1hewLoadPSUpdateTimer = setTimeout(() => {
                    this._comfy1hewLoadPSUpdateTimer = null;
                    this._comfy1hewLoadPSPendingPreview = false;
                    this._comfy1hewLoadPSStateKey = computeStateKey();
                    void updatePreview();
                }, delay);
            };
            this._comfy1hewSchedulePSPreviewUpdate = schedulePreviewUpdate;
            this.updatePreview = updatePreview;

            const applyFilePath = (path) => {
                const finalPath = String(path || "").trim();
                if (!finalPath) return false;
                setWidgetValue(fileWidget, finalPath);
                setWidgetValue(indexWidget, 0);
                this._comfy1hewLoadPSStateKey = computeStateKey();
                if (previewWidget?.value) {
                    schedulePreviewUpdate(0);
                } else {
                    clearPreview();
                }
                app?.graph?.setDirtyCanvas?.(true, true);
                return true;
            };

            const uploadFile = async (file) => {
                if (!isValidPSFile(file) || uploading) return;
                const directPath = typeof file === "string" ? file : getDroppedPath(file);
                if (directPath && applyFilePath(directPath)) {
                    return;
                }
                if (!isUploadablePSFile(file)) {
                    if (uploadWidget) {
                        uploadWidget.name = "drop psd file only";
                        app?.graph?.setDirtyCanvas?.(true, true);
                        setTimeout(() => {
                            if (uploadWidget && uploadWidget.name === "drop psd file only") {
                                uploadWidget.name = "choose psd to upload";
                                app?.graph?.setDirtyCanvas?.(true, true);
                            }
                        }, 1200);
                    }
                    return;
                }
                uploading = true;
                const oldName = uploadWidget?.name || "choose psd to upload";
                try {
                    if (uploadWidget) {
                        uploadWidget.name = "uploading psd...0%";
                    }
                    app?.graph?.setDirtyCanvas?.(true, true);

                    const data = await uploadPSFileWithProgress(file, (percent) => {
                        if (!uploadWidget) return;
                        uploadWidget.name = percent === null
                            ? "uploading psd..."
                            : `uploading psd...${percent}%`;
                        app?.graph?.setDirtyCanvas?.(true, true);
                    });
                    if (!data?.path) return;
                    applyFilePath(data.path);
                } catch (error) {
                    console.error("[1hewNodes.load_ps] upload failed:", error);
                    if (uploadWidget) {
                        uploadWidget.name = "upload failed";
                    }
                } finally {
                    uploading = false;
                    setTimeout(() => {
                        if (uploadWidget && uploadWidget.name === "upload failed") {
                            uploadWidget.name = oldName;
                            app?.graph?.setDirtyCanvas?.(true, true);
                        } else if (uploadWidget && uploadWidget.name.startsWith("uploading psd")) {
                            uploadWidget.name = oldName;
                            app?.graph?.setDirtyCanvas?.(true, true);
                        }
                    }, 1200);
                }
                app?.graph?.setDirtyCanvas?.(true, true);
            };
            this._comfy1hewUploadPSFile = uploadFile;

            const fileInputEl = document.createElement("input");
            fileInputEl.type = "file";
            fileInputEl.accept = ".psd,.psb";
            fileInputEl.style.display = "none";
            const openFilePicker = () => {
                try {
                    fileInputEl.value = "";
                } catch {}
                fileInputEl.click();
            };
            uploadWidget = this.addWidget(
                "button",
                "choose psd to upload",
                "psd",
                () => {
                    app.canvas.node_widget = null;
                    openFilePicker();
                }
            );
            uploadWidget.serialize = false;
            fileInputEl.addEventListener("change", async () => {
                const file = fileInputEl.files && fileInputEl.files[0];
                await uploadFile(file);
            });

            attachCanvasUploadHandlers({
                node: this,
                setDragPassthrough: () => {},
                resetDragPassthrough: () => {},
                onFileDrop: (file) => {
                    void uploadFile(file);
                },
                onEventDrop: (event) => {
                    if (event?.[LOAD_PS_GLOBAL_DROP_HANDLED]) {
                        return true;
                    }
                    void (async () => {
                        try {
                            const file = await getPSDFromDropEvent(event);
                            if (file) {
                                await uploadFile(file);
                            }
                        } catch {}
                    })();
                    return true;
                },
            });

            const watchWidgets = [
                fileWidget,
                indexWidget,
                includeHiddenWidget,
                groupModeWidget,
                outputModeWidget,
                previewWidget,
            ];
            for (const widget of watchWidgets) {
                if (widget) widget.callback = () => schedulePreviewUpdate(0);
            }

            this._comfy1hewLoadPSStateKey = computeStateKey();
            if (!this._comfy1hewLoadPSPoller) {
                this._comfy1hewLoadPSPoller = setInterval(() => {
                    try {
                        const nextKey = computeStateKey();
                        if (nextKey === this._comfy1hewLoadPSStateKey) return;
                        this._comfy1hewLoadPSStateKey = nextKey;
                        schedulePreviewUpdate(0);
                    } catch {}
                }, 200);
            }

            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                try {
                    loadPSNodes.delete(this);
                    if (this._comfy1hewLoadPSPoller) {
                        clearInterval(this._comfy1hewLoadPSPoller);
                        this._comfy1hewLoadPSPoller = null;
                    }
                    if (this._comfy1hewLoadPSUpdateTimer) {
                        clearTimeout(this._comfy1hewLoadPSUpdateTimer);
                        this._comfy1hewLoadPSUpdateTimer = null;
                    }
                    this._comfy1hewPreviewImageEl = null;
                    this.imgs = null;
                } catch {}
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            if ((fileWidget && fileWidget.value) || this._comfy1hewLoadPSPendingPreview) {
                schedulePreviewUpdate(0);
            }
            loadPSNodes.add(this);

            return r;
        };
    },
}));
