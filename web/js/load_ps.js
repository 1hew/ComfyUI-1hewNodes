import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
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

            let domPreview = null;
            let inFlightImage = null;
            const ensureDomPreview = () => {
                if (
                    domPreview
                    || !LiteGraph.vueNodesMode
                    || typeof this.addDOMWidget !== "function"
                ) {
                    return domPreview;
                }
                const container = document.createElement("div");
                Object.assign(container.style, {
                    display: "none",
                    width: "100%",
                    height: "100%",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden",
                    boxSizing: "border-box",
                });
                const image = document.createElement("img");
                Object.assign(image.style, {
                    display: "block",
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                });
                container.appendChild(image);
                const widget = this.addDOMWidget(
                    "ps_preview",
                    "div",
                    container,
                    { serialize: false },
                );
                if (!widget) {
                    return null;
                }
                widget.serialize = false;
                widget.computeSize = (width) => [width, widget._comfy1hewHeight || 0];
                domPreview = { container, image, widget };
                return domPreview;
            };

            const updateDomPreview = (image) => {
                const dom = ensureDomPreview();
                if (!dom) {
                    return;
                }
                if (!image) {
                    dom.container.style.display = "none";
                    dom.widget._comfy1hewHeight = 0;
                    app?.graph?.setDirtyCanvas?.(true, true);
                    return;
                }
                dom.image.src = image.src;
                dom.container.style.display = "flex";
                const width = Number(this.size?.[0]) || 240;
                const ratio = image.naturalHeight / Math.max(1, image.naturalWidth);
                const height = Math.max(120, Math.min(520, width * ratio));
                dom.widget._comfy1hewHeight = height;
                try {
                    this.setSize?.([width, Math.max(Number(this._comfy1hewLoadPSBaseSize?.[1]) || 0, 0) + height]);
                } catch {}
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            this._comfy1hewLoadPSUpdateTimer = null;
            this._comfy1hewLoadPSLastImgSrc = undefined;
            this._comfy1hewPreviewImageEl = null;
            this._comfy1hewLoadPSBaseSize = null;
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

            const computeDataKey = () => {
                const p = String(fileWidget ? fileWidget.value : "");
                const i = String(indexWidget ? indexWidget.value : 0);
                const h = String(includeHiddenWidget ? includeHiddenWidget.value : false);
                const g = String(groupModeWidget ? groupModeWidget.value : "layer");
                const o = String(outputModeWidget ? outputModeWidget.value : "all_layers");
                return `${p}||${i}||${h}||${g}||${o}`;
            };

            let batchViewer = null;
            const batchState = {
                key: "",
                version: "",
                count: 0,
                names: [],
                sizes: [],
                index: 0,
                reqId: 0,
                infoReq: 0,
                cache: new Map(),
            };

            const nativeBatchState = {
                key: "",
                version: "",
                images: null,
            };

            const updateNativeBatch = async (dataKey) => {
                const file = String(fileWidget?.value || "").trim();
                if (!file) {
                    nativeBatchState.key = "";
                    nativeBatchState.images = null;
                    clearPreview();
                    return;
                }
                if (nativeBatchState.key !== dataKey || !nativeBatchState.images?.length) {
                    const params = new URLSearchParams({
                        file,
                        include_hidden: includeHiddenWidget?.value ? "true" : "false",
                        group_mode: String(groupModeWidget?.value || "layer"),
                    });
                    const response = await api.fetchApi(
                        `/1hew/view_ps_info?${params.toString()}`,
                    );
                    if (!response.ok) {
                        clearPreview();
                        return;
                    }
                    const data = await response.json();
                    const count = Number(data?.count) || 0;
                    if (count <= 0) {
                        clearPreview();
                        return;
                    }
                    const version = String(data?.version || "");
                    const images = await Promise.all(
                        Array.from({ length: count }, (_, index) => {
                            const image = new Image();
                            image.crossOrigin = "anonymous";
                            image.decoding = "async";
                            const layerParams = new URLSearchParams({
                                file,
                                include_hidden: includeHiddenWidget?.value ? "true" : "false",
                                group_mode: String(groupModeWidget?.value || "layer"),
                                index: String(index),
                                v: version,
                            });
                            return new Promise((resolve) => {
                                image.onload = () => resolve(image);
                                image.onerror = () => resolve(image);
                                image.src = `/1hew/view_ps_layer?${layerParams.toString()}`;
                            });
                        }),
                    );
                    nativeBatchState.key = dataKey;
                    nativeBatchState.version = version;
                    nativeBatchState.images = images;
                }

                if (!Array.isArray(this._comfy1hewLoadPSBaseSize)) {
                    this._comfy1hewLoadPSBaseSize = [
                        Number(this.size?.[0]) || 240,
                        Number(this.size?.[1]) || 0,
                    ];
                }
                this.imgs = nativeBatchState.images;
                this.imageIndex = 0;
                try {
                    this.setSizeForImage?.();
                } catch {}
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            const ensureBatchViewer = () => {
                if (batchViewer || typeof this.addDOMWidget !== "function") {
                    return batchViewer;
                }
                const container = document.createElement("div");
                Object.assign(container.style, {
                    display: "none",
                    position: "relative",
                    width: "100%",
                    height: "100%",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden",
                    boxSizing: "border-box",
                });

                const image = document.createElement("img");
                Object.assign(image.style, {
                    display: "block",
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                    pointerEvents: "none",
                });

                const toolbar = document.createElement("div");
                Object.assign(toolbar.style, {
                    position: "absolute",
                    top: "6px",
                    left: "50%",
                    transform: "translateX(-50%)",
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                    padding: "2px 6px",
                    borderRadius: "6px",
                    background: "rgba(0, 0, 0, 0.55)",
                    color: "#fff",
                    fontFamily: "inherit",
                    fontSize: "12px",
                    lineHeight: "18px",
                    zIndex: "3",
                });

                const makeButton = (label, step) => {
                    const button = document.createElement("button");
                    button.type = "button";
                    button.textContent = label;
                    Object.assign(button.style, {
                        width: "22px",
                        height: "20px",
                        padding: "0",
                        border: "none",
                        borderRadius: "4px",
                        background: "transparent",
                        color: "inherit",
                        cursor: "pointer",
                        fontSize: "16px",
                        lineHeight: "18px",
                    });
                    button.addEventListener("click", (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (batchState.count <= 0) {
                            return;
                        }
                        const next = (
                            batchState.index + step + batchState.count
                        ) % batchState.count;
                        void loadBatchIndex(next);
                    });
                    return button;
                };

                const prevButton = makeButton("‹", -1);
                const counter = document.createElement("span");
                counter.textContent = "0 / 0";
                const nextButton = makeButton("›", 1);
                toolbar.append(prevButton, counter, nextButton);

                const nameEl = document.createElement("div");
                Object.assign(nameEl.style, {
                    position: "absolute",
                    left: "0",
                    right: "0",
                    bottom: "4px",
                    textAlign: "center",
                    color: "#ddd",
                    fontFamily: "inherit",
                    fontSize: "12px",
                    lineHeight: "16px",
                    textShadow: "0 1px 2px rgba(0,0,0,0.8)",
                    pointerEvents: "none",
                    zIndex: "2",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                });

                container.append(image, toolbar, nameEl);
                const widget = this.addDOMWidget(
                    "ps_layers_preview",
                    "div",
                    container,
                    { serialize: false },
                );
                if (!widget) {
                    return null;
                }
                widget.serialize = false;
                widget.computeSize = (width) => [width, widget._comfy1hewHeight || 0];
                batchViewer = {
                    container,
                    image,
                    toolbar,
                    counter,
                    nameEl,
                    widget,
                };
                return batchViewer;
            };

            const updateBatchControls = () => {
                if (!batchViewer) {
                    return;
                }
                batchViewer.counter.textContent = batchState.count
                    ? `${batchState.index + 1} / ${batchState.count}`
                    : "0 / 0";
                const name = batchState.names[batchState.index] || "";
                const size = batchState.sizes[batchState.index];
                const dimensions = Array.isArray(size)
                    ? `${size[0]} × ${size[1]}`
                    : "";
                batchViewer.nameEl.textContent = [name, dimensions]
                    .filter(Boolean)
                    .join("  ·  ");
                batchViewer.toolbar.style.display =
                    batchState.count > 1 ? "flex" : "none";
            };

            const updateBatchSize = (image) => {
                const viewer = ensureBatchViewer();
                if (!viewer) {
                    return;
                }
                viewer.image.src = image.src;
                viewer.container.style.display = "flex";
                if (!Array.isArray(this._comfy1hewLoadPSBaseSize)) {
                    this._comfy1hewLoadPSBaseSize = [
                        Number(this.size?.[0]) || 240,
                        Number(this.size?.[1]) || 0,
                    ];
                }
                const width = Number(this.size?.[0]) || 240;
                const ratio = image.naturalHeight / Math.max(1, image.naturalWidth);
                const height = Math.max(120, Math.min(520, width * ratio));
                viewer.widget._comfy1hewHeight = height;
                try {
                    this.setSize?.([
                        width,
                        Math.max(Number(this._comfy1hewLoadPSBaseSize[1]) || 0, 0) + height,
                    ]);
                } catch {}
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            const hideBatchViewer = () => {
                batchState.reqId += 1;
                if (batchViewer) {
                    batchViewer.container.style.display = "none";
                    batchViewer.widget._comfy1hewHeight = 0;
                }
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            const clearBatchViewer = () => {
                hideBatchViewer();
                batchState.key = "";
                batchState.version = "";
                batchState.count = 0;
                batchState.names = [];
                batchState.sizes = [];
                batchState.index = 0;
                batchState.cache.clear();
            };

            const loadBatchIndex = async (index) => {
                const file = String(fileWidget?.value || "").trim();
                if (!file || batchState.count <= 0) {
                    return;
                }
                const next = ((index % batchState.count) + batchState.count) % batchState.count;
                batchState.index = next;
                updateBatchControls();
                const cacheKey = `${batchState.version}|${next}`;
                const cached = batchState.cache.get(cacheKey);
                if (cached) {
                    updateBatchSize(cached);
                    return;
                }

                const reqId = ++batchState.reqId;
                const params = new URLSearchParams({
                    file,
                    include_hidden: includeHiddenWidget?.value ? "true" : "false",
                    group_mode: String(groupModeWidget?.value || "layer"),
                    index: String(next),
                    v: batchState.version,
                });
                const image = new Image();
                image.crossOrigin = "anonymous";
                image.decoding = "async";
                image.onload = () => {
                    if (
                        reqId !== batchState.reqId
                        || !previewWidget?.value
                        || String(outputModeWidget?.value || "") !== "all_layers"
                    ) {
                        return;
                    }
                    batchState.cache.set(cacheKey, image);
                    updateBatchSize(image);
                };
                image.onerror = () => {
                    if (reqId !== batchState.reqId) {
                        return;
                    }
                    app?.graph?.setDirtyCanvas?.(true, true);
                };
                image.src = `/1hew/view_ps_layer?${params.toString()}`;
            };

            const updateBatchViewer = async (dataKey) => {
                const file = String(fileWidget?.value || "").trim();
                if (!file) {
                    clearBatchViewer();
                    return;
                }
                if (batchState.key !== dataKey || batchState.count <= 0) {
                    const reqId = ++batchState.infoReq;
                    const params = new URLSearchParams({
                        file,
                        include_hidden: includeHiddenWidget?.value ? "true" : "false",
                        group_mode: String(groupModeWidget?.value || "layer"),
                    });
                    try {
                        const response = await api.fetchApi(
                            `/1hew/view_ps_info?${params.toString()}`,
                        );
                        if (
                            reqId !== batchState.infoReq
                            || !previewWidget?.value
                            || String(outputModeWidget?.value || "") !== "all_layers"
                        ) {
                            return;
                        }
                        if (!response.ok) {
                            clearBatchViewer();
                            return;
                        }
                        const data = await response.json();
                        const count = Number(data?.count) || 0;
                        if (count <= 0) {
                            clearBatchViewer();
                            return;
                        }
                        if (batchState.key !== dataKey) {
                            batchState.cache.clear();
                            batchState.index = 0;
                        }
                        batchState.key = dataKey;
                        batchState.version = String(data?.version || "");
                        batchState.count = count;
                        batchState.names = Array.isArray(data?.names) ? data.names : [];
                        batchState.sizes = Array.isArray(data?.sizes) ? data.sizes : [];
                    } catch (error) {
                        console.warn("[1hewNodes.load_ps] layer preview failed:", error);
                        clearBatchViewer();
                        return;
                    }
                }
                ensureBatchViewer();
                if (!batchViewer) {
                    return;
                }
                batchViewer.container.style.display = "flex";
                updateBatchControls();
                const cacheKey = `${batchState.version}|${batchState.index}`;
                const cached = batchState.cache.get(cacheKey);
                if (cached) {
                    updateBatchSize(cached);
                } else {
                    await loadBatchIndex(batchState.index);
                }
            };

            const corePreviewWidget = () => this.widgets?.find(
                (widget) => widget?.name === "$$canvas-image-preview",
            );

            const detachCorePreviewWidget = () => {
                const widget = corePreviewWidget();
                if (!widget) {
                    return;
                }
                const index = this.widgets.indexOf(widget);
                if (index >= 0) {
                    this.widgets.splice(index, 1);
                    this._comfy1hewLoadPSCorePreviewWidget = widget;
                }
            };

            const attachCorePreviewWidget = () => {
                const widget = this._comfy1hewLoadPSCorePreviewWidget;
                if (widget && !this.widgets.includes(widget)) {
                    this.widgets.push(widget);
                }
            };

            const collapsePreviewFrame = () => {
                const base = this._comfy1hewLoadPSBaseSize;
                if (!Array.isArray(base) || !Number.isFinite(base[1])) {
                    return;
                }
                const width = Number(this.size?.[0]) || base[0] || 240;
                try {
                    this.setSize?.([width, base[1]]);
                } catch {}
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            const hidePreview = () => {
                if (inFlightImage) {
                    try {
                        inFlightImage.src = "";
                    } catch {}
                    inFlightImage = null;
                }
                this.imgs = null;
                detachCorePreviewWidget();
                updateDomPreview(null);
                hideBatchViewer();
                collapsePreviewFrame();
                app?.graph?.setDirtyCanvas?.(true, true);
            };

            const clearPreview = () => {
                this._comfy1hewPreviewImageEl = null;
                this._comfy1hewLoadPSLastImgSrc = undefined;
                this._comfy1hewLoadPSDataKey = undefined;
                nativeBatchState.key = "";
                nativeBatchState.images = null;
                clearBatchViewer();
                hidePreview();
            };

            const updatePreview = async () => {
                this._comfy1hewLoadPSReqId = (this._comfy1hewLoadPSReqId || 0) + 1;
                const reqId = this._comfy1hewLoadPSReqId;
                const file = String(fileWidget?.value || "").trim();
                const dataKey = computeDataKey();
                if (
                    previewWidget?.value
                    && String(outputModeWidget?.value || "") === "all_layers"
                ) {
                    if (!LiteGraph.vueNodesMode) {
                        updateDomPreview(null);
                        hideBatchViewer();
                        await updateNativeBatch(dataKey);
                        return;
                    }
                    this.imgs = null;
                    detachCorePreviewWidget();
                    updateDomPreview(null);
                    await updateBatchViewer(dataKey);
                    return;
                }
                if (!previewWidget?.value) {
                    this._comfy1hewLoadPSReqId = (this._comfy1hewLoadPSReqId || 0) + 1;
                    if (this._comfy1hewLoadPSDataKey !== dataKey) {
                        this._comfy1hewPreviewImageEl = null;
                        this._comfy1hewLoadPSLastImgSrc = undefined;
                        this._comfy1hewLoadPSDataKey = undefined;
                    }
                    hidePreview();
                    return;
                }
                hideBatchViewer();
                if (!file) {
                    clearPreview();
                    return;
                }
                if (
                    this._comfy1hewPreviewImageEl
                    && this._comfy1hewLoadPSDataKey === dataKey
                ) {
                    // Toggle closed then open again: restore the cached frame
                    // immediately instead of re-rendering the PSD.
                    attachCorePreviewWidget();
                    this.imgs = [this._comfy1hewPreviewImageEl];
                    updateDomPreview(this._comfy1hewPreviewImageEl);
                    try {
                        if (typeof this.setSizeForImage === "function") {
                            this.setSizeForImage();
                        }
                    } catch {}
                    app?.graph?.setDirtyCanvas?.(true, true);
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

                if (inFlightImage) {
                    try {
                        inFlightImage.src = "";
                    } catch {}
                    inFlightImage = null;
                }
                const img = new Image();
                inFlightImage = img;
                img.crossOrigin = "anonymous";
                img.decoding = "async";
                img.onload = () => {
                    if (inFlightImage === img) {
                        inFlightImage = null;
                    }
                    if (reqId !== this._comfy1hewLoadPSReqId || !previewWidget?.value) return;
                    if (!Array.isArray(this._comfy1hewLoadPSBaseSize)) {
                        this._comfy1hewLoadPSBaseSize = [
                            Number(this.size?.[0]) || 240,
                            Number(this.size?.[1]) || 0,
                        ];
                    }
                    this._comfy1hewPreviewImageEl = img;
                    this._comfy1hewLoadPSDataKey = dataKey;
                    attachCorePreviewWidget();
                    this.imgs = [img];
                    updateDomPreview(img);
                    try {
                        const imageUrl = new URL(img.src);
                        imageUrl.searchParams.delete("t");
                        this._comfy1hewLoadPSLastImgSrc = imageUrl.toString();
                    } catch {
                        this._comfy1hewLoadPSLastImgSrc = img.src;
                    }
                    try {
                        if (typeof this.setSizeForImage === "function") {
                            this.setSizeForImage();
                        } else {
                            const width = Number(this.size?.[0]) || 240;
                            const ratio = img.naturalHeight / Math.max(1, img.naturalWidth);
                            this.setSize?.([width, Math.max(120, width * ratio + 20)]);
                        }
                    } catch {}
                    app?.graph?.setDirtyCanvas?.(true, true);
                };
                img.onerror = () => {
                    if (inFlightImage === img) {
                        inFlightImage = null;
                    }
                    if (reqId !== this._comfy1hewLoadPSReqId) return;
                    clearPreview();
                };
                img.src = url;
            };

            const schedulePreviewUpdate = (delay = 120) => {
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
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    const result = originalCallback?.apply(this, arguments);
                    if (widget === previewWidget) {
                        if (!previewWidget.value) {
                            void updatePreview();
                        } else {
                            schedulePreviewUpdate(0);
                        }
                    } else {
                        schedulePreviewUpdate(150);
                    }
                    return result;
                };
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
                }, 500);
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
                    this._comfy1hewLoadPSBaseSize = null;
                    this._comfy1hewLoadPSCorePreviewWidget = null;
                    nativeBatchState.key = "";
                    nativeBatchState.images = null;
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
