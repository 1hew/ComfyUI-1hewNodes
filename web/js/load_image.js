import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addPreviewMenuOptions, applyPreviewHiddenState } from "./core/preview_menu.js";

async function srcToDataUrl(src) {
    const res = await fetch(src);
    const blob = await res.blob();
    return await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.readAsDataURL(blob);
    });
}

async function saveMaskFromClipspaceToSidecar(node) {
    const clipspace = window?.ComfyApp?.clipspace;
    const imgs = clipspace?.imgs;
    if (!Array.isArray(imgs) || imgs.length === 0) {
        return;
    }

    const combinedIndex =
        typeof clipspace?.combinedIndex === "number" ? clipspace.combinedIndex : 0;
    const combinedImg = imgs[combinedIndex] || imgs[0];
    const combinedSrc = combinedImg?.src;
    if (!combinedSrc) {
        return;
    }

    const getWidgetValue = (name, fallback) => {
        const w = node?.widgets?.find((x) => x?.name === name);
        return w ? w.value : fallback;
    };

    const path = getWidgetValue("path", "");
    const index = getWidgetValue("index", 0);
    const includeSubdir = getWidgetValue("include_subdir", true);
    const all = Boolean(getWidgetValue("all", false));

    const resolveParams = new URLSearchParams({
        path: path,
        index: index,
        include_subdir: includeSubdir,
        all: all ? "true" : "false",
    });
    const resolved = await api.fetchApi(
        `/1hew/resolve_image_from_folder?${resolveParams.toString()}`
    );
    if (resolved.status !== 200) {
        return;
    }
    const resolvedJson = await resolved.json();
    const imagePath = resolvedJson?.path;
    if (!imagePath) {
        return;
    }

    const maskDataUrl = await srcToDataUrl(combinedSrc);
    if (!maskDataUrl) {
        return;
    }

    await api.fetchApi("/1hew/save_sidecar_mask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image_path: imagePath,
            mask_data_url: maskDataUrl,
        }),
    });

    try {
        if (node?.updatePreview) {
            await node.updatePreview();
        }
    } catch {}

    try {
        app.graph.setDirtyCanvas(true, true);
    } catch {}
}

app.registerExtension({
    name: "ComfyUI-1hewNodes.load_image",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadImage") {
            return;
        }

        const applyPreviewStyle = (node) => {
            applyPreviewHiddenState(node);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this.widgets) {
                const pathWidget = this.widgets.find((w) => w.name === "path");
                if (pathWidget && pathWidget.value) {
                    this._comfy1hewLoadImagePendingPreview = true;
                    setTimeout(() => {
                        const update = this.updatePreview;
                        if (typeof update === "function") {
                            this._comfy1hewLoadImagePendingPreview = false;
                            update.call(this);
                        }
                    }, 0);
                }
            }
            setTimeout(() => applyPreviewStyle(this), 0);
            const ensurePreviewLayout = () => {
                if (this.imageWidget.aspectRatio) {
                    requestAutoSize();
                    updateLayout();
                }
            };
            
            // 初始多次尝试触发布局更新，解决首次加载出画框问题
            setTimeout(ensurePreviewLayout, 100);
            setTimeout(ensurePreviewLayout, 500);

            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const pathWidget = this.widgets.find((w) => w.name === "path");
            const indexWidget = this.widgets.find((w) => w.name === "index");
            const includeSubdirWidget = this.widgets.find(
                (w) => w.name === "include_subdir"
            );
            const allWidget = this.widgets.find((w) => w.name === "all");

            try {
                const baseW =
                    Array.isArray(this.size) &&
                    typeof this.size[0] === "number" &&
                    this.size[0] > 0
                        ? this.size[0]
                        : 200;
                const desiredSize = this.computeSize();
                const baseH =
                    Array.isArray(desiredSize) && typeof desiredSize[1] === "number"
                        ? desiredSize[1]
                        : 0;
            this._comfy1hewLoadImageBaseSize = [baseW, baseH];
        } catch {}
            this._comfy1hewLoadImageWasEmptyPath = true;
            this._comfy1hewLoadImageRedrawQueued = false;
            this._comfy1hewLoadImageHadPreview = false;
            this._comfy1hewLoadImageLastImgSrc = undefined;

            const isValidImageFile = (file) => {
                if (!file) return false;
                if (file.type && file.type.startsWith("image/")) return true;
                const name = (file.name || "").toLowerCase();
                return (
                    name.endsWith(".png") ||
                    name.endsWith(".jpg") ||
                    name.endsWith(".jpeg") ||
                    name.endsWith(".webp") ||
                    name.endsWith(".bmp") ||
                    name.endsWith(".tiff") ||
                    name.endsWith(".gif")
                );
            };

            const uploadFilesAsFolder = async (pairs, hasDirectory) => {
                const files = (pairs || []).filter((p) => isValidImageFile(p?.file));
                if (files.length === 0) return;

                const form = new FormData();
                for (const p of files) {
                    const name = p.relativePath || p.file.name;
                    form.append("files", p.file, name);
                }

                const res = await api.fetchApi("/1hew/upload_images", {
                    method: "POST",
                    body: form,
                });
                if (res.status !== 200) return;

                const data = await res.json();
                const uploadedFolder = data?.folder;
                if (!uploadedFolder) return;

                if (pathWidget) {
                    pathWidget.value = uploadedFolder;
                    if (typeof pathWidget.callback === "function") {
                        pathWidget.callback();
                    }
                }

                if (indexWidget) {
                    indexWidget.value = 0;
                    if (typeof indexWidget.callback === "function") {
                        indexWidget.callback();
                    }
                }

                if (includeSubdirWidget && hasDirectory) {
                    includeSubdirWidget.value = true;
                    if (typeof includeSubdirWidget.callback === "function") {
                        includeSubdirWidget.callback();
                    }
                }

                if (this.updatePreview) {
                    await this.updatePreview();
                }

                app.graph.setDirtyCanvas(true, true);
            };

            const readAllEntries = async (dirEntry) => {
                const reader = dirEntry.createReader();
                const entries = [];
                while (true) {
                    const batch = await new Promise((resolve) => reader.readEntries(resolve));
                    if (!batch || batch.length === 0) break;
                    entries.push(...batch);
                }
                return entries;
            };

            const walkEntry = async (entry) => {
                if (!entry) return [];
                if (entry.isFile) {
                    const file = await new Promise((resolve) => entry.file(resolve));
                    const relativePath = (entry.fullPath || file.name).replace(/^\/+/, "").trim();
                    return [{ file, relativePath }];
                }
                if (entry.isDirectory) {
                    const entries = await readAllEntries(entry);
                    const out = [];
                    for (const e of entries) {
                        const sub = await walkEntry(e);
                        out.push(...sub);
                    }
                    return out;
                }
                return [];
            };

            const collectDropPayload = async (e) => {
                const items = e?.dataTransfer?.items;
                if (items && items.length > 0) {
                    const out = [];
                    let hasDirectory = false;
                    for (const item of items) {
                        const entry = item?.webkitGetAsEntry?.();
                        if (!entry) continue;
                        if (entry.isDirectory) hasDirectory = true;
                        const sub = await walkEntry(entry);
                        out.push(...sub);
                    }
                    return { pairs: out, hasDirectory };
                }

                const files = Array.from(e?.dataTransfer?.files || []);
                return {
                    pairs: files.map((f) => ({ file: f, relativePath: f.name })),
                    hasDirectory: false,
                };
            };

            const imageEl = document.createElement("img");
            Object.assign(imageEl.style, {
                width: "100%",
                maxWidth: "100%",
                height: "auto",
                maxHeight: "calc(100% - 20px)",
                display: "block",
                flex: "0 0 auto",
                objectFit: "contain",
                minHeight: "0",
            });

            const infoEl = document.createElement("div");
            Object.assign(infoEl.style, {
                width: "100%",
                height: "20px",
                lineHeight: "20px",
                textAlign: "center",
                fontSize: "12px",
                color: "#aaa",
                fontFamily: "sans-serif",
                flex: "0 0 20px",
                background: "transparent",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
            });
            infoEl.innerText = "";

            const container = document.createElement("div");
            Object.assign(container.style, {
                display: "none",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                width: "100%",
                height: "100%",
                minHeight: "0",
                overflow: "hidden",
                boxSizing: "border-box",
            });

            container.appendChild(imageEl);
            container.appendChild(infoEl);

            this.imageWidget = this.addDOMWidget(
                "image_preview",
                "div",
                container,
                {
                    serialize: false,
                    hideOnZoom: false,
                }
            );

            this.imageWidget.computeSize = function (width) {
                if (this.aspectRatio) {
                    return [width, width * this.aspectRatio + 20];
                }
                return [width, 0];
            };

            this._comfy1hewImageAutoSizeKey = "";

            const autoSizeToContent = () => {
                if (!this.imageWidget.aspectRatio) {
                    return;
                }

                const maxAspectRatio =
                    typeof this.imageWidget._comfy1hew_maxAspectRatio === "number"
                        ? this.imageWidget._comfy1hew_maxAspectRatio
                        : null;
                const aspectRatio =
                    maxAspectRatio && isFinite(maxAspectRatio)
                        ? Math.min(this.imageWidget.aspectRatio, maxAspectRatio)
                        : this.imageWidget.aspectRatio;

                const width = this.size[0];
                const desiredWidgetHeight = width * aspectRatio + 20;

                let desiredHeight;
                if (Number.isFinite(this.imageWidget.last_y)) {
                    desiredHeight = this.imageWidget.last_y + desiredWidgetHeight;
                } else {
                    try {
                        const computed = this.computeSize?.([
                            this.size[0],
                            this.size[1],
                        ]);
                        if (
                            Array.isArray(computed) &&
                            computed.length >= 2 &&
                            Number.isFinite(computed[1])
                        ) {
                            desiredHeight = computed[1];
                        }
                    } catch {}
                    if (!Number.isFinite(desiredHeight)) {
                        const estimatedTop = 130;
                        desiredHeight = estimatedTop + desiredWidgetHeight;
                    }
                }

                if (allWidget && allWidget.value) {
                    desiredHeight = Math.min(desiredHeight, 420);
                }

                if (this.size[1] + 1 < desiredHeight) {
                    this.setSize([this.size[0], desiredHeight]);
                }
            };

            const requestAutoSize = () => {
                if (!imageEl.naturalWidth || !imageEl.naturalHeight) {
                    return;
                }
                const key = `${imageEl.naturalWidth}x${imageEl.naturalHeight}`;
                if (this._comfy1hewImageAutoSizeKey === key) {
                    return;
                }
                this._comfy1hewImageAutoSizeKey = key;
                setTimeout(autoSizeToContent, 0);
            };

            const updateLayout = () => {
                if (container.dataset.comfy1hewForceHidden === "1") {
                    container.style.height = "0px";
                    container.style.display = "none";
                    app.graph.setDirtyCanvas(true, true);
                    return;
                }
                if (!this.imageWidget?.aspectRatio) {
                    container.style.height = "0px";
                    container.style.display = "none";
                    app.graph.setDirtyCanvas(true, true);
                    return;
                }
                container.style.display = "flex";

                let availableHeight;
                if (Number.isFinite(this.imageWidget?.last_y)) {
                    availableHeight = this.size[1] - this.imageWidget.last_y - 15;
                } else {
                    const maxAspectRatio =
                        typeof this.imageWidget._comfy1hew_maxAspectRatio ===
                        "number"
                            ? this.imageWidget._comfy1hew_maxAspectRatio
                            : null;
                    const aspectRatio =
                        maxAspectRatio && isFinite(maxAspectRatio)
                            ? Math.min(this.imageWidget.aspectRatio, maxAspectRatio)
                            : this.imageWidget.aspectRatio;
                    const width = this.size[0];
                    availableHeight = width * aspectRatio + 20;
                }

                if (availableHeight < 0) availableHeight = 0;
                container.style.height = `${availableHeight}px`;

                app.graph.setDirtyCanvas(true, true);
            };

            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                const r2 = originalOnResize
                    ? originalOnResize.apply(this, arguments)
                    : undefined;
                try {
                    updateLayout();
                } catch {}
                return r2;
            };

            this.updatePreview = async () => {
                this._comfy1hewLoadImageReqId = (this._comfy1hewLoadImageReqId || 0) + 1;
                const reqId = this._comfy1hewLoadImageReqId;

                const path = pathWidget.value;
                const index = indexWidget.value;
                const includeSubdir = includeSubdirWidget
                    ? includeSubdirWidget.value
                    : true;
                const all = allWidget ? allWidget.value : false;
                if (this.imageWidget) {
                    this.imageWidget._comfy1hew_maxAspectRatio = all ? 1.25 : null;
                }
                const trimmedPath = String(path || "").trim();
                if (trimmedPath === "") {
                    this._comfy1hewLoadImageWasEmptyPath = true;
                    const lastImgSrc = this._comfy1hewLoadImageLastImgSrc;

                    try {
                        imageEl.onload = null;
                        imageEl.onerror = null;
                        imageEl.removeAttribute("src");
                    } catch {}
                    this._comfy1hewLoadImageHadPreview = false;
                    this._comfy1hewImageAutoSizeKey = "";

                    this.imageWidget.aspectRatio = undefined;
                    infoEl.innerText = "";
                    updateLayout();
                    this.setSize([this.size[0], 0]);

                    try {
                        const clipspace = window?.ComfyApp?.clipspace;
                        const clipspaceNode = window?.ComfyApp?.clipspace_return_node;
                        const clipspaceImgs = clipspace?.imgs;

                        const normalizeSrc = (src) => {
                            try {
                                const u = new URL(src);
                                u.searchParams.delete("t");
                                u.searchParams.delete("preview");
                                return u.toString();
                            } catch {
                                return src;
                            }
                        };
                        const lastSrcNorm =
                            typeof lastImgSrc === "string"
                                ? normalizeSrc(lastImgSrc)
                                : undefined;

                        const shouldClearClipspace =
                            clipspaceNode === this ||
                            (typeof lastSrcNorm === "string" &&
                                Array.isArray(clipspaceImgs) &&
                                clipspaceImgs.some((i) => {
                                    if (!i?.src) return false;
                                    return normalizeSrc(i.src) === lastSrcNorm;
                                }));

                        if (shouldClearClipspace) {
                            window.ComfyApp.clipspace = null;
                            window.ComfyApp.clipspace_return_node = null;

                            const dialog =
                                window?.comfyAPI?.clipspace?.ClipspaceDialog;
                            if (dialog?.instance?.close) {
                                dialog.instance.close();
                            }

                            const maskEditor =
                                window?.comfyAPI?.maskEditorOld
                                    ?.MaskEditorDialogOld;
                            if (maskEditor?.instance?.close) {
                                maskEditor.instance.close();
                            } else if (maskEditor?.getInstance) {
                                const inst = maskEditor.getInstance();
                                if (inst?.close) {
                                    inst.close();
                                }
                            }
                        }
                    } catch {}
                    this._comfy1hewLoadImageLastImgSrc = undefined;

                    if (app?.graph?.setDirtyCanvas) {
                        app.graph.setDirtyCanvas(true, true);
                    }
                    return;
                }

                this._comfy1hewLoadImageWasEmptyPath = false;

                const params = new URLSearchParams({
                    path: path,
                    include_subdir: includeSubdir,
                    t: Date.now(),
                });
                if (!all) {
                    params.set("index", index);
                }
                params.set("all", all ? "true" : "false");

                const url = `/1hew/view_image_from_folder?${params.toString()}`;
                const normalizeUrl = (src) => {
                    if (!src) return "";
                    try {
                        const u = new URL(src, window.location.href);
                        u.searchParams.delete("t");
                        u.searchParams.delete("preview");
                        return u.toString();
                    } catch {
                        return String(src);
                    }
                };
                const desiredNorm = normalizeUrl(url);
                const currentNorm = normalizeUrl(imageEl.src);

                if (currentNorm !== desiredNorm) {
                    this._comfy1hewImageAutoSizeKey = "";
                    this.imageWidget.aspectRatio = undefined;
                    infoEl.innerText = "";
                    container.style.display = "flex";
                    this.setSize([this.size[0], 0]);

                    imageEl.onload = () => {
                        if (String(imageEl.dataset.comfy1hewReqId || "") !== String(reqId)) {
                            return;
                        }
                        if (imageEl.naturalWidth && imageEl.naturalHeight) {
                            this.imageWidget.aspectRatio =
                                imageEl.naturalHeight / imageEl.naturalWidth;
                            infoEl.innerText = `${imageEl.naturalWidth} x ${imageEl.naturalHeight}`;
                            this._comfy1hewLoadImageHadPreview = true;
                            try {
                                const u = new URL(imageEl.src);
                                u.searchParams.delete("t");
                                u.searchParams.delete("preview");
                                this._comfy1hewLoadImageLastImgSrc = u.toString();
                            } catch {
                                this._comfy1hewLoadImageLastImgSrc = imageEl.src;
                            }
                            requestAutoSize();
                            updateLayout();
                        }
                    };

                    imageEl.onerror = () => {
                        if (String(imageEl.dataset.comfy1hewReqId || "") !== String(reqId)) {
                            return;
                        }
                        this.imageWidget.aspectRatio = undefined;
                        infoEl.innerText = "";
                        this._comfy1hewLoadImageHadPreview = false;
                        updateLayout();
                        this.setSize([this.size[0], 0]);
                    };

                    imageEl.dataset.comfy1hewReqId = String(reqId);
                    imageEl.src = url;
                    updateLayout();
                }
            };

            this.onDropFile = function (file) {
                (async () => {
                    try {
                        await uploadFilesAsFolder([{ file, relativePath: file?.name }], false);
                    } catch {}
                })();
                return true;
            };

            this.onDragDrop = function (e, graphCanvas) {
                (async () => {
                    try {
                        const payload = await collectDropPayload(e);
                        const pairs = (payload?.pairs || []).filter((p) => p?.file);
                        const hasDirectory = Boolean(payload?.hasDirectory);
                        await uploadFilesAsFolder(pairs, hasDirectory);
                    } catch {}
                })();
                return true;
            };

            this.onDragOver = function (e) {
                if (e.dataTransfer) {
                    e.dataTransfer.dropEffect = "copy";
                }
                return true;
            };

            // 监听 widget 变化
            if (pathWidget) pathWidget.callback = this.updatePreview;
            if (indexWidget) indexWidget.callback = this.updatePreview;
            if (includeSubdirWidget) includeSubdirWidget.callback = this.updatePreview;
            if (allWidget) allWidget.callback = this.updatePreview;

            const computeStateKey = () => {
                const p = String(pathWidget ? pathWidget.value : "");
                const i = String(indexWidget ? indexWidget.value : 0);
                const s = String(includeSubdirWidget ? includeSubdirWidget.value : true);
                const a = String(allWidget ? allWidget.value : false);
                return `${p}||${i}||${s}||${a}`;
            };

            this._comfy1hewLoadImageStateKey = computeStateKey();
            if (!this._comfy1hewLoadImagePoller) {
                this._comfy1hewLoadImagePoller = setInterval(() => {
                    try {
                        const nextKey = computeStateKey();
                        if (nextKey === this._comfy1hewLoadImageStateKey) {
                            return;
                        }
                        this._comfy1hewLoadImageStateKey = nextKey;
                        if (typeof this.updatePreview === "function") {
                            this.updatePreview();
                        }
                    } catch {}
                }, 200);
            }

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                try {
                    if (this._comfy1hewLoadImagePoller) {
                        clearInterval(this._comfy1hewLoadImagePoller);
                        this._comfy1hewLoadImagePoller = null;
                    }
                } catch {}
                if (originalOnRemoved) {
                    return originalOnRemoved.apply(this, arguments);
                }
            };

            const ensurePreviewLayout = () => {
                if (this.imageWidget.aspectRatio) {
                    requestAutoSize();
                    updateLayout();
                }
            };
            
            // 初始多次尝试触发布局更新，解决首次加载出画框问题
            setTimeout(ensurePreviewLayout, 100);
            setTimeout(ensurePreviewLayout, 500);

            // 初始加载
            if (pathWidget && pathWidget.value) {
                this.updatePreview();
            }
            if (this._comfy1hewLoadImagePendingPreview) {
                this._comfy1hewLoadImagePendingPreview = false;
                setTimeout(() => {
                    if (this.updatePreview) {
                        this.updatePreview();
                    }
                }, 0);
            }
            setTimeout(() => applyPreviewStyle(this), 0);
            setTimeout(() => applyPreviewStyle(this), 120);
            setTimeout(() => applyPreviewStyle(this), 600);

            return r;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }

            addPreviewMenuOptions(options, { app, currentNode: this });

            const canvas = app.canvas;
            const selectedNodes = canvas.selected_nodes || {};
            let targetNodes = Object.values(selectedNodes);
            if (targetNodes.length === 0) {
                targetNodes = [this];
            }

            if (!options) {
                return;
            }

            options.push({
                content: "Save Mask",
                callback: async () => {
                    for (const node of targetNodes) {
                        await saveMaskFromClipspaceToSidecar(node);
                    }
                },
            });
        };

        if (!window.__comfy1hewLoadImageClipspacePatched) {
            window.__comfy1hewLoadImageClipspacePatched = true;
            const install = () => {
                const comfyApp = window?.ComfyApp;
                if (!comfyApp) {
                    return;
                }
                const originalPaste = comfyApp.pasteFromClipspace;
                if (typeof originalPaste !== "function") {
                    return;
                }
                if (comfyApp.__comfy1hewPasteFromClipspaceWrapped) {
                    return;
                }
                comfyApp.__comfy1hewPasteFromClipspaceWrapped = true;
                comfyApp.pasteFromClipspace = function () {
                    const r2 = originalPaste.apply(this, arguments);
                    setTimeout(() => {
                        const node = window?.ComfyApp?.clipspace_return_node;
                        if (node?.type !== "load_image") {
                            return;
                        }
                        saveMaskFromClipspaceToSidecar(node);
                    }, 0);
                    return r2;
                };
            };
            setTimeout(install, 0);
        }

    },
});
