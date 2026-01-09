import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addPreviewMenuOptions, applyPreviewHiddenState } from "./core/preview_menu.js";
import {
    addCopyMediaFrameMenuOption,
    addSaveMediaMenuOption,
    collectDropPayload,
    installImagePreviewLayout,
} from "./core/media_utils.js";
import { saveMaskFromClipspaceToSidecar } from "./core/image_mask_sidecar.js";

app.registerExtension({
    name: "ComfyUI-1hewNodes.load_image",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadImage") {
            return;
        }

        const applyPreviewStyle = (node) => {
            applyPreviewHiddenState(node);
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const r = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;

            // Remove "Save Mask" option if present (robust check)
            if (Array.isArray(options)) {
                for (let i = options.length - 1; i >= 0; i--) {
                    const opt = options[i];
                    if (opt && typeof opt.content === "string" && opt.content.trim() === "Save Mask") {
                        options.splice(i, 1);
                    }
                }
            }

            // Add standard preview options (Hide/Show, etc.)
            addPreviewMenuOptions(options, { app, currentNode: this });

            let imgEl = null;
            if (this.imageWidget && this.imageWidget.element) {
                imgEl = this.imageWidget.element.querySelector("img");
            }

            const getImgElFromNode = (node) =>
                node?.imageWidget?.element?.querySelector("img");

            addSaveMediaMenuOption(options, {
                app,
                currentNode: this,
                content: "Save Image",
                getMediaElFromNode: getImgElFromNode,
                filenamePrefix: "image",
                filenameExt: "png",
            });

            if (imgEl && imgEl.src) {
                addCopyMediaFrameMenuOption(options, {
                    content: "Copy Image",
                    getWidth: () => imgEl.naturalWidth,
                    getHeight: () => imgEl.naturalHeight,
                    drawToCanvas: (ctx) => ctx.drawImage(imgEl, 0, 0),
                    copyErrorMessage: "Failed to copy image to clipboard:",
                    prepareErrorMessage: "Error preparing image copy:",
                });
            }

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this.widgets) {
                const fileWidget = this.widgets.find((w) => w.name === "file");
                if (fileWidget && fileWidget.value) {
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

            const fileWidget = this.widgets.find((w) => w.name === "file");
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
                const uploadedFiles = data?.files;

                if (!uploadedFolder && (!uploadedFiles || uploadedFiles.length === 0)) return;

                let finalPath = uploadedFolder;
                // If backend returns specific file paths and we uploaded files, try to use the direct file path
                if (Array.isArray(uploadedFiles) && uploadedFiles.length > 0) {
                    // If we only have one file, use its full path
                    if (uploadedFiles.length === 1) {
                        finalPath = uploadedFiles[0];
                    }
                }

                if (fileWidget) {
                    fileWidget.value = finalPath;
                    if (typeof fileWidget.callback === "function") {
                        fileWidget.callback();
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

            const fileInputEl = document.createElement("input");
            fileInputEl.type = "file";
            fileInputEl.accept = "image/*";
            fileInputEl.style.display = "none";
            container.appendChild(fileInputEl);

            const uploadWidget = this.addWidget(
                "button",
                "choose file to upload",
                "image",
                () => {
                    app.canvas.node_widget = null;
                    try {
                        fileInputEl.value = "";
                    } catch {}
                    fileInputEl.click();
                }
            );
            uploadWidget.serialize = false;

            fileInputEl.addEventListener("change", async () => {
                const file = fileInputEl.files && fileInputEl.files[0];
                if (!file) return;
                try {
                    await uploadFilesAsFolder(
                        [{ file: file, relativePath: file?.name }],
                        false
                    );
                } catch {}
            });

            imageEl.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                const node = this;
                if (app.canvas) {
                    app.canvas.processContextMenu(node, e);
                }
            });

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
            const { requestAutoSize, updateLayout } = installImagePreviewLayout({
                app,
                node: this,
                imageWidget: this.imageWidget,
                container,
                imageEl,
                allWidget,
            });

            this.updatePreview = async () => {
                this._comfy1hewLoadImageReqId = (this._comfy1hewLoadImageReqId || 0) + 1;
                const reqId = this._comfy1hewLoadImageReqId;

                const file = fileWidget.value;
                const index = indexWidget.value;
                const includeSubdir = includeSubdirWidget
                    ? includeSubdirWidget.value
                    : true;
                const all = allWidget ? allWidget.value : false;
                if (this.imageWidget) {
                    this.imageWidget._comfy1hew_maxAspectRatio = all ? 1.25 : null;
                }
                const trimmedFile = String(file || "").trim();
                if (trimmedFile === "") {
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
                    file: file,
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
            if (fileWidget) fileWidget.callback = this.updatePreview;
            if (indexWidget) indexWidget.callback = this.updatePreview;
            if (includeSubdirWidget) includeSubdirWidget.callback = this.updatePreview;
            if (allWidget) allWidget.callback = this.updatePreview;

            const computeStateKey = () => {
                const p = String(fileWidget ? fileWidget.value : "");
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

            const onPaste = (e) => {
                if (!e.clipboardData) return;
                
                // Check if this node is selected
                if (!app.canvas.selected_nodes || !app.canvas.selected_nodes[this.id]) {
                    return;
                }

                const items = e.clipboardData.items;
                if (!items) return;

                const files = [];
                for (let i = 0; i < items.length; i++) {
                    if (items[i].kind === 'file' && items[i].type.startsWith('image/')) {
                        const file = items[i].getAsFile();
                        if (file) {
                            const name = file.name || "pasted_image.png";
                            files.push({ file: file, relativePath: name });
                        }
                    }
                }
                
                if (files.length > 0) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    uploadFilesAsFolder(files, false);
                }
            };
            document.addEventListener("paste", onPaste, { capture: true });

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                try {
                    document.removeEventListener("paste", onPaste, { capture: true });
                } catch {}
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
            if (fileWidget && fileWidget.value) {
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
                        saveMaskFromClipspaceToSidecar({ node, api, app });
                    }, 0);
                    return r2;
                };
            };
            setTimeout(install, 0);
        }

    },
});
