import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { collectDropPayload } from "./core/media_utils.js";
import { saveMaskFromClipspaceToSidecar } from "./core/image_mask_sidecar.js";
import { extendLoadImageMenu } from "./core/load_image_menu.js";
import { createLoadImagePreviewController } from "./core/load_image_preview.js";
import {
    installLoadImageClipspacePatch,
    registerLoadImagePasteTarget,
} from "./core/load_image_runtime.js";
import { createLoadImageUploader } from "./core/load_image_upload.js";
import { chainOnRemoved, createDisposer, registerExtensionOnce } from "./core/runtime.js";
import { attachCanvasUploadHandlers } from "./core/upload_drop_runtime.js";

registerExtensionOnce("__comfy1hewLoadImageExtensionRegistered", () => app.registerExtension({
    name: "ComfyUI-1hewNodes.load_image",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadImage") {
            return;
        }

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const r = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
            extendLoadImageMenu({ app, node: this, options });
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;

            if (this.widgets) {
                const fileWidget = this.widgets.find((w) => w.name === "file");
                if (fileWidget && fileWidget.value) {
                    this._comfy1hewLoadImagePendingPreview = true;
                    this._comfy1hewSchedulePreviewUpdate?.(0);
                }
            }

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
            this._comfy1hewImagePreviewUserResized = false;
            this._comfy1hewLoadImageUpdateTimer = null;
            this._comfy1hewPreviewStyleTimers = [];

            const computeStateKey = () => {
                const p = String(fileWidget ? fileWidget.value : "");
                const i = String(indexWidget ? indexWidget.value : 0);
                const s = String(includeSubdirWidget ? includeSubdirWidget.value : true);
                const a = String(allWidget ? allWidget.value : false);
                return `${p}||${i}||${s}||${a}`;
            };

            const schedulePreviewUpdate = (delay = 0) => {
                try {
                    if (this._comfy1hewLoadImageUpdateTimer) {
                        clearTimeout(this._comfy1hewLoadImageUpdateTimer);
                    }
                } catch {}
                this._comfy1hewLoadImageUpdateTimer = setTimeout(() => {
                    this._comfy1hewLoadImageUpdateTimer = null;
                    this._comfy1hewLoadImagePendingPreview = false;
                    this._comfy1hewLoadImageStateKey = computeStateKey();
                    if (typeof this.updatePreview === "function") {
                        this.updatePreview();
                    }
                }, delay);
            };
            this._comfy1hewSchedulePreviewUpdate = schedulePreviewUpdate;
            const ensurePreviewPassthrough = () => {};

            const { uploadFilesAsFolder } = createLoadImageUploader({
                app,
                api,
                node: this,
                fileWidget,
                indexWidget,
                includeSubdirWidget,
                computeStateKey,
                onSettled: () => ensurePreviewPassthrough?.(),
            });
            const unregisterPasteTarget = registerLoadImagePasteTarget({
                app,
                node: this,
                handlePasteFiles: (files) => uploadFilesAsFolder(files, false),
            });

            const fileInputEl = document.createElement("input");
            fileInputEl.type = "file";
            fileInputEl.accept = "image/*";
            fileInputEl.style.display = "none";

            const openFilePicker = () => {
                try {
                    fileInputEl.value = "";
                } catch {}
                fileInputEl.click();
            };
            const uploadWidget = this.addWidget(
                "button",
                "choose file to upload",
                "image",
                () => {
                    app.canvas.node_widget = null;
                    openFilePicker();
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

            const disposables = createDisposer();

            this._comfy1hewImageAutoSizeKey = "";
            this._comfy1hewPreviewImageEl = null;

            this.updatePreview = createLoadImagePreviewController({
                app,
                node: this,
                fileWidget,
                indexWidget,
                includeSubdirWidget,
                allWidget,
            });

            attachCanvasUploadHandlers({
                node: this,
                setDragPassthrough: () => {},
                resetDragPassthrough: () => {},
                onFileDrop: (file) => {
                    void uploadFilesAsFolder([{ file, relativePath: file?.name }], false);
                },
                onEventDrop: (event) => {
                    void (async () => {
                        try {
                            const payload = await collectDropPayload(event);
                            const pairs = (payload?.pairs || []).filter((pair) => pair?.file);
                            const hasDirectory = Boolean(payload?.hasDirectory);
                            await uploadFilesAsFolder(pairs, hasDirectory);
                        } catch {}
                    })();
                    return true;
                },
            });

            // 监听 widget 变化
            if (fileWidget) fileWidget.callback = () => schedulePreviewUpdate(0);
            if (indexWidget) indexWidget.callback = () => schedulePreviewUpdate(0);
            if (includeSubdirWidget) {
                includeSubdirWidget.callback = () => schedulePreviewUpdate(0);
            }
            if (allWidget) allWidget.callback = () => schedulePreviewUpdate(0);

            this._comfy1hewLoadImageStateKey = computeStateKey();
            if (!this._comfy1hewLoadImagePoller) {
                this._comfy1hewLoadImagePoller = setInterval(() => {
                    try {
                        const nextKey = computeStateKey();
                        if (nextKey === this._comfy1hewLoadImageStateKey) {
                            return;
                        }
                        this._comfy1hewLoadImageStateKey = nextKey;
                        schedulePreviewUpdate(0);
                    } catch {}
                }, 200);
            }

            chainOnRemoved(this, function () {
                try {
                    disposables.dispose();
                } catch {}
                try {
                    unregisterPasteTarget?.();
                    if (this._comfy1hewLoadImagePoller) {
                        clearInterval(this._comfy1hewLoadImagePoller);
                        this._comfy1hewLoadImagePoller = null;
                    }
                    if (this._comfy1hewLoadImageUpdateTimer) {
                        clearTimeout(this._comfy1hewLoadImageUpdateTimer);
                        this._comfy1hewLoadImageUpdateTimer = null;
                    }
                    if (Array.isArray(this._comfy1hewPreviewStyleTimers)) {
                        for (const timer of this._comfy1hewPreviewStyleTimers) {
                            clearTimeout(timer);
                        }
                        this._comfy1hewPreviewStyleTimers = [];
                    }
                    this._comfy1hewPreviewImageEl = null;
                    this.imgs = null;
                } catch {}
            });

            if (fileWidget && fileWidget.value || this._comfy1hewLoadImagePendingPreview) {
                schedulePreviewUpdate(0);
            }

            return r;
        };
        installLoadImageClipspacePatch({
            app,
            api,
            saveMaskFromClipspaceToSidecar,
        });

    },
}));
