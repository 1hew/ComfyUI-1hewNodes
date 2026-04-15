import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { applyPreviewHiddenState } from "./core/preview_menu.js";
import { collectDropPayload, installImagePreviewLayout } from "./core/media_utils.js";
import { saveMaskFromClipspaceToSidecar } from "./core/image_mask_sidecar.js";
import { createLoadImageDom } from "./core/load_image_dom.js";
import { extendLoadImageMenu } from "./core/load_image_menu.js";
import { createLoadImagePreviewController } from "./core/load_image_preview.js";
import {
    installLoadImageClipspacePatch,
    registerLoadImagePasteTarget,
} from "./core/load_image_runtime.js";
import { createLoadImageUploader } from "./core/load_image_upload.js";
import {
    installLoadImageSetSizeGuard,
    scheduleLoadImagePreserveRefresh,
    scheduleLoadImagePreviewStyleSync,
} from "./core/load_image_node_runtime.js";
import { createDomWidgetInteractionManager } from "./core/dom_widget_runtime.js";
import { chainOnRemoved, createDisposer, registerExtensionOnce } from "./core/runtime.js";
import { attachCanvasUploadHandlers, bindDomUploadDropTargets } from "./core/upload_drop_runtime.js";

registerExtensionOnce("__comfy1hewLoadImageExtensionRegistered", () => app.registerExtension({
    name: "ComfyUI-1hewNodes.load_image",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadImage") {
            return;
        }
        installLoadImageSetSizeGuard(nodeType);

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const r = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
            extendLoadImageMenu({ app, node: this, options });
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const data = arguments[0];
            const serializedSize =
                data && Array.isArray(data.size) && data.size.length >= 2
                    ? [data.size[0], data.size[1]]
                    : null;

            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;

            if (this.widgets) {
                const fileWidget = this.widgets.find((w) => w.name === "file");
                if (fileWidget && fileWidget.value) {
                    this._comfy1hewLoadImagePendingPreview = true;
                    this._comfy1hewSchedulePreviewUpdate?.(0);
                } else {
                    scheduleLoadImagePreviewStyleSync(
                        this,
                        applyPreviewHiddenState,
                        [0]
                    );
                }
            }
            scheduleLoadImagePreserveRefresh(this, { serializedSize });

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
                const currentH = Array.isArray(this.size) ? this.size[1] : 0;
                const hasPath =
                    fileWidget && String(fileWidget.value || "").trim() !== "";
                if (
                    hasPath
                    && Number.isFinite(currentH)
                    && Number.isFinite(baseH)
                    && currentH > baseH + 5
                ) {
                    this._comfy1hewPreserveFrameHeightUntilPreview = currentH;
                }
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

            const { uploadFilesAsFolder } = createLoadImageUploader({
                app,
                api,
                node: this,
                fileWidget,
                indexWidget,
                includeSubdirWidget,
                computeStateKey,
            });
            const unregisterPasteTarget = registerLoadImagePasteTarget({
                app,
                node: this,
                handlePasteFiles: (files) => uploadFilesAsFolder(files, false),
            });

            const openFilePicker = () => {
                try {
                    fileInputEl.value = "";
                } catch {}
                fileInputEl.click();
            };
            const {
                container,
                imageEl,
                infoEl,
                fileInputEl,
            } = createLoadImageDom({
                app,
                node: this,
                openFilePicker,
            });

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

            this.imageWidget = this.addDOMWidget(
                "image_preview",
                "div",
                container,
                {
                    serialize: false,
                    hideOnZoom: false,
                }
            );
            if (this.imageWidget?.element?.style) {
                this.imageWidget.element.style.pointerEvents = "none";
            }

            const disposables = createDisposer();
            const interaction = createDomWidgetInteractionManager({
                getWidgetElement: () => this.imageWidget?.element,
                container,
                interactiveElements: [imageEl, infoEl],
            });
            const { setDragPassthrough, resetDragPassthrough } = interaction;
            disposables.add(interaction.bindGlobalDragCleanup());

            this.imageWidget.computeSize = function (width) {
                if (this.aspectRatio) {
                    const maxAspectRatio =
                        typeof this._comfy1hew_maxAspectRatio === "number"
                            ? this._comfy1hew_maxAspectRatio
                            : null;
                    const aspectRatio =
                        maxAspectRatio && isFinite(maxAspectRatio)
                            ? Math.min(this.aspectRatio, maxAspectRatio)
                            : this.aspectRatio;
                    let height = width * aspectRatio + 20;
                    const maxPreviewHeight =
                        typeof this._comfy1hew_maxPreviewHeight === "number"
                            ? this._comfy1hew_maxPreviewHeight
                            : null;
                    if (maxPreviewHeight && isFinite(maxPreviewHeight)) {
                        height = Math.min(height, maxPreviewHeight);
                    }
                    return [width, height];
                }
                return [width, 0];
            };

            this._comfy1hewImageAutoSizeKey = "";
            const { ensurePreviewLayout, updateLayout } = installImagePreviewLayout({
                app,
                node: this,
                imageWidget: this.imageWidget,
                container,
                imageEl,
                allWidget,
                fileWidget,
            });

            const resetNodeHeightToBase = () => {
                try {
                    const baseH =
                        Array.isArray(this._comfy1hewLoadImageBaseSize)
                        && Number.isFinite(this._comfy1hewLoadImageBaseSize[1])
                            ? this._comfy1hewLoadImageBaseSize[1]
                            : null;
                    if (typeof baseH === "number" && baseH > 0) {
                        this.setSize([this.size[0], baseH]);
                        return;
                    }
                } catch {}
                // Fallback: avoid collapsing to zero-height which can make the DOM preview overflow the node frame
                try {
                    this.setSize([this.size[0], 130]);
                } catch {}
            };

            const ensureProvisionalPreviewAspect = () => {
                try {
                    if (
                        this.imageWidget?.aspectRatio
                        || !Array.isArray(this.size)
                        || !Number.isFinite(this.size[0])
                        || !Number.isFinite(this.size[1])
                        || this.size[0] <= 0
                    ) {
                        return false;
                    }

                    let availableHeight = null;
                    if (Number.isFinite(this.imageWidget?.last_y)) {
                        availableHeight = this.size[1] - this.imageWidget.last_y - 15;
                    } else {
                        const baseH =
                            Array.isArray(this._comfy1hewLoadImageBaseSize)
                            && Number.isFinite(this._comfy1hewLoadImageBaseSize[1])
                                ? this._comfy1hewLoadImageBaseSize[1]
                                : 130;
                        availableHeight = this.size[1] - baseH;
                    }

                    if (!Number.isFinite(availableHeight) || availableHeight <= 20) {
                        return false;
                    }

                    const provisionalAspect = Math.max(
                        0.05,
                        Math.min(8, (availableHeight - 20) / this.size[0])
                    );
                    this.imageWidget.aspectRatio = provisionalAspect;
                    return true;
                } catch {
                    return false;
                }
            };

            const dragTargets = [container, imageEl, infoEl];
            bindDomUploadDropTargets({
                disposables,
                interaction,
                dragTargets,
                onDrop: async (event) => {
                    try {
                        const payload = await collectDropPayload(event);
                        const pairs = (payload?.pairs || []).filter((pair) => pair?.file);
                        const hasDirectory = Boolean(payload?.hasDirectory);
                        await uploadFilesAsFolder(pairs, hasDirectory);
                    } catch {}
                },
            });

            this.updatePreview = createLoadImagePreviewController({
                app,
                node: this,
                imageEl,
                infoEl,
                container,
                imageWidget: this.imageWidget,
                fileWidget,
                indexWidget,
                includeSubdirWidget,
                allWidget,
                updateLayout,
                ensurePreviewLayout,
                resetNodeHeightToBase,
                ensureProvisionalPreviewAspect,
                schedulePreviewStyleSync: () =>
                    scheduleLoadImagePreviewStyleSync(
                        this,
                        applyPreviewHiddenState,
                        [0]
                    ),
            });

            attachCanvasUploadHandlers({
                node: this,
                setDragPassthrough,
                resetDragPassthrough,
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
                    resetDragPassthrough();
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
                } catch {}
            });

            if (fileWidget && fileWidget.value || this._comfy1hewLoadImagePendingPreview) {
                schedulePreviewUpdate(0);
            } else {
                scheduleLoadImagePreviewStyleSync(
                    this,
                    applyPreviewHiddenState,
                    [0, 200]
                );
            }

            scheduleLoadImagePreserveRefresh(this);

            return r;
        };
        installLoadImageClipspacePatch({
            app,
            api,
            saveMaskFromClipspaceToSidecar,
        });

    },
}));
