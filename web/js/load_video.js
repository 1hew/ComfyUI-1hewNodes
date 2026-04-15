import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {VIDEO_FORMATS, resolveComboValue, resolveFormatConfig} from "./core/format_step.js";
import {installWidgetSourceOverlay, roundToPrecision} from "./core/annotated_widget.js";
import { addPreviewMenuOptions, applyPreviewHiddenState } from "./core/preview_menu.js";
import {
    applyLoopedHoverAudioPreview,
    collectDropPayload,
    installVideoPreviewLayout,
} from "./core/media_utils.js";
import { createLoadVideoDom } from "./core/load_video_dom.js";
import { extendLoadVideoMenu } from "./core/load_video_menu.js";
import { scheduleLoadVideoPreviewRestore } from "./core/load_video_node_runtime.js";
import { createLoadVideoPreviewController } from "./core/load_video_preview.js";
import { createLoadVideoUploader } from "./core/load_video_upload.js";
import { installVideoPlaybackState } from "./core/video_playback_state.js";
import { createDomWidgetInteractionManager } from "./core/dom_widget_runtime.js";
import { chainOnRemoved, createDisposer, registerExtensionOnce } from "./core/runtime.js";
import { attachCanvasUploadHandlers, bindDomUploadDropTargets } from "./core/upload_drop_runtime.js";

registerExtensionOnce("__comfy1hewLoadVideoExtensionRegistered", () => app.registerExtension({
    name: "ComfyUI-1hewNodes.LoadVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (
            nodeData.name !== "1hew_LoadVideo" &&
            nodeData.name !== "1hew_LoadVideoToImage"
        ) {
            return;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure
                ? onConfigure.apply(this, arguments)
                : undefined;
            if (this.widgets) {
                const fileWidget = this.widgets.find((w) => w.name === "file");
                if (fileWidget && fileWidget.value) {
                    scheduleLoadVideoPreviewRestore(this);
                }
            }
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const nodeInstance = this;
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;

            const fileWidget = this.widgets.find((w) => w.name === "file");
            const indexWidget = this.widgets.find((w) => w.name === "video_index");
            const includeSubdirWidget = this.widgets.find(
                (w) => w.name === "include_subdir"
            );

            const formatWidget = this.widgets.find((w) => w.name === "format");

            this._comfy1hewVideoInfo = null;
            this._comfy1hewVideoPreviewUserResized = false;
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
                this._comfy1hewLoadVideoBaseSize = [baseW, baseH];
            } catch {}

            // --- Logic ported from VHS initializeLoadFormat ---
            // Allows format selection (e.g. AnimateDiff) to control other widget defaults/constraints
            // and dynamic annotation updates based on frame filtering.
            
            // Note: VIDEO_FORMATS is defined at the top of the file

            const resolveFormatName = (rawValue) => {
                return resolveComboValue(formatWidget, rawValue);
            };

            const fpsWidget = this.widgets.find((w) => w.name === "fps");
            const frameLimitWidget = this.widgets.find(
                (w) => w.name === "frame_limit"
            );
            if (frameLimitWidget) {
                frameLimitWidget._comfy1hew_use_format_step = true;
                // Inject custom step resolver for annotated_widget.js
                // This ensures the widget click handler always uses the authoritative format configuration
                frameLimitWidget._comfy1hew_getStep = () => {
                     const val = formatWidget ? formatWidget.value : "4n+1";
                     const name = resolveFormatName(val);
                     return resolveFormatConfig(name) || { step: 1, mod: 0 };
                };
            }
            if (fpsWidget) {
                const fpsStep =
                    Number(fpsWidget?.options?.step ?? fpsWidget?.step ?? 1) || 1;
                fpsWidget._comfy1hew_use_format_step = false;
                fpsWidget._comfy1hew_force_step = fpsStep;
                delete fpsWidget._comfy1hew_format_config;
                if (fpsWidget.options) {
                    delete fpsWidget.options.mod;
                    fpsWidget.options.step = fpsStep;
                }
                fpsWidget.step = fpsStep;
            }
            
            // --- Helper: Install Widget Source Overlay (VHS Style) ---
            // We install this BEFORE processing format logic so that any modifications 
            // (like .options.reset removal) are captured in the base state if needed.
            // Or, more importantly, so that the widgets are enhanced before we interact with them.
            if (fpsWidget) {
                installWidgetSourceOverlay(this, fpsWidget, () => {
                    const fps = this?._comfy1hewVideoInfo?.fps;
                    return typeof fps === "number" && isFinite(fps) ? fps : null;
                });
                fpsWidget.annotation = (value) => {
                    if (
                        (Number(value) || 0) === 0
                        && this._comfy1hewVideoInfo?.fps !== undefined
                    ) {
                        return (
                            roundToPrecision(this._comfy1hewVideoInfo.fps, 2)
                            + "\u21FD"
                        );
                    }
                    return "";
                };
            }
            if (frameLimitWidget) {
            installWidgetSourceOverlay(
                this,
                frameLimitWidget,
                () => {
                    const n = this?._comfy1hewVideoInfo?.frame_count;
                    return typeof n === "number" && isFinite(n) ? n : null;
                },
                "disable"
            );

            frameLimitWidget.annotation = (value) => {
                    const info = this._comfy1hewVideoInfo;
                    if (!info || !info.frame_count) {
                        return "";
                    }

                    const source_fps = info.fps || 0;
                    const source_count = info.frame_count || 0;

                    // 1. Apply Skip (Subset of original frames)
                    let skip = startSkipWidget ? (startSkipWidget.value || 0) : 0;
                    let end_skip = endSkipWidget ? (endSkipWidget.value || 0) : 0;
                    skip = Number(skip) || 0;
                    end_skip = Number(end_skip) || 0;

                    let subset_count = Math.max(0, source_count - skip - end_skip);

                    // 2. Force Frame Rate (Resampling)
                    let resampled_count = subset_count;
                    const target_fps = Number(fpsWidget ? fpsWidget.value : 0) || 0;

                    if (target_fps > 0 && source_fps > 0 && subset_count > 0) {
                        // Duration logic mirroring backend: (count - 1) / source_fps
                        const duration = (subset_count - 1) / source_fps;
                        resampled_count = Math.floor(duration * target_fps + 1e-9) + 1;
                        resampled_count = Math.max(resampled_count, 1);
                    } else if (target_fps > 0 && source_fps > 0 && subset_count <= 0) {
                         resampled_count = 0;
                    }

                    // 3. Apply Format Constraint
                    const formatName = resolveFormatName(
                        formatWidget ? formatWidget.value : "4n+1"
                    ) || "4n+1";
                    const format = VIDEO_FORMATS[formatName] || VIDEO_FORMATS["4n+1"];
                    
                    const step = format?.frames?.[0] ?? 1;
                    const mod = format?.frames?.[1] ?? 0;
                    
                    let loadable_frames = resampled_count;

                    if (loadable_frames < mod) {
                        loadable_frames = 0;
                    } else {
                        // Calculate max k such that step*k + mod <= count
                        const k = Math.floor((loadable_frames - mod) / step);
                        loadable_frames = step * k + mod;
                        loadable_frames = Math.max(0, loadable_frames);
                    }
                    
                    if (value && value <= loadable_frames) {
                        return "";
                    }

                    return loadable_frames + "\u21FD";
                };

                const originalFrameLimitCallback = frameLimitWidget.callback;
                frameLimitWidget.callback = function () {
                    const args = Array.from(arguments);
                    let v = Number(args[0]);
                    const lastCommitted = Number(
                        this._comfy1hew_last_frame_limit_value
                    );
                    
                    if (Number.isFinite(v)) {
                        let fmtConfig = this._comfy1hew_format_config;
                        
                        if (nodeInstance?.widgets) {
                             const formatWidget = nodeInstance.widgets.find(w => w.name === 'format');
                             if (formatWidget) {
                                 const formatName = resolveComboValue(formatWidget, formatWidget.value);
                                 const cfg = resolveFormatConfig(formatName);
                                 if (cfg) {
                                     fmtConfig = cfg;
                                 }
                             }
                        }

                        if (!fmtConfig && this.options?.mod !== undefined) {
                            fmtConfig = { step: this.options.step, mod: this.options.mod };
                        }

                        const step = fmtConfig?.step ?? this.options?.step ?? 1;
                        const mod = fmtConfig?.mod ?? this.options?.mod ?? 0;
                        
                        if (v > 0 && step) {
                            const currentMod = (v - mod) % step;
                            const isAligned = Math.abs(currentMod) < 0.001 || Math.abs(currentMod - step) < 0.001;

                            // Direction-aware snapping:
                            // when value increases (default right arrow), snap upward;
                            // when value decreases (default left arrow), snap downward.
                            // This keeps xn+1 stepping correct even if widget.mouse is overridden.
                            if (!isAligned) {
                                const q = (v - mod) / step;
                                if (Number.isFinite(lastCommitted)) {
                                    if (v > lastCommitted) {
                                        v = Math.ceil(q - 1e-9) * step + mod;
                                    } else if (v < lastCommitted) {
                                        v = Math.floor(q + 1e-9) * step + mod;
                                    } else {
                                        v = Math.round(q) * step + mod;
                                    }
                                } else {
                                    v = Math.round(q) * step + mod;
                                }
                            }

                            if (v <= 0) {
                                v = 0;
                            }
                        } else if (v < 0) {
                            v = 0;
                        }
                        if (this.options?.min != null && v < this.options.min) {
                            v = this.options.min;
                        }
                        if (this.options?.max != null && v > this.options.max) {
                            v = this.options.max;
                        }
                        this.value = v;
                        this._comfy1hew_last_frame_limit_value = v;
                        args[0] = v;
                    }
                    if (originalFrameLimitCallback) {
                        originalFrameLimitCallback.apply(this, args);
                    }
                    
                    if (nodeInstance && nodeInstance.updateVideoPlaybackState) {
                        nodeInstance.updateVideoPlaybackState();
                    }

                    try {
                        app.graph.setDirtyCanvas(true, true);
                    } catch {}
                };
            }

            if (formatWidget) {
                // Store original widget configurations to restore when switching back to Default
                let baseConfig = {};
                for (let widget of this.widgets) {
                    if (
                        frameLimitWidget
                        && widget.name === frameLimitWidget.name
                    ) {
                        baseConfig[widget.name] = Object.assign({}, widget.options);
                    }
                }

                const applyFormat = (formatName) => {
                    const resolvedName = resolveFormatName(formatName) || "4n+1";
                    const formatConfig = resolveFormatConfig(resolvedName) || resolveFormatConfig("4n+1");
                    if (!formatConfig) return;

                    // Apply constraints/defaults from format
                    for (let widget of this.widgets) {
                        if (widget.name in baseConfig) {
                            // Determine "previous default" to check if user has customized the value
                            // We use a custom property _comfy1hewFormatDefault to track what the code set last time
                            // If not set, we fallback to the reset value in baseConfig (usually 0)
                            const baseReset = baseConfig[widget.name].reset ?? 0;
                            const prevDefault = widget._comfy1hewFormatDefault ?? baseReset;
                            
                            const wasDefault = widget.value == prevDefault;

                            // Determine new target default value
                            let newTarget = baseReset;
                            
                            // Update tracking
                            widget._comfy1hewFormatDefault = newTarget;
                            
                            // Merge base config with format overrides
                            let overrides = {};
                            
                            // Reset format config property first
                            // delete widget._comfy1hew_format_config; // Removed: We set it explicitly above

                            if (
                                frameLimitWidget
                                && widget.name === frameLimitWidget.name
                                && formatConfig.step !== undefined
                            ) {
                                overrides = {
                                    step: formatConfig.step,
                                    mod: formatConfig.mod,
                                    min: 0
                                };
                                widget.step = formatConfig.step;
                                // Explicitly set a custom config object to bypass any LiteGraph resets
                                widget._comfy1hew_step_config = { 
                                    step: formatConfig.step, 
                                    mod: formatConfig.mod 
                                };
                            }

                            // Use Object.assign but ensure we don't revert step if it was just set
                            const mergedOptions = Object.assign({}, baseConfig[widget.name], overrides);
                            if (
                                frameLimitWidget
                                && widget.name === frameLimitWidget.name
                                && formatConfig.step !== undefined
                            ) {
                                mergedOptions.step = formatConfig.step;
                                mergedOptions.mod = formatConfig.mod;
                                mergedOptions.min = 0;
                            }
                            widget.options = mergedOptions;
                            if (
                                frameLimitWidget
                                && widget.name === frameLimitWidget.name
                                && formatConfig.step !== undefined
                            ) {
                                widget.step = formatConfig.step;
                                widget._comfy1hew_step_config = { 
                                    step: formatConfig.step, 
                                    mod: formatConfig.mod 
                                };
                            }
                            
                            // If value was previously at default, update to new default
                            if (wasDefault) {
                                widget.value = newTarget;
                            }
                            if (widget.callback) {
                                widget.callback(widget.value);
                            }
                        }
                    }
                };

                // Hook callback
                const originalFormatCallback = formatWidget.callback;
                formatWidget.callback = function(value) {
                    if (originalFormatCallback) originalFormatCallback.apply(this, arguments);
                    applyFormat(value);
                    if (nodeInstance && nodeInstance.updateVideoPlaybackState) {
                        nodeInstance.updateVideoPlaybackState();
                    }
                    app.graph.setDirtyCanvas(true, true);
                };
                
                // Apply initial state
                applyFormat(formatWidget.value);
            }

            const startSkipWidget = this.widgets.find(
                (w) => w.name === "start_skip"
            );
            const endSkipWidget = this.widgets.find((w) => w.name === "end_skip");
            const frameStepWidget = this.widgets.find((w) => w.name === "frame_step");

            const updatePlayback = () => {
                if (this.updateVideoPlaybackState) {
                    this.updateVideoPlaybackState();
                }
                app.graph.setDirtyCanvas(true, true);
            };

            if (startSkipWidget) {
                 const originalCallback = startSkipWidget.callback;
                 startSkipWidget.callback = function() {
                     if (originalCallback) originalCallback.apply(this, arguments);
                     updatePlayback();
                 };
            }
            if (endSkipWidget) {
                const originalCallback = endSkipWidget.callback;
                endSkipWidget.callback = function() {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updatePlayback();
                };
            }
            if (frameStepWidget) {
                const originalCallback = frameStepWidget.callback;
                frameStepWidget.callback = function() {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updatePlayback();
                };
            }

            if (fpsWidget) {
                const originalFpsCallback = fpsWidget.callback;
                fpsWidget.callback = function() {
                    if (originalFpsCallback) {
                        originalFpsCallback.apply(this, arguments);
                    }
                    updatePlayback();
                };
            }
            // --- End VHS Logic ---

            const openFilePicker = () => {
                try {
                    fileInputEl.value = "";
                } catch {}
                fileInputEl.click();
            };
            const {
                container,
                videoEl,
                infoEl,
                fileInputEl,
            } = createLoadVideoDom({
                app,
                node: this,
                openFilePicker,
            });

            applyLoopedHoverAudioPreview(videoEl);

            const disposables = createDisposer();
            const interaction = createDomWidgetInteractionManager({
                getWidgetElement: () => this.videoWidget?.element,
                container,
                interactiveElements: [videoEl, infoEl],
                idlePointerEvents: "auto",
                dragPointerEvents: "auto",
            });
            const { setDragPassthrough, resetDragPassthrough } = interaction;
            disposables.add(interaction.bindGlobalDragCleanup());

            const { uploadSingleFile, uploadFilesAsFolder } = createLoadVideoUploader({
                app,
                api,
                node: this,
                fileWidget,
                indexWidget,
                infoEl,
            });

            attachCanvasUploadHandlers({
                node: this,
                setDragPassthrough,
                resetDragPassthrough,
                includeDragEnter: true,
                onFileDrop: (file) => {
                    void uploadSingleFile(file);
                },
                onEventDrop: (event) => {
                    if (
                        event.dataTransfer
                        && event.dataTransfer.files
                        && event.dataTransfer.files.length > 0
                    ) {
                        const file = event.dataTransfer.files[0];
                        void uploadSingleFile(file);
                        return true;
                    }
                    return false;
                },
            });

            fileInputEl.addEventListener("change", async () => {
                const file = fileInputEl.files && fileInputEl.files[0];
                await uploadSingleFile(file);
            });

            container.addEventListener("click", (event) => {
                if (event?.target === videoEl) {
                    return;
                }
                try {
                    fileInputEl.value = "";
                } catch {}
                openFilePicker();
            });

            const dragTargets = [container, videoEl, infoEl];
            bindDomUploadDropTargets({
                disposables,
                interaction,
                dragTargets,
                onDrop: async (event) => {
                    const payload = await collectDropPayload(event);
                    const files = (payload?.pairs || []).filter((pair) => pair?.file);
                    if (payload?.hasDirectory || files.length > 1) {
                        await uploadFilesAsFolder(files);
                        return;
                    }
                    await uploadSingleFile(files?.[0]?.file);
                },
            });

            this.videoWidget = this.addDOMWidget(
                "video_preview",
                "div",
                container,
                {
                    serialize: false,
                    hideOnZoom: false,
                }
            );

            this.videoWidget.computeSize = function (width) {
                if (this.aspectRatio) {
                    const maxPreviewHeight =
                        typeof this._comfy1hew_maxPreviewHeight === "number"
                            ? this._comfy1hew_maxPreviewHeight
                            : null;
                    let height = width * this.aspectRatio + 20;
                    if (maxPreviewHeight && isFinite(maxPreviewHeight)) {
                        height = Math.min(height, maxPreviewHeight);
                    }
                    return [width, height];
                }
                return [width, 0];
            };

            this._comfy1hewVideoAutoSizeKey = "";
            const { updateLayout, ensurePreviewLayout } = installVideoPreviewLayout({
                app,
                node: this,
                videoWidget: this.videoWidget,
                container,
                videoEl,
            });

            const resetNodeHeightToBase = () => {
                try {
                    const baseH =
                        Array.isArray(this._comfy1hewLoadVideoBaseSize)
                        && Number.isFinite(this._comfy1hewLoadVideoBaseSize[1])
                            ? this._comfy1hewLoadVideoBaseSize[1]
                            : null;
                    if (typeof baseH === "number" && baseH > 0) {
                        this.setSize([this.size[0], baseH]);
                        return;
                    }
                } catch {}
                try {
                    this.setSize([this.size[0], 130]);
                } catch {}
            };

            videoEl.addEventListener("loadedmetadata", () => {
                const info = this?._comfy1hewVideoInfo;
                const w = Number(info?.width) || 0;
                const h = Number(info?.height) || 0;
                if (w > 0 && h > 0) {
                    infoEl.innerText = `${w} x ${h}`;
                } else {
                    infoEl.innerText = `${videoEl.videoWidth} x ${videoEl.videoHeight}`;
                }
                ensurePreviewLayout({
                    allowShrink: true,
                    forceAutoSize: true,
                });
            });

            videoEl.addEventListener("error", () => {
                this.videoWidget.aspectRatio = undefined;
                infoEl.innerText = "";
                updateLayout();
                resetNodeHeightToBase();
            });

            videoEl.addEventListener("loadeddata", updateLayout);
            videoEl.addEventListener("play", updateLayout);
            videoEl.addEventListener("pause", updateLayout);

            applyPreviewHiddenState(this, { respectFrameAccurateOnShow: true });
            setTimeout(
                () => ensurePreviewLayout({ allowShrink: true, forceAutoSize: true }),
                0
            );
            setTimeout(
                () => ensurePreviewLayout({ allowShrink: true, forceAutoSize: true }),
                200
            );
            setTimeout(
                () => ensurePreviewLayout({ allowShrink: true, forceAutoSize: true }),
                800
            );

            this.updatePreview = createLoadVideoPreviewController({
                app,
                api,
                node: this,
                videoEl,
                infoEl,
                container,
                videoWidget: this.videoWidget,
                fileWidget,
                indexWidget,
                includeSubdirWidget,
                updateLayout,
                ensurePreviewLayout,
                resetNodeHeightToBase,
            });

            const playback = installVideoPlaybackState({
                node: this,
                videoEl,
                widgets: {
                    startSkipWidget,
                    endSkipWidget,
                    fpsWidget,
                    frameLimitWidget,
                    formatWidget,
                },
                resolveComboValue,
            });
            this.updateVideoPlaybackState = playback.updateVideoPlaybackState;

            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                const r2 = originalOnResize
                    ? originalOnResize.apply(this, arguments)
                    : undefined;
                try {
                    if (this.updateVideoLayout) this.updateVideoLayout();
                } catch {}
                return r2;
            };

            if (fileWidget) fileWidget.callback = this.updatePreview;
            if (indexWidget) indexWidget.callback = this.updatePreview;
            if (includeSubdirWidget) includeSubdirWidget.callback = this.updatePreview;

            if (fileWidget && fileWidget.value) {
                this.updatePreview();
            }

            setTimeout(updateLayout, 0);

            chainOnRemoved(this, function () {
                try {
                    resetDragPassthrough();
                    disposables.dispose();
                } catch {}
            });

            return r;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            const r = getExtraMenuOptions
                ? getExtraMenuOptions.apply(this, arguments)
                : undefined;
            extendLoadVideoMenu({ app, api, node: this, options });
            return r;
        };
    },
}));
