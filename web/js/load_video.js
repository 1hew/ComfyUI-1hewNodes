import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {VIDEO_FORMATS, resolveComboValue, resolveFormatConfig} from "./core/format_step.js";
import {installWidgetSourceOverlay, roundToPrecision} from "./core/annotated_widget.js";
import { addPreviewMenuOptions, applyPreviewHiddenState } from "./core/preview_menu.js";
import {
    addCopyMediaFrameMenuOption,
    addSaveMediaMenuOption,
    applyLoopedHoverAudioPreview,
    collectDropPayload,
    installVideoPreviewLayout,
} from "./core/media_utils.js";
import { installVideoPlaybackState } from "./core/video_playback_state.js";

app.registerExtension({
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
                    setTimeout(() => {
                        const update = this.updatePreview;
                        if (update) update();
                    }, 100);
                    setTimeout(() => {
                        this._comfy1hewEnsurePreviewLayout?.();
                    }, 200);
                    setTimeout(() => {
                        this._comfy1hewEnsurePreviewLayout?.();
                    }, 800);
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
                            
                            if (!isAligned) {
                                v = Math.round((v - mod) / step) * step + mod;
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
                        if (v > 0 && step) {
                            if (v < mod) {
                                v = 0;
                            } else {
                                v = Math.floor((v - mod) / step) * step + mod;
                                if (v <= 0) {
                                    v = 0;
                                }
                            }
                            if (this.options?.min != null && v < this.options.min) {
                                v = this.options.min;
                            }
                            // 对齐后可能再次超出上限，必须再钳位一次
                            if (this.options?.max != null && v > this.options.max) {
                                v = this.options.max;
                            }
                        }
                        this.value = v;
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
                    // console.log("[Debug] applyFormat called with", formatName);
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

            const videoEl = document.createElement("video");
            Object.assign(videoEl.style, {
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

            container.appendChild(videoEl);
            container.appendChild(infoEl);
            container.style.cursor = "pointer";

            applyLoopedHoverAudioPreview(videoEl);

            const fileInputEl = document.createElement("input");
            fileInputEl.type = "file";
            fileInputEl.accept = ".mp4,.webm,.mkv,.mov,.avi,video/*";
            fileInputEl.style.display = "none";
            container.appendChild(fileInputEl);

            // Add upload button widget
            const uploadWidget = this.addWidget("button", "choose file to upload", "image", () => {
                app.canvas.node_widget = null; // Clear active widget
                fileInputEl.click();
            });
            uploadWidget.serialize = false;

            const uploadSingleFile = async (file) => {
                if (!file) return;

                infoEl.innerText = "uploading...";
                const form = new FormData();
                form.append("file", file, file.name);

                const res = await api.fetchApi("/1hew/upload_video", {
                    method: "POST",
                    body: form,
                });
                if (res.status !== 200) {
                    infoEl.innerText = "upload failed";
                    return;
                }

                const data = await res.json();
                const newPath = data?.path;
                if (!newPath) {
                    infoEl.innerText = "upload failed";
                    return;
                }

                if (fileWidget) {
                    fileWidget.value = newPath;
                    if (typeof fileWidget.callback === "function") {
                        fileWidget.callback();
                    }
                }
                if (this.updatePreview) {
                    await this.updatePreview();
                }

                app.graph.setDirtyCanvas(true, true);
            };

            const uploadFilesAsFolder = async (pairs) => {
                if (!pairs || pairs.length === 0) return;

                infoEl.innerText = "uploading...";

                const form = new FormData();
                for (const p of pairs) {
                    if (!p?.file) continue;
                    const name = p.relativePath || p.file.name;
                    form.append("files", p.file, name);
                }

                const res = await api.fetchApi("/1hew/upload_videos", {
                    method: "POST",
                    body: form,
                });
                if (res.status !== 200) {
                    infoEl.innerText = "upload failed";
                    return;
                }

                const data = await res.json();
                const folder = data?.folder;
                if (!folder) {
                    infoEl.innerText = "upload failed";
                    return;
                }

                if (fileWidget) {
                    fileWidget.value = folder;
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

                if (this.updatePreview) {
                    await this.updatePreview();
                }

                app.graph.setDirtyCanvas(true, true);
            };

            this.onDropFile = function (file) {
                try {
                    uploadSingleFile(file);
                } catch (err) {
                    console.error("[LoadVideoFromPath] Upload failed:", err);
                }
                return true;
            };

            this.onDragDrop = function (e, graphCanvas) {
                if (
                    e.dataTransfer &&
                    e.dataTransfer.files &&
                    e.dataTransfer.files.length > 0
                ) {
                    const file = e.dataTransfer.files[0];
                    try {
                        uploadSingleFile(file);
                    } catch (err) {
                        console.error("[LoadVideoFromPath] Upload failed:", err);
                    }
                    return true;
                }
                return false;
            };

            this.onDragOver = function (e) {
                if (e.dataTransfer) {
                    e.dataTransfer.dropEffect = "copy";
                }
                return true;
            };

            this.onDragEnter = function (e) {
                return true;
            };

            container.addEventListener("click", (e) => {
                if (e?.target === videoEl) return;
                try {
                    fileInputEl.value = "";
                } catch {}
                fileInputEl.click();
            });

            fileInputEl.addEventListener("change", async () => {
                const file = fileInputEl.files && fileInputEl.files[0];
                await uploadSingleFile(file);
            });

            container.addEventListener("dragover", (e) => {
                e.preventDefault();
            });

            container.addEventListener("dragleave", (e) => {
                e.preventDefault();
            });

            container.addEventListener("drop", async (e) => {
                e.preventDefault();
                e.stopPropagation();

                const payload = await collectDropPayload(e);
                const files = (payload?.pairs || []).filter((p) => p?.file);
                if (payload?.hasDirectory || files.length > 1) {
                    await uploadFilesAsFolder(files);
                    return;
                }

                await uploadSingleFile(files?.[0]?.file);
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
                    return [width, width * this.aspectRatio + 20];
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

            videoEl.addEventListener("loadedmetadata", () => {
                const info = this?._comfy1hewVideoInfo;
                const w = Number(info?.width) || 0;
                const h = Number(info?.height) || 0;
                if (w > 0 && h > 0) {
                    infoEl.innerText = `${w} x ${h}`;
                } else {
                    infoEl.innerText = `${videoEl.videoWidth} x ${videoEl.videoHeight}`;
                }
                ensurePreviewLayout();
            });

            videoEl.addEventListener("error", () => {
                this.videoWidget.aspectRatio = undefined;
                infoEl.innerText = "";
                updateLayout();
                this.setSize([this.size[0], 0]);
            });

            videoEl.addEventListener("loadeddata", updateLayout);
            videoEl.addEventListener("play", updateLayout);
            videoEl.addEventListener("pause", updateLayout);

            applyPreviewHiddenState(this, { respectFrameAccurateOnShow: true });
            setTimeout(ensurePreviewLayout, 0);
            setTimeout(ensurePreviewLayout, 200);
            setTimeout(ensurePreviewLayout, 800);

            this.updatePreview = async () => {
                const file = fileWidget.value;
                const index = indexWidget.value;
                const includeSubdir = includeSubdirWidget.value;

                const trimmedFile = String(file || "").trim();
                if (trimmedFile === "") {
                    this._comfy1hewVideoInfo = null;
                    this.videoWidget.aspectRatio = undefined;
                    infoEl.innerText = "";
                    container.style.display = "none";
                    try {
                        videoEl.pause();
                    } catch {}
                    videoEl.removeAttribute("src");
                    try {
                        videoEl.load();
                    } catch {}
                    updateLayout();
                    this.setSize([this.size[0], 0]);
                    setTimeout(ensurePreviewLayout, 0);
                    return;
                }

                container.style.display = "flex";
                const params = new URLSearchParams({
                    file: trimmedFile,
                    index: index,
                    include_subdir: includeSubdir,
                    preview: "true",
                    t: Date.now(),
                });

                const url = `/1hew/view_video_from_folder?${params.toString()}`;
                if (videoEl.src.indexOf(url.split("&t=")[0]) === -1) {
                    this._comfy1hewVideoAutoSizeKey = "";
                    this.videoWidget.aspectRatio = undefined;
                    infoEl.innerText = "";
                    videoEl.src = url;
                    updateLayout();
                    setTimeout(ensurePreviewLayout, 0);
                    setTimeout(ensurePreviewLayout, 200);
                }

                try {
                    const infoParams = new URLSearchParams({
                        file: trimmedFile,
                        index: index,
                        include_subdir: includeSubdir,
                    });
                    const infoRes = await api.fetchApi(
                        `/1hew/video_info_from_folder?${infoParams.toString()}`,
                        { cache: "no-store" }
                    );
                    if (infoRes && infoRes.status === 200) {
                        this._comfy1hewVideoInfo = await infoRes.json();
                    } else {
                        this._comfy1hewVideoInfo = null;
                    }
                } catch {
                    this._comfy1hewVideoInfo = null;
                }

                const info = this._comfy1hewVideoInfo;
                const w = Number(info?.width) || 0;
                const h = Number(info?.height) || 0;
                if (w > 0 && h > 0) {
                    infoEl.innerText = `${w} x ${h}`;
                    this.videoWidget.aspectRatio = h / w;
                    ensurePreviewLayout();
                }
                
                if (this.updateVideoPlaybackState) {
                    this.updateVideoPlaybackState();
                }

                app.graph.setDirtyCanvas(true, true);
            };

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

            return r;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            addPreviewMenuOptions(options, {
                app,
                currentNode: this,
                respectFrameAccurateOnShow: true,
            });

            let videoEl = null;
            if (this.videoWidget && this.videoWidget.element) {
                 videoEl = this.videoWidget.element.querySelector("video");
                 if (!videoEl && this.videoWidget.element.tagName === "VIDEO") {
                     videoEl = this.videoWidget.element;
                 }
            }

            const getVideoElFromNode = (node) => {
                if (!node?.videoWidget?.element) {
                    return null;
                }
                const el = node.videoWidget.element;
                return el.tagName === "VIDEO" ? el : el.querySelector("video");
            };

            addSaveMediaMenuOption(options, {
                app,
                currentNode: this,
                content: "Save Video",
                getMediaElFromNode: getVideoElFromNode,
                filenamePrefix: "video",
                filenameExt: "mp4",
            });

            if (videoEl) {
                addCopyMediaFrameMenuOption(options, {
                    content: "Copy Frame",
                    getWidth: () => videoEl.videoWidth,
                    getHeight: () => videoEl.videoHeight,
                    drawToCanvas: (ctx) => ctx.drawImage(videoEl, 0, 0),
                    copyErrorMessage: "Failed to copy frame to clipboard:",
                    prepareErrorMessage: "Error preparing frame copy:",
                });
            }
        };
    },
});
