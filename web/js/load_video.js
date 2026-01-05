import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {VIDEO_FORMATS, resolveComboValue, resolveFormatConfig} from "./core/format_step.js";
import {installWidgetSourceOverlay, roundToPrecision} from "./core/annotated_widget.js";
import { addPreviewMenuOptions, applyPreviewHiddenState } from "./core/preview_menu.js";

const managedVideos = new Set();
let resumeHooksInstalled = false;

function installResumeHooks() {
    if (resumeHooksInstalled) {
        return;
    }
    resumeHooksInstalled = true;

    const resumeAll = () => {
        for (const videoEl of managedVideos) {
            if (!videoEl) {
                continue;
            }
            if (videoEl.dataset.comfy1hewUserPaused === "1") {
                continue;
            }
            const p = videoEl.play();
            if (p && typeof p.catch === "function") {
                p.catch(() => {});
            }
        }
    };

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            return;
        }
        resumeAll();
    });
    window.addEventListener("focus", resumeAll);
    window.addEventListener("pageshow", resumeAll);
}

function applyLoopedHoverAudioPreview(videoEl) {
    if (!videoEl || videoEl.dataset.comfy1hewPreviewApplied === "1") {
        return;
    }
    videoEl.dataset.comfy1hewPreviewApplied = "1";

    installResumeHooks();
    managedVideos.add(videoEl);

    videoEl.autoplay = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.controls = false;
    videoEl.preload = "auto";

    const safePlay = () => {
        if (videoEl.dataset.comfy1hewUserPaused === "1") {
            return;
        }
        if (videoEl.dataset.comfy1hewFrameAccurate === "1") {
            return;
        }
        const p = videoEl.play();
        if (p && typeof p.catch === "function") {
            p.catch(() => {});
        }
    };

    safePlay();
    videoEl.addEventListener("loadeddata", safePlay);
    videoEl.addEventListener("canplay", safePlay);
    videoEl.addEventListener("playing", () => managedVideos.add(videoEl));

    videoEl.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (videoEl.dataset.comfy1hewFrameAccurate === "1") {
            if (videoEl.dataset.comfy1hewUserPaused === "1") {
                videoEl.dataset.comfy1hewUserPaused = "0";
                if (videoEl.dataset.comfy1hewForceMute !== "1") {
                    videoEl.muted = false;
                    videoEl.volume = 1.0;
                }
            } else {
                videoEl.dataset.comfy1hewUserPaused = "1";
                videoEl.muted = true;
            }
            return;
        }
        if (videoEl.paused) {
            videoEl.dataset.comfy1hewUserPaused = "0";
            if (videoEl.dataset.comfy1hewForceMute !== "1") {
                videoEl.muted = false;
                videoEl.volume = 1.0;
            }
            safePlay();
        } else {
            videoEl.dataset.comfy1hewUserPaused = "1";
            videoEl.pause();
        }
    });

    videoEl.addEventListener("pointerenter", () => {
        if (videoEl.dataset.comfy1hewUserPaused === "1") {
            return;
        }
        if (videoEl.dataset.comfy1hewForceMute === "1") {
            videoEl.muted = true;
        } else {
            videoEl.muted = false;
            videoEl.volume = 1.0;
        }
        if (videoEl.dataset.comfy1hewFrameAccurate === "1") {
            return;
        }
        safePlay();
    });

    videoEl.addEventListener("pointerleave", () => {
        videoEl.muted = true;
    });
}

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
                const pathWidget = this.widgets.find((w) => w.name === "path");
                if (pathWidget && pathWidget.value) {
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

            const pathWidget = this.widgets.find((w) => w.name === "path");
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
                display: "flex",
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
            const uploadWidget = this.addWidget("button", "choose video to upload", "image", () => {
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

                if (pathWidget) {
                    pathWidget.value = newPath;
                    if (typeof pathWidget.callback === "function") {
                        pathWidget.callback();
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

                if (pathWidget) {
                    pathWidget.value = folder;
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

                if (this.updatePreview) {
                    await this.updatePreview();
                }

                app.graph.setDirtyCanvas(true, true);
            };

            const readAllEntries = async (dirEntry) => {
                const reader = dirEntry.createReader();
                const entries = [];
                while (true) {
                    const batch = await new Promise((resolve) =>
                        reader.readEntries(resolve)
                    );
                    if (!batch || batch.length === 0) break;
                    entries.push(...batch);
                }
                return entries;
            };

            const walkEntry = async (entry) => {
                if (!entry) return [];
                if (entry.isFile) {
                    const file = await new Promise((resolve) =>
                        entry.file(resolve)
                    );
                    const relativePath = (entry.fullPath || file.name)
                        .replace(/^\/+/, "")
                        .trim();
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
                        if (entry.isDirectory) {
                            hasDirectory = true;
                        }
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

            const autoSizeToContent = () => {
                if (!this.videoWidget.aspectRatio) {
                    return;
                }
                
                const width = this.size[0];
                // 重新计算理想高度：宽度 * 比例 + 底部文字高度(20)
                const desiredWidgetHeight = width * this.videoWidget.aspectRatio + 20;
                
                // 获取 widget 的位置，计算总高度
                let desiredHeight;
                if (Number.isFinite(this.videoWidget.last_y)) {
                    desiredHeight = this.videoWidget.last_y + desiredWidgetHeight;
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
                        let estimatedTop = 130;
                        if (this.widgets && this.widgets.length > 0) {
                            const widgetCount = this.widgets.filter(
                                (w) => w !== this.videoWidget && w.type !== "hidden"
                            ).length;
                            // 估算: 标题栏(30) + (Widget高度(20)+间距(6)) * 数量 + 底部留白
                            estimatedTop = 30 + widgetCount * 26 + 30;
                        }
                        if (estimatedTop < 130) estimatedTop = 130;
                        desiredHeight = estimatedTop + desiredWidgetHeight;
                    }
                }

                // 如果当前高度小于理想高度，自动撑开
                if (this.size[1] + 1 < desiredHeight) {
                    this.setSize([this.size[0], desiredHeight]);
                }
            };

            const requestAutoSize = () => {
                if (!videoEl.videoWidth || !videoEl.videoHeight) {
                    return;
                }
                const key = `${videoEl.videoWidth}x${videoEl.videoHeight}`;
                if (this._comfy1hewVideoAutoSizeKey === key) {
                    return;
                }
                this._comfy1hewVideoAutoSizeKey = key;
                setTimeout(autoSizeToContent, 0);
            };

            const updateLayout = () => {
                if (container.dataset.comfy1hewForceHidden === "1") {
                    container.style.height = "0px";
                    container.style.display = "none";
                    return;
                }
                if (!this.videoWidget.aspectRatio) {
                    container.style.height = "0px";
                    container.style.display = "none";
                    return;
                }
                container.style.display = "flex";

                // 1. 尝试使用 last_y 准确计算起始位置
                // last_y 是 LiteGraph 计算出的当前 widget 的 y 坐标
                let availableHeight;
                if (Number.isFinite(this.videoWidget?.last_y)) {
                    // 节点高度 - widget起始位置 - 底部留白 (15px)
                    availableHeight = this.size[1] - this.videoWidget.last_y - 15;
                } else {
                    const width = this.size[0];
                    availableHeight = width * this.videoWidget.aspectRatio + 20;
                }

                // 3. 确保高度有效
                if (availableHeight < 0) availableHeight = 0;

                // 4. 设置容器高度
                container.style.height = `${availableHeight}px`;
                
                // 5. 强制重绘，确保 LiteGraph 更新布局
                app.graph.setDirtyCanvas(true, true);
            };

            const ensurePreviewLayout = () => {
                if (container.dataset.comfy1hewForceHidden === "1") {
                    updateLayout();
                    return;
                }

                if (videoEl.videoWidth && videoEl.videoHeight) {
                    this.videoWidget.aspectRatio =
                        videoEl.videoHeight / videoEl.videoWidth;
                }

                requestAutoSize();
                updateLayout();
                autoSizeToContent();
            };

            videoEl.addEventListener("loadedmetadata", () => {
                infoEl.innerText = `${videoEl.videoWidth} x ${videoEl.videoHeight}`;
                if (videoEl.videoWidth && videoEl.videoHeight) {
                    this.videoWidget.aspectRatio =
                        videoEl.videoHeight / videoEl.videoWidth;
                    requestAutoSize();
                    updateLayout();
                }
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

            this.updateVideoLayout = updateLayout;
            this._comfy1hewEnsurePreviewLayout = ensurePreviewLayout;
            applyPreviewHiddenState(this, { respectFrameAccurateOnShow: true });
            setTimeout(ensurePreviewLayout, 0);
            setTimeout(ensurePreviewLayout, 200);
            setTimeout(ensurePreviewLayout, 800);

            this.updatePreview = async () => {
                const path = pathWidget.value;
                const index = indexWidget.value;
                const includeSubdir = includeSubdirWidget.value;

                if (!path) {
                    this._comfy1hewVideoInfo = null;
                    this.videoWidget.aspectRatio = undefined;
                    infoEl.innerText = "";
                    videoEl.removeAttribute("src");
                    updateLayout();
                    this.setSize([this.size[0], 0]);
                    setTimeout(ensurePreviewLayout, 0);
                    return;
                }

                const params = new URLSearchParams({
                    path: path,
                    index: index,
                    include_subdir: includeSubdir,
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
                        path: path,
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
                
                if (this.updateVideoPlaybackState) {
                    this.updateVideoPlaybackState();
                }

                app.graph.setDirtyCanvas(true, true);
            };

            // --- Helper: Update Video Playback State ---
            const parseFormatSpec = (raw) => {
                const f = String(raw ?? "").trim().toLowerCase();
                if (!f || f === "n" || f === "default") {
                    return { step: 1, mod: 0 };
                }
                if (!f.includes("n")) {
                    return { step: 1, mod: 0 };
                }
                const parts = f.split("n");
                let step = 1;
                let mod = 0;
                try {
                    if (parts[0]) {
                        step = Number.parseInt(parts[0], 10);
                    }
                    if (parts.length > 1 && parts[1]) {
                        mod = Number.parseInt(parts[1], 10);
                    }
                } catch {
                    return { step: 1, mod: 0 };
                }
                if (!Number.isFinite(step) || step <= 0) {
                    step = 1;
                }
                if (!Number.isFinite(mod)) {
                    mod = 0;
                }
                return { step, mod };
            };

            const stopFrameAccuratePreview = () => {
                const runner = videoEl?._comfy1hewFrameAccurateRunner;
                if (runner && typeof runner.stop === "function") {
                    runner.stop();
                }
                if (videoEl) {
                    videoEl._comfy1hewFrameAccurateRunner = null;
                    videoEl.dataset.comfy1hewFrameAccurate = "0";
                }
            };

            const startFrameAccuratePreview = () => {
                if (!videoEl) {
                    return;
                }

                const cfg = videoEl._comfy1hew_previewConfig;
                if (!cfg || !cfg.enabled) {
                    stopFrameAccuratePreview();
                    return;
                }

                if (videoEl._comfy1hewFrameAccurateRunner) {
                    videoEl._comfy1hewFrameAccurateRunner.stop();
                    videoEl._comfy1hewFrameAccurateRunner = null;
                }

                videoEl.dataset.comfy1hewFrameAccurate = "1";
                const token = { stopped: false };

                const waitMs = (ms) =>
                    new Promise((resolve) => setTimeout(resolve, ms));

                const seekTo = (t) =>
                    new Promise((resolve) => {
                        let timeoutId = null;
                        const onSeeked = () => {
                            if (timeoutId) {
                                clearTimeout(timeoutId);
                            }
                            resolve();
                        };
                        timeoutId = setTimeout(() => {
                            try {
                                videoEl.removeEventListener("seeked", onSeeked);
                            } catch {}
                            resolve();
                        }, 250);
                        try {
                            videoEl.addEventListener("seeked", onSeeked, {
                                once: true,
                            });
                            videoEl.currentTime = t;
                        } catch {
                            clearTimeout(timeoutId);
                            resolve();
                        }
                    });

                const clampTime = (t) => {
                    const d = Number(videoEl.duration);
                    if (Number.isFinite(d) && d > 0) {
                        if (t < 0) return 0;
                        if (t > d) return d;
                        return t;
                    }
                    return t;
                };

                const computeSourceFrameIndex = (outputIndex, config) => {
                    const count = Number(config.subsetCount) || 0;
                    if (count <= 0) {
                        return Number(config.startSkip) || 0;
                    }

                    let subsetIndex = outputIndex;
                    const srcFps = Number(config.sourceFps) || 0;
                    const tgtFps = Number(config.targetFps) || 0;
                    if (srcFps > 0 && tgtFps > 0) {
                        subsetIndex = Math.round(
                            outputIndex * srcFps / tgtFps
                        );
                    }
                    subsetIndex = Math.max(0, Math.min(subsetIndex, count - 1));
                    return (Number(config.startSkip) || 0) + subsetIndex;
                };

                const runner = {
                    stop: () => {
                        token.stopped = true;
                    },
                };
                videoEl._comfy1hewFrameAccurateRunner = runner;

                const run = async () => {
                    let outputIndex = 0;
                    while (!token.stopped) {
                        const config = videoEl._comfy1hew_previewConfig;
                        if (!config || !config.enabled) {
                            break;
                        }

                        if (videoEl.dataset.comfy1hewUserPaused === "1") {
                            await waitMs(100);
                            continue;
                        }

                        const frameCount = Number(config.finalFrameCount) || 0;
                        if (frameCount <= 0) {
                            const startT = clampTime(Number(config.startTime) || 0);
                            await seekTo(startT);
                            await waitMs(150);
                            continue;
                        }

                        if (outputIndex >= frameCount) {
                            outputIndex = 0;
                        }

                        const srcIdx = computeSourceFrameIndex(outputIndex, config);
                        const fps = Number(config.sourceFps) || 0;
                        const t = fps > 0 ? (srcIdx + 0.5) / fps : 0;
                        await seekTo(clampTime(t));

                        outputIndex += 1;

                        const playbackFps = Number(config.playbackFps) || 0;
                        const tickFps =
                            playbackFps > 0 ? Math.min(playbackFps, 60) : 30;
                        await waitMs(1000 / tickFps);
                    }
                };

                try {
                    videoEl.pause();
                } catch {}
                run().catch(() => {});
            };

            const updateVideoPlaybackState = () => {
                if (!this._comfy1hewVideoInfo || !videoEl) return;

                const info = this._comfy1hewVideoInfo;
                const sourceFps = Number(info.fps) || 0;
                const duration = Number(info.duration) || 0;
                let sourceFrameCount = Number(info.frame_count) || 0;

                const hasControls =
                    !!startSkipWidget
                    || !!endSkipWidget
                    || !!fpsWidget
                    || !!frameLimitWidget
                    || !!formatWidget;

                if (!hasControls) {
                    stopFrameAccuratePreview();
                    return;
                }

                if (sourceFps <= 0) {
                    stopFrameAccuratePreview();
                    return;
                }

                if (sourceFrameCount === 0 && duration > 0) {
                    sourceFrameCount = Math.round(duration * sourceFps);
                }

                const startSkip = Number(startSkipWidget?.value) || 0;
                const endSkip = Number(endSkipWidget?.value) || 0;
                const frameLimit = Number(frameLimitWidget?.value) || 0;
                const targetFps = Number(fpsWidget?.value) || 0;

                const subsetCount = Math.max(
                    0,
                    sourceFrameCount - startSkip - endSkip
                );

                let resampledCount = subsetCount;
                if (subsetCount <= 0) {
                    resampledCount = 0;
                } else if (targetFps > 0) {
                    const subsetDuration = (subsetCount - 1) / sourceFps;
                    resampledCount =
                        Math.floor(subsetDuration * targetFps + 1e-9) + 1;
                    resampledCount = Math.max(resampledCount, 1);
                }

                let formatText = formatWidget ? formatWidget.value : "4n+1";
                if (formatWidget) {
                    formatText = resolveComboValue(formatWidget, formatText);
                }
                const fmt = parseFormatSpec(formatText);

                let formatCount = resampledCount;
                if (
                    String(formatText ?? "")
                        .trim()
                        .toLowerCase() !== "n"
                ) {
                    if (formatCount < fmt.mod) {
                        formatCount = 0;
                    } else {
                        const k = Math.floor(
                            (formatCount - fmt.mod) / fmt.step
                        );
                        formatCount = fmt.step * k + fmt.mod;
                        formatCount = Math.max(0, formatCount);
                    }
                }

                let finalFrameCount = formatCount;
                if (frameLimit > 0) {
                    finalFrameCount = Math.min(finalFrameCount, frameLimit);
                }
                finalFrameCount = Math.max(0, finalFrameCount);

                let lastSubsetIndex = 0;
                if (finalFrameCount > 0) {
                    const lastOut = finalFrameCount - 1;
                    if (targetFps > 0 && subsetCount > 0) {
                        lastSubsetIndex = Math.round(
                            lastOut * sourceFps / targetFps
                        );
                    } else {
                        lastSubsetIndex = lastOut;
                    }
                    if (subsetCount > 0) {
                        lastSubsetIndex = Math.max(
                            0,
                            Math.min(lastSubsetIndex, subsetCount - 1)
                        );
                    } else {
                        lastSubsetIndex = 0;
                    }
                }

                const startSourceIndex = Math.max(0, startSkip);
                const lastSourceIndex = Math.max(
                    startSourceIndex,
                    startSkip + lastSubsetIndex
                );

                const startTime = startSourceIndex / sourceFps;
                let endTime =
                    finalFrameCount > 0
                        ? (lastSourceIndex + 1) / sourceFps
                        : startTime;

                if (Number.isFinite(duration) && duration > 0) {
                    endTime = Math.min(endTime, duration);
                } else if (
                    Number.isFinite(videoEl.duration)
                    && videoEl.duration > 0
                ) {
                    endTime = Math.min(endTime, videoEl.duration);
                }
                if (endTime < startTime) {
                    endTime = startTime;
                }

                const playbackFps = targetFps > 0 ? targetFps : sourceFps;

                videoEl._comfy1hew_startTime = startTime;
                videoEl._comfy1hew_endTime = endTime;
                videoEl.playbackRate = 1.0;
                videoEl._comfy1hew_previewConfig = {
                    enabled: true,
                    sourceFps,
                    targetFps,
                    playbackFps,
                    sourceFrameCount,
                    subsetCount,
                    startSkip,
                    endSkip,
                    finalFrameCount,
                    startTime,
                    endTime,
                };

                if (
                    Number.isFinite(videoEl.currentTime)
                    && (videoEl.currentTime < startTime
                        || videoEl.currentTime > endTime)
                ) {
                    try {
                        videoEl.currentTime = (startSourceIndex + 0.5) / sourceFps;
                    } catch {}
                }

                stopFrameAccuratePreview();
            };

            // Use requestVideoFrameCallback for frame-accurate looping if available
            // Fallback to timeupdate if not supported
            const onVideoFrame = (now, metadata) => {
                if (!videoEl || !videoEl._comfy1hew_previewConfig) {
                     if (videoEl && typeof videoEl.requestVideoFrameCallback === 'function') {
                         videoEl.requestVideoFrameCallback(onVideoFrame);
                     }
                     return;
                }

                const start = videoEl._comfy1hew_startTime ?? 0;
                const end = videoEl._comfy1hew_endTime ?? videoEl.duration;
                
                // metadata.mediaTime provides the presentation timestamp in seconds
                // Use a slightly tighter tolerance than timeupdate
                // Use 1.0 frame tolerance to aggressively avoid rendering the next frame
                const sourceFps = (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.sourceFps) || 30;
                const tolerance = 1.0 / sourceFps;
                
                if (end > start && metadata.mediaTime >= (end - tolerance)) {
                    const startSourceIndex = (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.startSkip) || 0;
                    
                    // Jump back
                    videoEl.currentTime = (startSourceIndex + 0.5) / sourceFps;
                    
                    // Ensure it keeps playing if not paused by user
                    if (!videoEl.paused) {
                        const p = videoEl.play();
                        if (p && typeof p.catch === "function") p.catch(() => {});
                    }
                }

                // Schedule next frame check
                if (typeof videoEl.requestVideoFrameCallback === 'function') {
                    videoEl.requestVideoFrameCallback(onVideoFrame);
                }
            };

            if (typeof videoEl.requestVideoFrameCallback === 'function') {
                videoEl.requestVideoFrameCallback(onVideoFrame);
            } else {
                // Fallback for browsers without rVFC support
                videoEl.addEventListener("timeupdate", () => {
                    const start = videoEl._comfy1hew_startTime ?? 0;
                    const end = videoEl._comfy1hew_endTime ?? videoEl.duration;
                    
                    if (end > start && videoEl.currentTime >= end) {
                        const startSourceIndex = (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.startSkip) || 0;
                        const sourceFps = (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.sourceFps) || 30;
                        videoEl.currentTime = (startSourceIndex + 0.5) / sourceFps;
                        if (!videoEl.paused) {
                            const p = videoEl.play();
                            if (p && typeof p.catch === "function") p.catch(() => {});
                        }
                    }
                });
            }

            this.updateVideoPlaybackState = updateVideoPlaybackState;

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

            if (pathWidget) pathWidget.callback = this.updatePreview;
            if (indexWidget) indexWidget.callback = this.updatePreview;
            if (includeSubdirWidget) includeSubdirWidget.callback = this.updatePreview;

            if (pathWidget && pathWidget.value) {
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
        };
    },
});
