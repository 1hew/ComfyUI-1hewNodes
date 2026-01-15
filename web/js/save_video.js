import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addPreviewMenuOptions, applyPreviewHiddenState } from "./core/preview_menu.js";

const managedVideos = new Set();
let resumeHooksInstalled = false;

app.registerExtension({
    name: "ComfyUI-1hewNodes.SaveVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_SaveVideo" && nodeData.name !== "1hew_SaveVideoByImage") {
            return;
        }

        const basename = (p) => {
            if (!p) return "";
            const normalized = String(p).replace(/\\/g, "/");
            const parts = normalized.split("/");
            return parts.length ? parts[parts.length - 1] : normalized;
        };

        const extractOriginalPath = (message) => {
            const visit = (v, depth) => {
                if (depth <= 0) return null;
                if (typeof v === "string") {
                    const s = v.trim();
                    if (!s) return null;
                    if (
                        /^[a-zA-Z]:\\/.test(s)
                        || s.startsWith("\\\\")
                        || s.startsWith("/")
                    ) {
                        return s;
                    }
                    return null;
                }
                if (!v || typeof v !== "object") {
                    return null;
                }
                if (typeof v.file_path === "string" && v.file_path.trim()) {
                    return v.file_path.trim();
                }
                if (Array.isArray(v)) {
                    for (const item of v) {
                        const r = visit(item, depth - 1);
                        if (r) return r;
                    }
                    return null;
                }
                for (const key of Object.keys(v)) {
                    const r = visit(v[key], depth - 1);
                    if (r) return r;
                }
                return null;
            };
            return visit(message, 6);
        };

        const applyVideoStyle = (node) => {
            if (node.widgets) {
                for (const w of node.widgets) {
                    if (w.element) {
                        let videoEls = [];
                        if (w.element.tagName === "VIDEO") {
                            videoEls = [w.element];
                        } else {
                            videoEls = Array.from(
                                w.element.querySelectorAll("video")
                            );
                        }

                        for (const videoEl of videoEls) {
                            if (node?._comfy1hewOriginalVideoPath) {
                                videoEl.dataset.comfy1hewOriginalVideoPath =
                                    node._comfy1hewOriginalVideoPath;
                            } else {
                                delete videoEl.dataset.comfy1hewOriginalVideoPath;
                            }
                            attachDimensionDisplay(videoEl);
                            applyLoopedHoverAudioPreview(videoEl);
                        }

                        observeForVideos(w.element);
                    }
                }
            }

            applyPreviewHiddenState(node);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            const originalPath = extractOriginalPath(message);
            if (originalPath) {
                this._comfy1hewOriginalVideoPath = originalPath;
            }

            // 尝试寻找视频元素并添加尺寸显示
            const checkForVideo = () => applyVideoStyle(this);
            
            // 多次尝试以确保 DOM 已创建
            setTimeout(checkForVideo, 100);
            setTimeout(checkForVideo, 500);
            setTimeout(checkForVideo, 1000);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            setTimeout(() => applyVideoStyle(this), 100);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            setTimeout(() => applyVideoStyle(this), 100);
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (onDrawForeground) {
                onDrawForeground.apply(this, arguments);
            }
            if (this.flags.collapsed) return;

            if (this.widgets) {
                let needsApply = false;
                for (const w of this.widgets) {
                    if (w.element && w.element.dataset.comfy1hewVideoObserver !== "1") {
                        needsApply = true;
                        break;
                    }
                }
                if (needsApply) {
                    applyVideoStyle(this);
                }
            }
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            addPreviewMenuOptions(options, { app, currentNode: this });

            // Helper to find videos
            const getVideoEls = (node) => {
                let videoEls = [];
                if (node.widgets) {
                    for (const w of node.widgets) {
                        if (w.element) {
                            if (w.element.tagName === "VIDEO") {
                                videoEls.push(w.element);
                            } else {
                                videoEls.push(...Array.from(w.element.querySelectorAll("video")));
                            }
                        }
                    }
                }
                return videoEls;
            };

            const videoEls = getVideoEls(this);

            // Add "Save Video" option
            if (videoEls.length > 0 && videoEls.some(v => v.src)) {
                options.push({
                    content: "Save Video",
                    callback: () => {
                        const canvas = app.canvas;
                        const selected = canvas.selected_nodes || {};
                        const selection = Object.values(selected);
                        const targets = selection.length > 0 && selection.includes(this) ? selection : [this];

                        for (const node of targets) {
                            const originalPath = node?._comfy1hewOriginalVideoPath;
                            const url = originalPath
                                ? `/1hew/download_video_by_path?path=${encodeURIComponent(originalPath)}&t=${Date.now()}`
                                : null;
                            const suggested = originalPath
                                ? basename(originalPath)
                                : `video_${node.id}_${Date.now()}.mp4`;
                            const nodeVideos = getVideoEls(node);
                            const fallbackSrc = nodeVideos.find((v) => v && v.src)?.src;
                            const a = document.createElement("a");
                            a.href = url || fallbackSrc || "";
                            a.download = suggested;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                        }
                    }
                });
            }

            // Add "Copy Frame" option
            const firstVideo = videoEls.find(v => v.videoWidth && v.videoHeight);
            if (firstVideo) {
                options.push({
                    content: "Copy Frame",
                    callback: async () => {
                        try {
                            const originalPath = this?._comfy1hewOriginalVideoPath;
                            if (originalPath) {
                                const urlParams = new URLSearchParams({
                                    path: originalPath,
                                    t: String(firstVideo.currentTime || 0),
                                    r: String(Date.now()),
                                });
                                const res = await api.fetchApi(
                                    `/1hew/video_frame_by_path?${urlParams.toString()}`,
                                    { cache: "no-store" },
                                );
                                if (res && res.status === 200) {
                                    const blob = await res.blob();
                                    const item = new ClipboardItem({
                                        "image/png": blob,
                                    });
                                    await navigator.clipboard.write([item]);
                                    return;
                                }
                            }

                            const canvas = document.createElement("canvas");
                            canvas.width = firstVideo.videoWidth;
                            canvas.height = firstVideo.videoHeight;
                            const ctx = canvas.getContext("2d");
                            ctx.drawImage(firstVideo, 0, 0);

                            canvas.toBlob(async (blob) => {
                                if (!blob) {
                                    return;
                                }
                                try {
                                    const item = new ClipboardItem({
                                        "image/png": blob,
                                    });
                                    await navigator.clipboard.write([item]);
                                } catch (err) {
                                    console.error(
                                        "Failed to copy frame to clipboard:",
                                        err,
                                    );
                                }
                            }, "image/png");
                        } catch (err) {
                            console.error("Error preparing frame copy:", err);
                        }
                    }
                });
            }
        };
    }
});

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
        safePlay();
    });

    videoEl.addEventListener("pointerleave", () => {
        videoEl.muted = true;
    });
}

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

function observeForVideos(rootEl) {
    if (!rootEl || rootEl.dataset.comfy1hewVideoObserver === "1") {
        return;
    }
    rootEl.dataset.comfy1hewVideoObserver = "1";

    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (!(node instanceof Element)) {
                    continue;
                }
                const videos = [];
                if (node.tagName === "VIDEO") {
                    videos.push(node);
                } else {
                    videos.push(...node.querySelectorAll("video"));
                }
                for (const videoEl of videos) {
                    attachDimensionDisplay(videoEl);
                    applyLoopedHoverAudioPreview(videoEl);
                }
            }
        }
    });

    observer.observe(rootEl, { childList: true, subtree: true });
}

function attachDimensionDisplay(videoEl) {
    // 1. 获取或创建容器
    // ComfyUI 的视频通常直接放在 widget.element 中，或者包裹在一个 div 中
    // 我们需要确保视频被包裹在一个 Flex 容器中
    
    let container = videoEl.parentElement;
    if (!container) return;

    const applyStyles = (wrapperEl, videoElement, displayEl) => {
        Object.assign(wrapperEl.style, {
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            width: "100%",
            height: "100%",
            minHeight: "0",
            overflow: "hidden",
        });

        Object.assign(videoElement.style, {
            flex: "0 0 auto",
            width: "100%",
            height: "auto",
            maxHeight: "calc(100% - 20px)",
            objectFit: "contain",
            minHeight: "0",
            display: "block",
        });

        if (displayEl) {
            Object.assign(displayEl.style, {
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
        }
    };

    // 如果已经处理过，只需要更新尺寸
    if (container.classList.contains("comfy-1hew-video-container")) {
        const display = container.querySelector(".comfy-1hew-video-dimension");
        applyStyles(container, videoEl, display);
        if (display) {
            updateDimensionText(videoEl, display);
        }
        return;
    }

    // 2. 重构 DOM 结构
    // 创建一个新的 wrapper 来包裹 video，防止破坏原有的 parent 布局
    const wrapper = document.createElement("div");
    wrapper.className = "comfy-1hew-video-container";
    applyStyles(wrapper, videoEl, null);

    // 将 video 移动到 wrapper 中
    // 注意：需要先插入 wrapper，再移动 video
    container.insertBefore(wrapper, videoEl);
    wrapper.appendChild(videoEl);

    // 3. 调整 Video 样式
    // 4. 创建底部信息条
    const display = document.createElement("div");
    display.className = "comfy-1hew-video-dimension";
    applyStyles(wrapper, videoEl, display);

    wrapper.appendChild(display);

    // 5. 绑定事件更新尺寸
    const updateDim = () => updateDimensionText(videoEl, display);
    
    videoEl.addEventListener("loadedmetadata", updateDim);
    videoEl.addEventListener("resize", updateDim); // 监听 resize
    
    // 初始触发
    if (videoEl.readyState >= 1) {
        updateDim();
    }
}

function updateDimensionText(videoEl, displayEl) {
    if (videoEl.videoWidth && videoEl.videoHeight) {
        displayEl.textContent = `${videoEl.videoWidth} x ${videoEl.videoHeight}`;
        displayEl.style.display = "block";
    } else {
        displayEl.textContent = "";
        // displayEl.style.display = "none"; // 保持占位，避免抖动
    }
}
