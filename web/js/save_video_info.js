import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-1hewNodesV3.SaveVideoInfo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_SaveVideo" && nodeData.name !== "1hew_SaveVideoByImage") {
            return;
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            // 尝试寻找视频元素并添加尺寸显示
            const checkForVideo = () => {
                if (this.widgets) {
                    for (const w of this.widgets) {
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
                                attachDimensionDisplay(videoEl);
                                applyLoopedHoverAudioPreview(videoEl);
                            }
                        }
                    }
                }
            };
            
            // 多次尝试以确保 DOM 已创建
            setTimeout(checkForVideo, 100);
            setTimeout(checkForVideo, 500);
            setTimeout(checkForVideo, 1000);
        };
    }
});

function applyLoopedHoverAudioPreview(videoEl) {
    if (!videoEl || videoEl.dataset.comfy1hewPreviewApplied === "1") {
        return;
    }
    videoEl.dataset.comfy1hewPreviewApplied = "1";

    videoEl.autoplay = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.controls = false;
    videoEl.preload = "auto";

    const safePlay = () => {
        const p = videoEl.play();
        if (p && typeof p.catch === "function") {
            p.catch(() => {});
        }
    };

    safePlay();
    videoEl.addEventListener("loadeddata", safePlay);
    videoEl.addEventListener("canplay", safePlay);

    videoEl.addEventListener("pointerenter", () => {
        videoEl.muted = false;
        videoEl.volume = 1.0;
        safePlay();
    });

    videoEl.addEventListener("pointerleave", () => {
        videoEl.muted = true;
    });
}

function attachDimensionDisplay(videoEl) {
    // 1. 获取或创建容器
    // ComfyUI 的视频通常直接放在 widget.element 中，或者包裹在一个 div 中
    // 我们需要确保视频被包裹在一个 Flex 容器中
    
    let container = videoEl.parentElement;
    if (!container) return;

    // 如果已经处理过，只需要更新尺寸
    if (container.classList.contains("comfy-1hew-video-container")) {
        const display = container.querySelector(".comfy-1hew-video-dimension");
        if (display) {
            updateDimensionText(videoEl, display);
        }
        return;
    }

    // 2. 重构 DOM 结构
    // 创建一个新的 wrapper 来包裹 video，防止破坏原有的 parent 布局
    const wrapper = document.createElement("div");
    wrapper.className = "comfy-1hew-video-container";
    Object.assign(wrapper.style, {
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        height: "100%",
        minHeight: "0", // 关键
        overflow: "hidden"
    });

    // 将 video 移动到 wrapper 中
    // 注意：需要先插入 wrapper，再移动 video
    container.insertBefore(wrapper, videoEl);
    wrapper.appendChild(videoEl);

    // 3. 调整 Video 样式
    Object.assign(videoEl.style, {
        flex: "1 1 auto",
        width: "100%",
        height: "auto", // 让它自然适应 flex 空间
        objectFit: "contain",
        minHeight: "0", // 允许压缩
        display: "block" // 消除 inline 间隙
    });

    // 4. 创建底部信息条
    const display = document.createElement("div");
    display.className = "comfy-1hew-video-dimension";
    Object.assign(display.style, {
        width: "100%",
        height: "20px",
        lineHeight: "20px",
        textAlign: "center",
        fontSize: "12px",
        color: "#aaa",
        fontFamily: "sans-serif",
        flex: "0 0 20px", // 固定高度
        background: "transparent",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis"
    });

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
