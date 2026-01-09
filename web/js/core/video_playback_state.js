export function installVideoPlaybackState({
    node,
    videoEl,
    widgets,
    resolveComboValue,
}) {
    const startSkipWidget = widgets?.startSkipWidget;
    const endSkipWidget = widgets?.endSkipWidget;
    const fpsWidget = widgets?.fpsWidget;
    const frameLimitWidget = widgets?.frameLimitWidget;
    const formatWidget = widgets?.formatWidget;

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

        const waitMs = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
                    videoEl.addEventListener("seeked", onSeeked, { once: true });
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
                subsetIndex = Math.round(outputIndex * srcFps / tgtFps);
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
                const tickFps = playbackFps > 0 ? Math.min(playbackFps, 60) : 30;
                await waitMs(1000 / tickFps);
            }
        };

        try {
            videoEl.pause();
        } catch {}
        run().catch(() => {});
    };

    const updateVideoPlaybackState = () => {
        if (!node._comfy1hewVideoInfo || !videoEl) return;

        const info = node._comfy1hewVideoInfo;
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

        const subsetCount = Math.max(0, sourceFrameCount - startSkip - endSkip);

        let resampledCount = subsetCount;
        if (subsetCount <= 0) {
            resampledCount = 0;
        } else if (targetFps > 0) {
            const subsetDuration = (subsetCount - 1) / sourceFps;
            resampledCount = Math.floor(subsetDuration * targetFps + 1e-9) + 1;
            resampledCount = Math.max(resampledCount, 1);
        }

        let formatText = formatWidget ? formatWidget.value : "4n+1";
        if (formatWidget) {
            formatText = resolveComboValue(formatWidget, formatText);
        }
        const fmt = parseFormatSpec(formatText);

        let formatCount = resampledCount;
        if (String(formatText ?? "").trim().toLowerCase() !== "n") {
            if (formatCount < fmt.mod) {
                formatCount = 0;
            } else {
                const k = Math.floor((formatCount - fmt.mod) / fmt.step);
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
                lastSubsetIndex = Math.round(lastOut * sourceFps / targetFps);
            } else {
                lastSubsetIndex = lastOut;
            }
            if (subsetCount > 0) {
                lastSubsetIndex = Math.max(
                    0,
                    Math.min(lastSubsetIndex, subsetCount - 1),
                );
            } else {
                lastSubsetIndex = 0;
            }
        }

        const startSourceIndex = Math.max(0, startSkip);
        const lastSourceIndex = Math.max(startSourceIndex, startSkip + lastSubsetIndex);

        const startTime = startSourceIndex / sourceFps;
        let endTime =
            finalFrameCount > 0 ? (lastSourceIndex + 1) / sourceFps : startTime;

        if (Number.isFinite(duration) && duration > 0) {
            endTime = Math.min(endTime, duration);
        } else if (Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
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
            && (videoEl.currentTime < startTime || videoEl.currentTime > endTime)
        ) {
            try {
                videoEl.currentTime = (startSourceIndex + 0.5) / sourceFps;
            } catch {}
        }

        stopFrameAccuratePreview();
    };

    const onVideoFrame = (now, metadata) => {
        if (!videoEl || !videoEl._comfy1hew_previewConfig) {
            if (videoEl && typeof videoEl.requestVideoFrameCallback === "function") {
                videoEl.requestVideoFrameCallback(onVideoFrame);
            }
            return;
        }

        const start = videoEl._comfy1hew_startTime ?? 0;
        const end = videoEl._comfy1hew_endTime ?? videoEl.duration;

        const sourceFps =
            (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.sourceFps) || 30;
        const tolerance = 1.0 / sourceFps;

        if (end > start && metadata.mediaTime >= end - tolerance) {
            const startSourceIndex =
                (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.startSkip) || 0;

            videoEl.currentTime = (startSourceIndex + 0.5) / sourceFps;

            if (!videoEl.paused) {
                const p = videoEl.play();
                if (p && typeof p.catch === "function") p.catch(() => {});
            }
        }

        if (typeof videoEl.requestVideoFrameCallback === "function") {
            videoEl.requestVideoFrameCallback(onVideoFrame);
        }
    };

    if (typeof videoEl.requestVideoFrameCallback === "function") {
        videoEl.requestVideoFrameCallback(onVideoFrame);
    } else {
        videoEl.addEventListener("timeupdate", () => {
            const start = videoEl._comfy1hew_startTime ?? 0;
            const end = videoEl._comfy1hew_endTime ?? videoEl.duration;

            if (end > start && videoEl.currentTime >= end) {
                const startSourceIndex =
                    (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.startSkip) || 0;
                const sourceFps =
                    (videoEl._comfy1hew_previewConfig && videoEl._comfy1hew_previewConfig.sourceFps) || 30;
                videoEl.currentTime = (startSourceIndex + 0.5) / sourceFps;
                if (!videoEl.paused) {
                    const p = videoEl.play();
                    if (p && typeof p.catch === "function") p.catch(() => {});
                }
            }
        });
    }

    return {
        updateVideoPlaybackState,
        startFrameAccuratePreview,
        stopFrameAccuratePreview,
    };
}

