import { addPreviewMenuOptions } from "./preview_menu.js";

function getVideoElementFromNode(node) {
    if (!node?.videoWidget?.element) {
        return null;
    }
    const element = node.videoWidget.element;
    return element.tagName === "VIDEO" ? element : element.querySelector("video");
}

function getVideoMenuTargets(app, currentNode) {
    const canvas = app.canvas;
    const selected = canvas.selected_nodes || {};
    const selection = Object.values(selected);
    if (selection.length > 0 && selection.includes(currentNode)) {
        return selection;
    }
    return [currentNode];
}

function basename(pathValue) {
    if (!pathValue) return "";
    const normalized = String(pathValue).replace(/\\/g, "/");
    const parts = normalized.split("/");
    return parts.length ? parts[parts.length - 1] : normalized;
}

function getVideoNodeParams(node) {
    const fileWidget = node?.widgets?.find((widget) => widget.name === "file");
    const indexWidget = node?.widgets?.find((widget) => widget.name === "video_index");
    const includeSubdirWidget = node?.widgets?.find(
        (widget) => widget.name === "include_subdir"
    );
    const file = String(fileWidget?.value || "").trim();
    if (!file) {
        return null;
    }

    return {
        file,
        index: String(indexWidget?.value ?? "0"),
        includeSubdir: String(includeSubdirWidget?.value ?? true),
    };
}

export function extendLoadVideoMenu({ app, api, node, options }) {
    addPreviewMenuOptions(options, {
        app,
        currentNode: node,
        respectFrameAccurateOnShow: true,
    });

    const videoEl = getVideoElementFromNode(node);

    options.push({
        content: "Save Video",
        callback: () => {
            const targets = getVideoMenuTargets(app, node);
            for (const targetNode of targets) {
                const params = getVideoNodeParams(targetNode);
                if (!params) {
                    continue;
                }
                const urlParams = new URLSearchParams({
                    file: params.file,
                    index: params.index,
                    include_subdir: params.includeSubdir,
                    raw: "1",
                    t: Date.now(),
                });
                const url = `/1hew/view_video_from_folder?${urlParams.toString()}`;
                const suggested =
                    basename(targetNode?._comfy1hewVideoInfo?.path)
                    || `video_${targetNode.id}_${Date.now()}.mp4`;
                const anchor = document.createElement("a");
                anchor.href = url;
                anchor.download = suggested;
                document.body.appendChild(anchor);
                anchor.click();
                document.body.removeChild(anchor);
            }
        },
    });

    if (!videoEl) {
        return;
    }

    options.push({
        content: "Copy Frame",
        callback: async () => {
            const params = getVideoNodeParams(node);
            if (!params) {
                return;
            }
            const urlParams = new URLSearchParams({
                file: params.file,
                index: params.index,
                include_subdir: params.includeSubdir,
                t: String(videoEl.currentTime || 0),
                r: String(Date.now()),
            });
            try {
                const res = await api.fetchApi(
                    `/1hew/video_frame_from_folder?${urlParams.toString()}`,
                    { cache: "no-store" }
                );
                if (!res || res.status !== 200) {
                    return;
                }
                const blob = await res.blob();
                const item = new ClipboardItem({ "image/png": blob });
                await navigator.clipboard.write([item]);
            } catch (err) {
                console.error("Failed to copy frame to clipboard:", err);
            }
        },
    });
}

