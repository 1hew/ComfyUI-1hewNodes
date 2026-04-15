const LOAD_IMAGE_PRESERVE_REFRESH_LATE_MS = 160;

function hasLoadImagePath(node) {
    const fileWidget = node?.widgets?.find((widget) => widget.name === "file");
    return Boolean(fileWidget && String(fileWidget.value || "").trim());
}

function isLoadImagePreviewHidden(node) {
    return node?.properties?.comfy1hew_preview_hidden === 1;
}

function isLoadImagePreviewStillResolving(node) {
    if (!hasLoadImagePath(node) || isLoadImagePreviewHidden(node)) {
        return false;
    }
    if (node?._comfy1hewLoadImagePendingPreview) {
        return true;
    }

    const imageEl = node?.imageWidget?.element?.querySelector?.("img");
    if (!imageEl) {
        return false;
    }

    const hasSrc = Boolean(imageEl.src && String(imageEl.src).trim() !== "");
    return Boolean(hasSrc && (!imageEl.naturalWidth || !imageEl.naturalHeight));
}

export function installLoadImageSetSizeGuard(nodeType) {
    if (nodeType.prototype._comfy1hewLoadImageSetSizePatched) {
        return;
    }

    nodeType.prototype._comfy1hewLoadImageSetSizePatched = true;
    const originalSetSize = nodeType.prototype.setSize;
    nodeType.prototype.setSize = function (size) {
        const preserveHeight = this._comfy1hewPreserveFrameHeightUntilPreview;
        if (
            !this._comfy1hewInternalPreviewResize
            && !this._comfy1hewInternalVideoPreviewResize
            && Number.isFinite(preserveHeight)
            && Array.isArray(size)
            && size.length >= 2
            && Number.isFinite(size[1])
            && size[1] < preserveHeight - 1
            && isLoadImagePreviewStillResolving(this)
        ) {
            return originalSetSize.call(this, [size[0], preserveHeight]);
        }
        return originalSetSize.apply(this, arguments);
    };
}

export function refreshLoadImagePreserveFrameHeight(node, opts = {}) {
    try {
        if (!isLoadImagePreviewStillResolving(node)) {
            node._comfy1hewPreserveFrameHeightUntilPreview = undefined;
            return;
        }

        const width = Array.isArray(node.size) ? node.size[0] : 200;
        const currentHeight = Array.isArray(node.size) ? node.size[1] : 0;
        const desiredSize = node.computeSize?.([width, currentHeight]);
        const baseHeight =
            Array.isArray(desiredSize) && Number.isFinite(desiredSize[1])
                ? desiredSize[1]
                : 0;

        const serializedSize = opts.serializedSize;
        const serializedHeight =
            Array.isArray(serializedSize)
            && serializedSize.length >= 2
            && Number.isFinite(serializedSize[1])
                ? serializedSize[1]
                : null;

        const heights = [currentHeight, serializedHeight].filter((value) =>
            Number.isFinite(value)
        );
        const frameHeight = heights.length > 0
            ? Math.max(...heights)
            : currentHeight;

        if (
            !Number.isFinite(frameHeight)
            || !Number.isFinite(baseHeight)
            || frameHeight <= baseHeight + 5
        ) {
            return;
        }

        const previous = node._comfy1hewPreserveFrameHeightUntilPreview;
        node._comfy1hewPreserveFrameHeightUntilPreview = Number.isFinite(previous)
            ? Math.max(previous, frameHeight)
            : frameHeight;
        node.updateImageLayout?.();
    } catch {}
}

export function scheduleLoadImagePreserveRefresh(node, opts = {}) {
    refreshLoadImagePreserveFrameHeight(node, opts);
    setTimeout(() => refreshLoadImagePreserveFrameHeight(node, opts), 0);
    setTimeout(
        () => refreshLoadImagePreserveFrameHeight(node, opts),
        LOAD_IMAGE_PRESERVE_REFRESH_LATE_MS
    );
}

export function scheduleLoadImagePreviewStyleSync(
    node,
    applyPreviewHiddenState,
    delays = [0]
) {
    if (!node) {
        return;
    }

    try {
        if (Array.isArray(node._comfy1hewPreviewStyleTimers)) {
            for (const timer of node._comfy1hewPreviewStyleTimers) {
                clearTimeout(timer);
            }
        }
    } catch {}

    node._comfy1hewPreviewStyleTimers = [];
    for (const delay of delays) {
        const timer = setTimeout(() => {
            try {
                node._comfy1hewEnsurePreviewLayout?.({ allowShrink: true });
                applyPreviewHiddenState(node);
            } catch {}
        }, delay);
        node._comfy1hewPreviewStyleTimers.push(timer);
    }
}

