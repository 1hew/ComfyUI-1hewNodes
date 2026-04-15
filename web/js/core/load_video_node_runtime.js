export function scheduleLoadVideoPreviewRestore(node) {
    setTimeout(() => {
        const update = node.updatePreview;
        if (update) {
            update();
        }
    }, 100);
    setTimeout(() => {
        node._comfy1hewEnsurePreviewLayout?.({
            allowShrink: true,
            forceAutoSize: true,
        });
    }, 200);
    setTimeout(() => {
        node._comfy1hewEnsurePreviewLayout?.({
            allowShrink: true,
            forceAutoSize: true,
        });
    }, 800);
    setTimeout(() => {
        node._comfy1hewEnsurePreviewLayout?.({
            allowShrink: true,
            forceAutoSize: true,
        });
    }, 1500);
}

