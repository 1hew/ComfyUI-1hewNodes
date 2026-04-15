export function createLoadImagePreviewController({
    app,
    node,
    imageEl,
    infoEl,
    container,
    imageWidget,
    fileWidget,
    indexWidget,
    includeSubdirWidget,
    allWidget,
    updateLayout,
    ensurePreviewLayout,
    resetNodeHeightToBase,
    ensureProvisionalPreviewAspect,
    schedulePreviewStyleSync,
}) {
    const normalizeUrl = (src) => {
        if (!src) return "";
        try {
            const url = new URL(src, window.location.href);
            url.searchParams.delete("t");
            url.searchParams.delete("preview");
            return url.toString();
        } catch {
            return String(src);
        }
    };

    return async function updatePreview() {
        node._comfy1hewLoadImageReqId = (node._comfy1hewLoadImageReqId || 0) + 1;
        const reqId = node._comfy1hewLoadImageReqId;

        const file = fileWidget.value;
        const index = indexWidget.value;
        const includeSubdir = includeSubdirWidget
            ? includeSubdirWidget.value
            : true;
        const all = allWidget ? allWidget.value : false;

        if (imageWidget) {
            imageWidget._comfy1hew_maxAspectRatio = all ? 1.25 : null;
            imageWidget._comfy1hew_maxPreviewHeight = all ? 260 : 220;
        }

        const trimmedFile = String(file || "").trim();
        if (trimmedFile === "") {
            node._comfy1hewLoadImageWasEmptyPath = true;
            const lastImgSrc = node._comfy1hewLoadImageLastImgSrc;

            try {
                imageEl.onload = null;
                imageEl.onerror = null;
                imageEl.removeAttribute("src");
            } catch {}
            node._comfy1hewLoadImageHadPreview = false;
            node._comfy1hewImageAutoSizeKey = "";
            imageWidget.aspectRatio = undefined;
            infoEl.innerText = "";
            node._comfy1hewPreserveFrameHeightUntilPreview = undefined;
            updateLayout();
            resetNodeHeightToBase();

            try {
                const clipspace = window?.ComfyApp?.clipspace;
                const clipspaceNode = window?.ComfyApp?.clipspace_return_node;
                const clipspaceImgs = clipspace?.imgs;
                const lastSrcNorm =
                    typeof lastImgSrc === "string"
                        ? normalizeUrl(lastImgSrc)
                        : undefined;

                const shouldClearClipspace =
                    clipspaceNode === node
                    || (typeof lastSrcNorm === "string"
                        && Array.isArray(clipspaceImgs)
                        && clipspaceImgs.some((image) => {
                            if (!image?.src) return false;
                            return normalizeUrl(image.src) === lastSrcNorm;
                        }));

                if (shouldClearClipspace) {
                    window.ComfyApp.clipspace = null;
                    window.ComfyApp.clipspace_return_node = null;

                    const dialog = window?.comfyAPI?.clipspace?.ClipspaceDialog;
                    if (dialog?.instance?.close) {
                        dialog.instance.close();
                    }

                    const maskEditor =
                        window?.comfyAPI?.maskEditorOld?.MaskEditorDialogOld;
                    if (maskEditor?.instance?.close) {
                        maskEditor.instance.close();
                    } else if (maskEditor?.getInstance) {
                        const instance = maskEditor.getInstance();
                        if (instance?.close) {
                            instance.close();
                        }
                    }
                }
            } catch {}
            node._comfy1hewLoadImageLastImgSrc = undefined;

            if (app?.graph?.setDirtyCanvas) {
                app.graph.setDirtyCanvas(true, true);
            }
            return;
        }

        node._comfy1hewLoadImageWasEmptyPath = false;

        const params = new URLSearchParams({
            file,
            include_subdir: includeSubdir,
            t: Date.now(),
        });
        if (!all) {
            params.set("index", index);
        }
        params.set("all", all ? "true" : "false");

        const url = `/1hew/view_image_from_folder?${params.toString()}`;
        const desiredNorm = normalizeUrl(url);
        const currentNorm = normalizeUrl(imageEl.src);

        if (currentNorm !== desiredNorm) {
            node._comfy1hewImageAutoSizeKey = "";
            ensureProvisionalPreviewAspect();

            imageEl.onload = () => {
                if (String(imageEl.dataset.comfy1hewReqId || "") !== String(reqId)) {
                    return;
                }
                if (imageEl.naturalWidth && imageEl.naturalHeight) {
                    imageWidget.aspectRatio = imageEl.naturalHeight / imageEl.naturalWidth;
                    infoEl.innerText = `${imageEl.naturalWidth} x ${imageEl.naturalHeight}`;
                    node._comfy1hewLoadImageHadPreview = true;
                    try {
                        const imageUrl = new URL(imageEl.src);
                        imageUrl.searchParams.delete("t");
                        imageUrl.searchParams.delete("preview");
                        node._comfy1hewLoadImageLastImgSrc = imageUrl.toString();
                    } catch {
                        node._comfy1hewLoadImageLastImgSrc = imageEl.src;
                    }
                    ensurePreviewLayout({
                        allowShrink: true,
                        forceAutoSize: true,
                    });
                    node._comfy1hewPreserveFrameHeightUntilPreview = undefined;
                    schedulePreviewStyleSync();
                }
            };

            imageEl.onerror = () => {
                if (String(imageEl.dataset.comfy1hewReqId || "") !== String(reqId)) {
                    return;
                }
                imageWidget.aspectRatio = undefined;
                infoEl.innerText = "";
                node._comfy1hewLoadImageHadPreview = false;
                node._comfy1hewPreserveFrameHeightUntilPreview = undefined;
                updateLayout();
                resetNodeHeightToBase();
            };

            imageEl.dataset.comfy1hewReqId = String(reqId);
            imageEl.src = url;
            updateLayout();
            return;
        }

        if (imageEl.complete && imageEl.naturalWidth && imageEl.naturalHeight) {
            if (!imageWidget.aspectRatio) {
                imageWidget.aspectRatio = imageEl.naturalHeight / imageEl.naturalWidth;
            }
            infoEl.innerText = `${imageEl.naturalWidth} x ${imageEl.naturalHeight}`;
            node._comfy1hewLoadImageHadPreview = true;
            ensurePreviewLayout({
                allowShrink: true,
                forceAutoSize: true,
            });
            node._comfy1hewPreserveFrameHeightUntilPreview = undefined;
            schedulePreviewStyleSync();
            return;
        }

        const hasSrc =
            Boolean(imageEl?.src && String(imageEl.src).trim() !== "");
        const decoding = hasSrc && (!imageEl.naturalWidth || !imageEl.naturalHeight);
        if (decoding) {
            updateLayout();
        }
    };
}

