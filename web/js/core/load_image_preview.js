export function createLoadImagePreviewController({
    app,
    node,
    fileWidget,
    indexWidget,
    includeSubdirWidget,
    allWidget,
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

        const trimmedFile = String(file || "").trim();
        if (trimmedFile === "") {
            node._comfy1hewLoadImageWasEmptyPath = true;
            const lastImgSrc = node._comfy1hewLoadImageLastImgSrc;

            try {
                if (node._comfy1hewPreviewImageEl) {
                    node._comfy1hewPreviewImageEl.onload = null;
                    node._comfy1hewPreviewImageEl.onerror = null;
                    node._comfy1hewPreviewImageEl.removeAttribute?.("src");
                }
            } catch {}
            node._comfy1hewLoadImageHadPreview = false;
            node._comfy1hewImageAutoSizeKey = "";
            node._comfy1hewPreviewImageEl = null;
            node.imgs = null;
            node.setDirtyCanvas?.(true, true);

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
        const currentNorm = normalizeUrl(node._comfy1hewLoadImageLastImgSrc);
        if (currentNorm === desiredNorm && Array.isArray(node.imgs) && node.imgs[0]) {
            node._comfy1hewLoadImageHadPreview = true;
            app?.graph?.setDirtyCanvas?.(true, true);
            return;
        }

        node._comfy1hewImageAutoSizeKey = "";
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.decoding = "async";
        img.onload = () => {
            if (reqId !== node._comfy1hewLoadImageReqId) {
                return;
            }
            node._comfy1hewPreviewImageEl = img;
            node.imgs = [img];
            node._comfy1hewLoadImageHadPreview = true;
            try {
                const imageUrl = new URL(img.src);
                imageUrl.searchParams.delete("t");
                imageUrl.searchParams.delete("preview");
                node._comfy1hewLoadImageLastImgSrc = imageUrl.toString();
            } catch {
                node._comfy1hewLoadImageLastImgSrc = img.src;
            }
            try {
                node.setSizeForImage?.();
            } catch {}
            app?.graph?.setDirtyCanvas?.(true, true);
        };
        img.onerror = () => {
            if (reqId !== node._comfy1hewLoadImageReqId) {
                return;
            }
            node._comfy1hewPreviewImageEl = null;
            node.imgs = null;
            node._comfy1hewLoadImageHadPreview = false;
            node._comfy1hewLoadImageLastImgSrc = undefined;
            app?.graph?.setDirtyCanvas?.(true, true);
        };
        img.src = url;
    };
}

