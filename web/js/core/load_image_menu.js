import {
    addCopyMediaFrameMenuOption,
    addSaveMediaMenuOption,
} from "./media_utils.js";

export function extendLoadImageMenu({ app, node, options }) {
    if (!Array.isArray(options)) {
        return;
    }

    for (let i = options.length - 1; i >= 0; i--) {
        const option = options[i];
        if (
            option
            && typeof option.content === "string"
            && option.content.trim() === "Save Mask"
        ) {
            options.splice(i, 1);
        }
    }

    const imageEl = node?._comfy1hewPreviewImageEl || null;
    addSaveMediaMenuOption(options, {
        app,
        currentNode: node,
        content: "Save Image",
        getMediaElFromNode: (currentNode) => currentNode?._comfy1hewPreviewImageEl,
        filenamePrefix: "image",
        filenameExt: "png",
    });

    if (imageEl?.src) {
        addCopyMediaFrameMenuOption(options, {
            content: "Copy Image",
            getWidth: () => imageEl.naturalWidth,
            getHeight: () => imageEl.naturalHeight,
            drawToCanvas: (ctx) => ctx.drawImage(imageEl, 0, 0),
            copyErrorMessage: "Failed to copy image to clipboard:",
            prepareErrorMessage: "Error preparing image copy:",
        });
    }
}

