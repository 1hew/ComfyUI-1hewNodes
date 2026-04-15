function collectPastedImageFiles(event) {
    const items = event?.clipboardData?.items;
    if (!items) {
        return [];
    }

    const files = [];
    for (let i = 0; i < items.length; i++) {
        if (items[i].kind === "file" && items[i].type.startsWith("image/")) {
            const file = items[i].getAsFile();
            if (file) {
                files.push({
                    file,
                    relativePath: file.name || "pasted_image.png",
                });
            }
        }
    }
    return files;
}

const managedLoadImageNodes = new Set();
let globalLoadImagePasteInstalled = false;

function installGlobalLoadImagePasteHandler(app) {
    if (globalLoadImagePasteInstalled) {
        return;
    }
    globalLoadImagePasteInstalled = true;

    document.addEventListener(
        "paste",
        (event) => {
            const files = collectPastedImageFiles(event);
            if (files.length === 0) {
                return;
            }

            const selectedNodes = app?.canvas?.selected_nodes || {};
            const candidates = Array.from(managedLoadImageNodes).filter(
                (node) =>
                    selectedNodes[node?.id]
                    && typeof node?._comfy1hewHandlePasteFiles === "function"
            );
            if (candidates.length === 0) {
                return;
            }

            const targetNode = candidates[candidates.length - 1];
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            targetNode._comfy1hewHandlePasteFiles(files);
        },
        { capture: true }
    );
}

export function registerLoadImagePasteTarget({ app, node, handlePasteFiles }) {
    node._comfy1hewHandlePasteFiles = handlePasteFiles;
    managedLoadImageNodes.add(node);
    installGlobalLoadImagePasteHandler(app);

    return () => {
        managedLoadImageNodes.delete(node);
        node._comfy1hewHandlePasteFiles = null;
    };
}

export function installLoadImageClipspacePatch({
    app,
    api,
    saveMaskFromClipspaceToSidecar,
}) {
    if (window.__comfy1hewLoadImageClipspacePatched) {
        return;
    }
    window.__comfy1hewLoadImageClipspacePatched = true;

    const install = () => {
        const comfyApp = window?.ComfyApp;
        if (!comfyApp) {
            return;
        }

        const originalPaste = comfyApp.pasteFromClipspace;
        if (typeof originalPaste !== "function") {
            return;
        }
        if (comfyApp.__comfy1hewPasteFromClipspaceWrapped) {
            return;
        }

        comfyApp.__comfy1hewPasteFromClipspaceWrapped = true;
        comfyApp.pasteFromClipspace = function () {
            const result = originalPaste.apply(this, arguments);
            setTimeout(() => {
                const node = window?.ComfyApp?.clipspace_return_node;
                if (node?.type !== "load_image") {
                    return;
                }
                saveMaskFromClipspaceToSidecar({ node, api, app });
            }, 0);
            return result;
        };
    };

    setTimeout(install, 0);
}

