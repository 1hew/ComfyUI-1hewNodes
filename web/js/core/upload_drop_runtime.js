export function attachCanvasUploadHandlers({
    node,
    setDragPassthrough,
    resetDragPassthrough,
    onFileDrop,
    onEventDrop,
    includeDragEnter = false,
}) {
    node.onDropFile = function (file) {
        resetDragPassthrough();
        try {
            onFileDrop?.(file);
        } catch (err) {
            console.error("[1hewNodes.upload_drop_runtime] file drop failed:", err);
        }
        return true;
    };

    node.onDragDrop = function (event) {
        resetDragPassthrough();
        try {
            const result = onEventDrop?.(event);
            return result !== false;
        } catch (err) {
            console.error("[1hewNodes.upload_drop_runtime] event drop failed:", err);
            return false;
        }
    };

    node.onDragOver = function (event) {
        setDragPassthrough(true);
        if (event?.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
        return true;
    };

    if (includeDragEnter) {
        node.onDragEnter = function () {
            return true;
        };
    }
}

export function bindDomUploadDropTargets({
    disposables,
    interaction,
    dragTargets,
    onDrop,
}) {
    const handleDomDragEnter = (event) => {
        if (!event) {
            return;
        }
        event.preventDefault();
        interaction.dragDepth += 1;
        interaction.setDragPassthrough(true);
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
    };

    const handleDomDragOver = (event) => {
        if (!event) {
            return;
        }
        event.preventDefault();
        interaction.setDragPassthrough(true);
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
    };

    const handleDomDragLeave = (event) => {
        if (!event) {
            return;
        }
        event.preventDefault();
        interaction.dragDepth -= 1;
        if (interaction.dragDepth === 0) {
            interaction.setDragPassthrough(false);
        }
    };

    const handleDomDrop = async (event) => {
        if (!event) {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        interaction.resetDragPassthrough();
        await onDrop?.(event);
    };

    disposables.add(
        interaction.bindDragTargets(dragTargets, {
            dragenter: handleDomDragEnter,
            dragover: handleDomDragOver,
            dragleave: handleDomDragLeave,
            drop: handleDomDrop,
        })
    );
}

