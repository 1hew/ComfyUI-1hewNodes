function setElementPointerEvents(element, pointerEvents) {
    if (element?.style) {
        element.style.pointerEvents = pointerEvents;
    }
}

export function createDomWidgetInteractionManager({
    getWidgetElement,
    container,
    interactiveElements = [],
    idlePointerEvents = "none",
    dragPointerEvents = "auto",
}) {
    const elements = [container, ...interactiveElements];
    let dragDepth = 0;

    const setDragPassthrough = (active) => {
        const pointerEvents = active ? dragPointerEvents : idlePointerEvents;
        for (const element of elements) {
            setElementPointerEvents(element, pointerEvents);
        }
        setElementPointerEvents(getWidgetElement?.(), pointerEvents);
    };

    const resetDragPassthrough = () => {
        dragDepth = 0;
        setDragPassthrough(false);
    };

    const handleGlobalDragCleanup = () => {
        resetDragPassthrough();
    };

    const bindGlobalDragCleanup = () => {
        window.addEventListener("dragend", handleGlobalDragCleanup, true);
        window.addEventListener("drop", handleGlobalDragCleanup, true);
        return () => {
            window.removeEventListener("dragend", handleGlobalDragCleanup, true);
            window.removeEventListener("drop", handleGlobalDragCleanup, true);
        };
    };

    const bindDragTargets = (targets, handlers) => {
        for (const target of targets) {
            target.addEventListener("dragenter", handlers.dragenter);
            target.addEventListener("dragover", handlers.dragover);
            target.addEventListener("dragleave", handlers.dragleave);
            target.addEventListener("drop", handlers.drop);
        }
        return () => {
            for (const target of targets) {
                target.removeEventListener("dragenter", handlers.dragenter);
                target.removeEventListener("dragover", handlers.dragover);
                target.removeEventListener("dragleave", handlers.dragleave);
                target.removeEventListener("drop", handlers.drop);
            }
        };
    };

    return {
        get dragDepth() {
            return dragDepth;
        },
        set dragDepth(value) {
            dragDepth = Math.max(0, Number(value) || 0);
        },
        setDragPassthrough,
        resetDragPassthrough,
        bindGlobalDragCleanup,
        bindDragTargets,
    };
}

