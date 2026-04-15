export function registerExtensionOnce(windowKey, register) {
    if (window[windowKey]) {
        return false;
    }
    window[windowKey] = true;
    register();
    return true;
}

export function chainOnRemoved(node, cleanup) {
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        try {
            cleanup?.call(this);
        } catch {}
        if (typeof originalOnRemoved === "function") {
            return originalOnRemoved.apply(this, arguments);
        }
    };
}

export function createDisposer() {
    const cleanups = [];

    return {
        add(cleanup) {
            if (typeof cleanup === "function") {
                cleanups.push(cleanup);
            }
            return cleanup;
        },
        dispose() {
            while (cleanups.length > 0) {
                const cleanup = cleanups.pop();
                try {
                    cleanup?.();
                } catch {}
            }
        },
    };
}

