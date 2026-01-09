async function srcToDataUrl(src) {
    const res = await fetch(src);
    const blob = await res.blob();
    return await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.readAsDataURL(blob);
    });
}

export async function saveMaskFromClipspaceToSidecar({ node, api, app }) {
    const clipspace = window?.ComfyApp?.clipspace;
    const imgs = clipspace?.imgs;
    if (!Array.isArray(imgs) || imgs.length === 0) {
        return;
    }

    const combinedIndex =
        typeof clipspace?.combinedIndex === "number" ? clipspace.combinedIndex : 0;
    const combinedImg = imgs[combinedIndex] || imgs[0];
    const combinedSrc = combinedImg?.src;
    if (!combinedSrc) {
        return;
    }

    const getWidgetValue = (name, fallback) => {
        const w = node?.widgets?.find((x) => x?.name === name);
        return w ? w.value : fallback;
    };

    const path = getWidgetValue("path", "");
    const index = getWidgetValue("index", 0);
    const includeSubdir = getWidgetValue("include_subdir", true);
    const all = Boolean(getWidgetValue("all", false));

    const resolveParams = new URLSearchParams({
        path: path,
        index: index,
        include_subdir: includeSubdir,
        all: all ? "true" : "false",
    });
    const resolved = await api.fetchApi(
        `/1hew/resolve_image_from_folder?${resolveParams.toString()}`,
    );
    if (resolved.status !== 200) {
        return;
    }
    const resolvedJson = await resolved.json();
    const imagePath = resolvedJson?.path;
    if (!imagePath) {
        return;
    }

    const maskDataUrl = await srcToDataUrl(combinedSrc);
    if (!maskDataUrl) {
        return;
    }

    await api.fetchApi("/1hew/save_sidecar_mask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image_path: imagePath,
            mask_data_url: maskDataUrl,
        }),
    });

    try {
        if (node?.updatePreview) {
            await node.updatePreview();
        }
    } catch {}

    try {
        app.graph.setDirtyCanvas(true, true);
    } catch {}
}

