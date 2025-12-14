import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-1hewNodes.LoadVideoFromFolder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadVideoFromFolder") {
            return;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this.widgets) {
                const folderWidget = this.widgets.find((w) => w.name === "folder");
                if (folderWidget && folderWidget.value) {
                    setTimeout(() => {
                        const update = this.updatePreview;
                        if (update) update();
                    }, 100);
                }
            }
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const folderWidget = this.widgets.find((w) => w.name === "folder");
            const indexWidget = this.widgets.find((w) => w.name === "index");
            const includeSubfolderWidget = this.widgets.find((w) => w.name === "include_subfolder");

            // Create video element
            const videoEl = document.createElement("video");
            videoEl.controls = true;
            videoEl.loop = true;
            Object.assign(videoEl.style, {
                width: "100%",
                height: "auto",
                display: "block",
                margin: "0 auto",
                flex: "1 1 auto",
                objectFit: "contain",
                minHeight: "0"
            });
            
            // Create info element
            const infoEl = document.createElement("div");
            Object.assign(infoEl.style, {
                width: "100%",
                height: "20px",
                lineHeight: "20px",
                textAlign: "center",
                fontSize: "12px",
                color: "#aaa",
                fontFamily: "sans-serif",
                flex: "0 0 20px",
                background: "transparent",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis"
            });
            infoEl.innerText = ""; 

            // Container for the video
            const container = document.createElement("div");
            Object.assign(container.style, {
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                width: "100%",
                height: "auto",
                minHeight: "0",
                overflow: "hidden"
            });
            
            // Wrapper removed, direct append
            container.appendChild(videoEl);
            container.appendChild(infoEl);

            // Add DOM widget
            this.videoWidget = this.addDOMWidget("video_preview", "div", container, {
                serialize: false,
                hideOnZoom: false,
            });
            
            // Ensure widget is sized correctly
            this.videoWidget.computeSize = function(width) {
                 if (this.aspectRatio) {
                     return [width, width * this.aspectRatio + 20];
                 }
                 return [width, 240];
            };

            const updateLayout = () => {
                const ar = this?.videoWidget?.aspectRatio;
                if (!ar) return;

                // Initialize lastSize if needed, but don't return early if we need to resize
                if (!this.lastSize) {
                    this.lastSize = [...this.size];
                }

                // Add extra padding to prevent overflow at the bottom
                const bottomPadding = 5; // Reduced from 30 to 5 to avoid large empty space
                
                // Manual calculation of overhead is more reliable than offsetTop during initial layout
                const headerHeight = 30;
                const widgetMargin = 10; // LiteGraph default margin
                
                // Calculate total height of all other widgets
                const widgetsHeight = (this.widgets || []).reduce((acc, w) => {
                    if (w === this.videoWidget) return acc;
                    // Use computeSize if available, otherwise assume standard height (20)
                    const sz = w.computeSize ? w.computeSize(this.size[0]) : [this.size[0], 20];
                    return acc + (Array.isArray(sz) ? (sz[1] || 20) : 20);
                }, 0);
                
                // Add some extra buffer for LiteGraph's internal padding/spacing
                // Reduced from 15 to 10 to tighten layout while keeping some safety against overflow
                const internalSpacing = 10; 

                const currentOverhead = headerHeight + widgetMargin + widgetsHeight + internalSpacing + bottomPadding;

                const baseOverhead = currentOverhead;
                // Calculate video height based on node width (assuming full width usage)
                const targetVideoHeight = this.size[0] * ar;
                const infoHeight = 20;
                const minWidgetHeight = targetVideoHeight + infoHeight;
                const minTotalHeight = baseOverhead + minWidgetHeight;

                // Smart Resize Logic
                const widthChanged = Math.abs(this.size[0] - this.lastSize[0]) > 1;
                const heightChanged = Math.abs(this.size[1] - this.lastSize[1]) > 1;

                if (widthChanged && !heightChanged) {
                    // Width changed but height didn't (dragged side handle)
                    // Update height to maintain the current vertical padding (or lack thereof)
                    // NewHeight = OldHeight + (NewVideoHeight - OldVideoHeight)
                    const oldVideoHeight = this.lastSize[0] * ar;
                    const heightDelta = targetVideoHeight - oldVideoHeight;
                    const newHeight = Math.max(minTotalHeight, this.size[1] + heightDelta);
                    
                    if (Math.abs(newHeight - this.size[1]) > 1) {
                         this.setSize([this.size[0], newHeight]);
                         // setSize triggers onResize, so we update lastSize here to avoid loops/stale data
                         this.lastSize = [this.size[0], newHeight];
                         // Early return because setSize will trigger updateLayout again
                         return;
                    }
                } else if (this.size[1] < minTotalHeight) {
                    // Height is too small (e.g. initial load or corner drag made it too small)
                    this.setSize([this.size[0], minTotalHeight]);
                    this.lastSize = [this.size[0], minTotalHeight];
                    return;
                }
                
                // Update lastSize
                this.lastSize = [...this.size];
                
                // Subtract baseOverhead (which includes bottomPadding) to get available height for container
                const safeAvailableHeight = this.size[1] - baseOverhead;
                
                // Ensure non-negative height
                container.style.height = `${Math.max(0, safeAvailableHeight)}px`;
                // Also force video to fill container to ensure object-fit works
                videoEl.style.height = "100%";

                this.setDirtyCanvas(true, true);
            };

            videoEl.addEventListener("loadedmetadata", () => {
                infoEl.innerText = `${videoEl.videoWidth} x ${videoEl.videoHeight}`;
                if (videoEl.videoWidth && videoEl.videoHeight) {
                    this.videoWidget.aspectRatio = videoEl.videoHeight / videoEl.videoWidth;
                    setTimeout(updateLayout, 0);
                }
            });

            videoEl.addEventListener("loadeddata", () => setTimeout(updateLayout, 0));
            videoEl.addEventListener("play", () => setTimeout(updateLayout, 0));
            videoEl.addEventListener("pause", () => setTimeout(updateLayout, 0));

            this.updateVideoLayout = updateLayout;

            try {
                const ro = new ResizeObserver(() => updateLayout());
                ro.observe(container);
                this._videoResizeObserver = ro;
            } catch {}

            this.updatePreview = async () => {
                const folder = folderWidget.value;
                const index = indexWidget.value;
                const includeSubfolder = includeSubfolderWidget.value;

                if (!folder) return;

                const params = new URLSearchParams({
                    folder: folder,
                    index: index,
                    include_subfolder: includeSubfolder,
                    t: Date.now()
                });
                
                const url = `/1hew/view_video_from_folder?${params.toString()}`;
                
                // Only update if src changed to avoid flickering
                if (videoEl.src.indexOf(url.split('&t=')[0]) === -1) {
                    videoEl.src = url;
                    setTimeout(updateLayout, 50);
                }
            };

            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                const r2 = onResize ? onResize.apply(this, arguments) : undefined;
                try {
                    if (this.updateVideoLayout) this.updateVideoLayout();
                } catch {}
                return r2;
            };

            // Listen for widget changes
            if (folderWidget) folderWidget.callback = this.updatePreview;
            if (indexWidget) indexWidget.callback = this.updatePreview;
            if (includeSubfolderWidget) includeSubfolderWidget.callback = this.updatePreview;

            // Initial load
            if (folderWidget && folderWidget.value) {
                this.updatePreview();
            }

            // Initialize lastSize for smart resizing
            this.lastSize = undefined;

            return r;
        };
    },
});
