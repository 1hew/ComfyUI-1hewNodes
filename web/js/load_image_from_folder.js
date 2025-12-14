import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-1hewNodes.LoadImageFromFolder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "1hew_LoadImageFromFolder") {
            return;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this.widgets) {
                const folderWidget = this.widgets.find((w) => w.name === "folder");
                if (folderWidget && folderWidget.value) {
                    // 延迟一点执行，确保节点完全初始化
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
            const allWidget = this.widgets.find((w) => w.name === "all");

            this.updatePreview = async () => {
                const folder = folderWidget.value;
                const index = indexWidget.value;
                const includeSubfolder = includeSubfolderWidget.value;
                const all = allWidget ? allWidget.value : false;

                if (!folder) return;

                if (all) {
                    // 批量模式：获取图片列表并加载多张图片，模拟 Preview Image 节点的行为
                    try {
                        const params = new URLSearchParams({
                            folder: folder,
                            include_subfolder: includeSubfolder,
                            return_list: "true"
                        });
                        const res = await api.fetchApi(`/1hew/view_image_from_folder?${params.toString()}`);
                        if (res.status === 200) {
                            const data = await res.json();
                            const count = data.count;
                            
                            // 限制预览数量，防止过多请求
                            const maxPreview = 200;
                            const loadCount = Math.min(count, maxPreview);
                            
                            // 并行加载图片
                            const promises = [];
                            for (let i = 0; i < loadCount; i++) {
                                const p = new Promise((resolve) => {
                                    const img = new Image();
                                    img.onload = () => {
                                        resolve(img);
                                    };
                                    img.onerror = () => {
                                        resolve(null);
                                    }
                                    // 请求单张图片 (all=false)，利用 index 获取特定图片
                                    const imgParams = new URLSearchParams({
                                        folder: folder,
                                        index: i,
                                        include_subfolder: includeSubfolder,
                                        all: "false"
                                    });
                                    img.src = `/1hew/view_image_from_folder?${imgParams.toString()}`;
                                });
                                promises.push(p);
                            }
                            
                            const loadedImgs = await Promise.all(promises);
                            this.imgs = loadedImgs.filter(img => img !== null);
                            
                            // 调整节点尺寸（基于第一张图）
                            if (this.imgs.length > 0) {
                                const img = this.imgs[0];
                                if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                                    const headerHeight = 30;
                                    const widgetMargin = 10;
                                    const widgetHeight = this.widgets.reduce((acc, w) => acc + (w.computeSize ? w.computeSize()[1] : 20), 0) + headerHeight + widgetMargin;
                                    
                                    const availableWidth = this.size[0];
                                    const ratio = img.naturalWidth / img.naturalHeight;
                                    const drawHeight = availableWidth / ratio;
                                    
                                    const totalHeight = widgetHeight + drawHeight + 10;
                                    if (this.size[1] < totalHeight) {
                                        this.setSize([this.size[0], totalHeight]);
                                    }
                                }
                            }
                            
                            app.graph.setDirtyCanvas(true);
                        }
                    } catch (e) {
                        console.error("Failed to load image list", e);
                    }
                } else {
                    // 单张模式
                    // 构建 API URL
                    const params = new URLSearchParams({
                        folder: folder,
                        index: index,
                        include_subfolder: includeSubfolder,
                        all: "false"
                    });
                    
                    const url = `/1hew/view_image_from_folder?${params.toString()}&t=${Date.now()}`;

                    // 创建图片对象并加载
                    const img = new Image();
                    img.onload = () => {
                        this.imgs = [img];
                        
                        // 自动调整节点尺寸
                        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                             const headerHeight = 30;
                             const widgetMargin = 10;
                             const widgetHeight = this.widgets.reduce((acc, w) => acc + (w.computeSize ? w.computeSize()[1] : 20), 0) + headerHeight + widgetMargin;
                             
                             const availableWidth = this.size[0];
                             const ratio = img.naturalWidth / img.naturalHeight;
                             const drawHeight = availableWidth / ratio;
                             
                             const totalHeight = widgetHeight + drawHeight + 10;
                             if (this.size[1] < totalHeight) {
                                 this.setSize([this.size[0], totalHeight]);
                             }
                        }

                        app.graph.setDirtyCanvas(true);
                    };
                    img.src = url;
                }
            };

            // 监听 widget 变化
            if (folderWidget) folderWidget.callback = this.updatePreview;
            if (indexWidget) indexWidget.callback = this.updatePreview;
            if (includeSubfolderWidget) includeSubfolderWidget.callback = this.updatePreview;
            if (allWidget) allWidget.callback = this.updatePreview;

            // 初始加载
            if (folderWidget && folderWidget.value) {
                this.updatePreview();
            }

            return r;
        };
        
        // 移除手动绘制，使用 ComfyUI 原生 this.imgs 机制避免重叠
        // const onDrawForeground = nodeType.prototype.onDrawForeground;
        // nodeType.prototype.onDrawForeground = function (ctx) { ... };
    },
});
