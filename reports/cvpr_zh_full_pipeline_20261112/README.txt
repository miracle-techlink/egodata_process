完整Pipeline 中文CVPR技术报告 · 手部与6D原理配图修订版

本版补齐前半段感知：视频/时间轴 → 双手21点 → 物体6D与表面运动 → 坐标与样本 → 图编码 → 时序预测 → 分层验证及RSI。
完整技术报告_手部与6D配图版.pdf 为明确命名的最新报告（与main.pdf字节一致），14页、18幅编号图。新增三张矢量原理图：手部21点/坐标转换、6D刚体变换/FoundationPose、双目三角化/跨帧刚体拟合。
手部与6D_三页配图预览.pdf 为三页直接看图版（与visual_check.pdf字节一致），将三张原理图与真实演示排在一起。
demo_materials.pdf 为逐阶段图文索引。真实图、指标验证图、公式和来源已放入正文。

直接查看
- 手部原理图与第一视角双手：第3页；真人手物与四目标：第4页。
- 6D变换/跟踪原理图与点云：第5页。
- 双目三角化/刚体拟合原理图与真实表面种子：第6页。
- 手部覆盖和标定复核：第7页；矩阵检查、逐帧延迟和感知性能：第8页。
- 样本时间轴：第9页；实际RSI逐轮曲线与骨干结果：第12页。
最终页码与素材清单.csv、evidence/figure_page_index.json对照。

视频在media/，不嵌入PDF。共十个MP4：原有六版交互预测，另有第一视角手部坐标、真人手物联合、HO-Cap四物体、完整6D后端四段。
推荐先看 hands_ego_coordinates.mp4、hands_and_object_rgbd.mp4、hocap_four_objects.mp4、demo_v6.mp4。
pose6d_backend_full.mp4含机械臂，为独立FoundationPose后端测试，不作为真人手或第一视角主线。
视频规格与SHA见evidence/pipeline_video_inventory.json；全部完整解码检查通过。

已直接读取服务器
- 取回训练配置、数据索引、benchmark和三个RSI系列记录；没有重新训练或更改原数据/模型。
- 扩展v4的test micro-F1精确值：LSTM/Transformer 0.9388532000951701，Mamba 0.9052002985817368，1326窗口、120维输入。
- rsi_v5 / rsi_auto / rsi_final分别8/8/30条账本记录，唯一已评分候选6/6/21；另有一个final候选目录无score。
- rsi_final使用132维和fasttrain，当前genome为两层64维LSTM，保存val F1为0.9885057471；最高候选val F1为1.0，不自动等于晋升或独立测试。
- 用户确认最终全量约0.93。本次未发现三个*_final或rsi_final中的test benchmark，仍需与用户指定的最终运行唯一绑定，不能仅凭数值近似认定历史0.938853就是最新冠军。

指标边界
- 手部/物体覆盖、旋转矩阵自洽、标定像素一致性与处理时延已用原始数据绘图。
- 独立手部、物体平移/旋转精度没有对应完整评测结果，明确写为未测；不把44/44或737/737称为准确率。
- 同帧二维叠加不自动等于米制三维融合；MediaPipe相对Z不是米制相机深度。
- 现有demo_v6仍有中文方框，正式重渲染需原始输入与最终checkpoint绑定。没有用生成图替换实验画面。

复现
bash build.sh
需要XeLaTeX、BibTeX、TeX Live的Fandol字体。Overleaf上传ZIP，选main.tex与XeLaTeX；素材册选demo_materials.tex。
静态图全部随包，编译不需要服务器、网络、GPU或Python。
重新计算指标图：python3 scripts/make_pipeline_figures.py，依赖numpy、matplotlib及Noto Sans CJK SC；读取随包evidence/。
cvpr.sty与ieeenat_fullname.bst未修改，模板源：https://github.com/cvpr-org/author-kit。

来源
evidence/pipeline_asset_sources.json：逐个感知文件和原图来源/哈希。
evidence/remote/：本次SSH读取的小型记录快照；数据根历史上可能被重新导出，必须使用run原本绑定信息解释。
evidence/remote_rsi_verification.json：候选统计、可取得的晋升日志和checkpoint SHA。
evidence/rsi_observed_curve.csv：图中逐条RSI数值。
evidence/perception/：感知原始JSON/CSV/NPZ。

本版为私有工作区报告。未公开发布视频/数据，未commit或push。旧七页稿保留作为前版，本版是完整pipeline修订。
