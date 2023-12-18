# Gait-extraction-for-posture-analysis
Using Yolov8 for both human segmentation and joints extraction for posture analysis

GPU: RTX 2080 for accelerated inference of Yolov8 models

2 Yolov8 model (Yolov8n-pose for joints extraction and Yolov8n-seg for human segmentation) were used for inference human in videos to infer and collect their joints coordinate and segmenting their silhouette. This could be useful for posture and gait analysis. Note that tracking of the person is also performed. While tracking is unnecessary for a single person video strem, it would be necessary if multiple people are present and effct of interactions between each tracked object on gait are required to be studied.  



https://github.com/jhng83/Gait-extraction-for-posture-analysis/assets/23288373/d0db7a74-4be0-448a-8c44-289947f7272a

