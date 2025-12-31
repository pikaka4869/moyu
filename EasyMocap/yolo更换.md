### 更换yolo模型
只需传入模型路径参数<br>
由于代码中使用了 YOLOv5 的 API ，后来的yolo模型（如yolov8）并不适用，还需更改detect()函数的输出格式<br>
后续发现问题：在可视化阶段，检测到边界框数组为空，但代码并未特判，直接访问，导致数组越界报错。加入特判数组是否为空部分即可<br>
***yolo github仓库[](https://github.com/ultralytics/assets/releases/tag/v8.3.0)***