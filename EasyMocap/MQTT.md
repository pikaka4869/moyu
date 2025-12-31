### MQTT -> 前端展示/保存

- 3D Keypoints保存为json文件
- 本地通过 MQTT 发布 3D Keypoints
  - 安装paho-mqtt库
    ```bash
    pip install paho-mqtt
    ```
  - 发布3D Keypoints
    ```python
    import paho.mqtt.client as mqtt
    import json
    import time
    import os

    class MQTTPublisher:
        def __init__(self, client_id="mqttx_aada30222"):
            # 创建客户端
            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
            except:
                self.client = mqtt.Client(client_id=client_id)
            
            # 连接状态
            self.connected = False
            self.published_count = 0
            
            # 设置回调（兼容版本）
            def on_connect(client, userdata, flags, rc, *args):
                if rc == 0:
                    self.connected = True
                    print(f"[{time.strftime('%H:%M:%S')}] 连接成功")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 连接失败: {rc}")
            
            def on_publish(client, userdata, mid, *args):
                self.published_count += 1
                print(f"[{time.strftime('%H:%M:%S')}] 消息已发布 (ID: {mid}, 总数: {self.published_count})")
            
            def on_disconnect(client, userdata, rc, *args):
                self.connected = False
                print(f"[{time.strftime('%H:%M:%S')}] 连接断开: {rc}")
            
            self.client.on_connect = on_connect
            self.client.on_publish = on_publish
            self.client.on_disconnect = on_disconnect
            
            # 自动重连
            self.client.reconnect_delay_set(1, 30)
        
        def connect(self, broker="broker.emqx.io", port=1883):
            """连接到MQTT服务器"""
            print(f"连接到 {broker}:{port}...")
            try:
                self.client.connect(broker, port, 60)
                self.client.loop_start()
                
                # 等待连接
                for i in range(30):
                    if self.connected:
                        return True
                    time.sleep(0.1)
                
                print("连接超时")
                return False
            except Exception as e:
                print(f"连接异常: {e}")
                return False
        
        
        def publish(self, file_path, topic="3d_keypoints", qos=0):
            """从文件读取并发布数据"""
            if not self.connected:
                print("未连接，无法发布")
                return False
            
            # 读取文件
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在: {file_path}")
                return False
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data is None:
                return False
            
            try:
                # 发布消息
                payload = json.dumps(data)
                print(type(data), type(payload))

                result = self.client.publish(topic, payload, qos)
                
                print(f"发布到Topic: {topic}")
                print(f"数据大小: {len(payload)} bytes")
                
                return result.rc == mqtt.MQTT_ERR_SUCCESS
                
            except Exception as e:
                print(f"发布异常: {e}")
                return False
        
        def disconnect(self):
            """断开连接"""
            self.client.disconnect()
            print("已断开连接")
    ```