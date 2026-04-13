# Vehicle Detection (OpenCV)

- 使用 **OpenCV** 進行車輛偵測與車流計數專案  
---

#### 專案目標：
- 透過影像前處理與輪廓分析，過濾雜訊並框選車輛  
- 加入計數線，統計通過的車輛數量  

---

#### 專案結構
```
vehicle-detection/
├── data/            # 測試影片、輸出
│   └── videos/
│       ├── cars.mp4
│       └── output.mp4
├── markdown/        # 筆記 (video-detection.md)
├── traffic_count.py # 主程式
└── README.md
```

#### 未來改進
- 加入車輛追蹤功能
- 使用深度學習模型提升準確度
- 支援多車道計數