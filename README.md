```graph TD
    A[Bắt đầu đánh giá] --> B{Tính IoU}
    B --> C[So sánh với ngưỡng Threshold 0.5]
    
    C --> D{Kết quả?}
    D -- "IoU >= 0.5" --> E[True Positive - TP]
    D -- "IoU < 0.5" --> F[False Positive - FP]
    D -- "Bỏ sót vật thể" --> G[False Negative - FN]
    
    E & F & G --> H[Tính Precision & Recall]
    H --> I[Thay đổi Confidence Score]
    I --> J[Vẽ đường cong PR Curve]
    J --> K[Tính AP cho từng lớp]
    K --> L[Tính mAP cho toàn bộ mô hình]
```# bunch-banana-detection
