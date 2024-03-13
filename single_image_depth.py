import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Added for interpolation

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

vid = cv2.VideoCapture(r'http://192.168.137.181:81')
frame_no=0
while True:
    ret, frame = vid.read()
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width / 1.5), int(height / 1.5)))
    if frame_no%5==0:
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(frame_gray).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('depth',output_norm)
    cv2.imshow('original', frame)
    print('EXECUTED')
    # cv2.imshow('original',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_no+=1
    

