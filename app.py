import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# 설정
IMG_SIZE = (224, 224)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 35  # 과일과 채소 클래스 수 (데이터셋에 맞게 조정)

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3])  # RGB 채널만 사용
])

# 클래스 이름 설정 (실제 데이터셋의 클래스 이름으로 변경 필요)
class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
               'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
               'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 
               'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 
               'soy beans', 'spinach', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# 모델 구축 함수들
def build_mobile_net(num_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

def build_resnet(num_classes):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )
    return model

# 모델 로드 함수
def load_model(model_type='mobilenet', model_path=None):
    if model_type == 'mobilenet':
        model = build_mobile_net(num_classes)
    elif model_type == 'resnet':
        model = build_resnet(num_classes)
    else:
        raise ValueError("지원되지 않는 모델 유형입니다.")
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

# 모델 로드 (경로를 실제 모델 경로로 변경해야 함)
model_path = 'mobilenet_model_regularized_best_acc.pt'  # 모델 파일 경로
model = load_model('mobilenet', model_path)

# 예측 함수
def predict_image(image, model):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_class_index]
        confidence = probabilities[predicted_class_index].item() * 100
    
    # 상위 3개 예측 결과 반환
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_predictions = [
        {
            'class': class_names[idx.item()],
            'probability': prob.item() * 100
        }
        for idx, prob in zip(top_indices, top_probs)
    ]
    
    return predicted_class, confidence, top_predictions

# 라우트 설정
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다'})
    
    try:
        # 이미지 열기
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # 예측
        class_name, confidence, top_predictions = predict_image(img, model)
        
        # 결과 반환
        return jsonify({
            'class_name': class_name,
            'confidence': f'{confidence:.2f}%',
            'top_predictions': top_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)