from flask import Flask, request, jsonify, render_template,send_file
import pymysql
import boto3
from botocore.exceptions import NoCredentialsError
import uuid
import requests
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import os

'''
s3와 rds에 티니핑 올리는 코드



app = Flask(__name__)

# AWS S3 설정 (보안 상 중요한 키 값은 환경 변수로 관리하는 것이 좋습니다)
AWS_ACCESS_KEY = 'AKIAU72LGP32CJKZFXAN'
AWS_SECRET_KEY = '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'
BUCKET_NAME = 'tinyping'

# S3 클라이언트 생성
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# AI 모델 엔드포인트 URL
AI_MODEL_URL = "https://your-ai-model-endpoint.com/analyze"

@app.route('/')
def home():
    return render_template('home.html')

def upload_image_to_s3(file, name):
    """ 이미지를 S3에 업로드하고 presigned URL을 반환합니다. """
    try:
        # 파일 확장자 추출 및 고유 파일명 생성
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{name}_{uuid.uuid4()}.{file_extension}"

        # 파일을 S3 버킷에 업로드
        s3_client.upload_fileobj(
            file,
            BUCKET_NAME,
            unique_filename
        )

        # 업로드된 파일의 S3 URL 반환
        file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{unique_filename}"
        return file_url, None  # 정상적인 경우 URL과 None 반환

    except NoCredentialsError:
        return None, "S3 접근 권한이 없습니다."
    except Exception as e:
        return None, str(e)


# RDS 데이터베이스 설정
db = pymysql.connect(
    host='database-1.c1wswg02u84e.ap-southeast-2.rds.amazonaws.com',
    
    user='admin',
    password='wldbsdk3895',
    db='first',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

DATABASE_URL = "mysql+pymysql://<admin>:<wldbsdk3895>@<database-1.c1wswg02u84e.ap-southeast-2.rds.amazonaws.com>:3306/<first>"

# SQLAlchemy 엔진 및 세션 설정
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# 테이블 메타데이터 로드
metadata = MetaData()
tiniping_info = Table('tiniping_info', metadata, autoload_with=engine)

# 이미지 업로드 페이지
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print("Upload endpoint reached")
    if request.method == 'GET':
        # GET 요청 시 업로드 페이지 렌더링
        return render_template('upload.html')
    elif request.method == 'POST':
        # POST 요청 시 파일 업로드 처리
        if 'file' not in request.files:
            return jsonify({'error': 'File is required.'}), 400

        file = request.files['file']
        name = request.form.get('name')

        file_url, error = upload_image_to_s3(file, name or "file")
        if error:
            return jsonify({'error': error}), 500

        if name:
            try:
                with db.cursor() as cursor:
                    sql = "INSERT INTO tinyping (name, image_url) VALUES (%s, %s)"
                    cursor.execute(sql, (name, file_url))
                    db.commit()
                    print("Data inserted successfully and committed.")
                return jsonify({
                    'message': 'TinyPing uploaded successfully!',
                    'name': name,
                    'image_url': file_url
                })
            except Exception as e:
                print(f"Error inserting into database: {str(e)}")  # 오류 메시지 출력
                return jsonify({'error': str(e)}), 500

        return jsonify({'url': file_url})
'''


app = Flask(__name__)


# AI 모델 로컬 URL
AI_MODEL_URL = "http://localhost:5000/analyze"  # AI 모델이 같은 디렉토리에서 실행 중

S3_BUCKET_NAME = 'tinyping'
S3_REGION = 'ap-southeast-2'
RDS_HOST = 'database-1.c1wswg02u84e.ap-southeast-2.rds.amazonaws.com'
RDS_PORT = 3306
RDS_USER = 'admin'
RDS_PASSWORD = 'wldbsdk3895'
RDS_DATABASE = 'first'

AWS_ACCESS_KEY = 'AKIAU72LGP32CJKZFXAN'
AWS_SECRET_KEY = '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'

# S3 클라이언트 설정
s3_client = boto3.client(
    's3',
    region_name=S3_REGION,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# RDS 연결
def get_rds_connection():
    return pymysql.connect(
        host=RDS_HOST,
        user=RDS_USER,
        password=RDS_PASSWORD,
        database=RDS_DATABASE,
        port=RDS_PORT,
        cursorclass=pymysql.cursors.DictCursor
    )

from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import torch  # PyTorch 사용
import pymysql  # RDS 연결용

app = Flask(__name__)

from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image
import pymysql  # RDS 연결용
from torchvision import models

# ResNet50 불러오기
feature_extractor = models.resnet50(pretrained=True)
feature_extractor.fc = torch.nn.Identity()  # 마지막 Fully Connected 레이어 제거

# 입력 데이터 처리 및 특징 추출
def extract_features(image_path, feature_extractor, device):
    input_tensor = preprocess_image(image_path, device)
    with torch.no_grad():
        features = feature_extractor(input_tensor)  # 특징 벡터 추출
        print(f"Extracted features shape: {features.shape}")  # (1, 2048)
    return features


class SimilarityHead(torch.nn.Module):
    def __init__(self, input_dim=3 * 224 * 224, output_dim=15):  # 입력 크기를 3x224x224로 수정
        super(SimilarityHead, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 입력 데이터
        return self.fc(x)

# 모델 로드
MODEL_PTH_PATH = os.path.join(os.getcwd(), 'model.pth')  # 모델 파일 경로
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU/CPU 설정
try:
    model = torch.load(MODEL_PTH_PATH, map_location=device)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")




# 클래스 이름 정의 (모델 학습 시 사용한 클래스 순서와 동일해야 함)
class_names = ['깜빡핑', '꾸래핑', '달콤핑', '덜덜핑', '바로핑', '샤샤핑', '솔찌핑', '아자핑', '악동핑', '조아핑', '차나핑', '투투핑', '포실핑', '하츄핑', '행운핑']

# RDS 연결 설정
def get_rds_connection():
    return pymysql.connect(
        host='database-1.c1wswg02u84e.ap-southeast-2.rds.amazonaws.com',
        user='admin',
        password='wldbsdk3895',
        database='first',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )

# 이미지 전처리 함수
def preprocess_image(image_path, device):
    """
    이미지를 전처리하여 모델 입력에 맞게 변환합니다.
    Args:
        image_path (str): 처리할 이미지 파일 경로.
        device (torch.device): 모델 실행 디바이스 (CPU/GPU).

    Returns:
        torch.Tensor: 모델에 입력할 전처리된 텐서.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet은 224x224 입력 크기를 요구
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # 이미지 로드 및 변환
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 후 디바이스로 이동
    print(f"Preprocessed tensor shape: {input_tensor.shape}")
    return input_tensor

# 예측 함수
def predict_image(image_path, model, class_names, device):
    """
    이미지를 입력받아 클래스 이름을 예측합니다.
    Args:
        image_path (str): 예측할 이미지 경로.
        model (torch.nn.Module): PyTorch 모델.
        class_names (list): 클래스 이름 목록.
        device (torch.device): 실행 디바이스 (CPU/GPU).

    Returns:
        str: 예측된 클래스 이름.
    """
    features = extract_features(image_path, feature_extractor, device)  # 특징 추출
    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted features: {features}")
    
    with torch.no_grad():
        prediction = model(features)
        predicted_index = prediction.argmax(dim=1).item()
        
        probabilities = torch.nn.functional.softmax(prediction, dim=1)  # 확률 계산
        predicted_probability = probabilities[0, predicted_index].item()  # 가장 높은 클래스 확률
        print(f"Predicted class: {class_names[predicted_index]} with probability: {predicted_probability:.2f}")
        
        return class_names[predicted_index]
    
    
    
    
@app.route('/', methods=['GET', 'POST'])
def analyze_photo():
    if request.method == 'GET':
        return render_template('analyze_photo.html')
    
    elif request.method == 'POST':
        if 'photo' not in request.files:
            return jsonify({"error": "사진 파일이 필요합니다."}), 400
        
        photo = request.files['photo']
        if photo.filename == '':
            return jsonify({"error": "파일 이름이 비어 있습니다."}), 400

        # 업로드된 이미지를 저장
        filename = secure_filename(photo.filename)
        filepath = os.path.join('uploads', filename)
        photo.save(filepath)

        try:
            # 모델을 사용하여 예측 수행
            predicted_class_name = predict_image(filepath, model, class_names, device)
            print(f"Predicted class name: {predicted_class_name}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "AI 모델 예측 중 오류 발생", "details": str(e)}), 500

        # RDS 데이터베이스에서 티니핑 정보 검색
        try:
            connection = get_rds_connection()
            print("Database connection successful.")
        except Exception as e:
            print(f"Database connection error: {e}")
            return jsonify({"error": "Database connection failed", "details": str(e)}), 500
        
        try:  
            with connection.cursor() as cursor:
                query = "SELECT name, image_url FROM tinyping WHERE name = %s"
                cursor.execute(query, (predicted_class_name,))
                tiniping_info = cursor.fetchone()
                if not tiniping_info:
                    return jsonify({"error": "RDS에서 티니핑 정보를 찾을 수 없습니다."}), 404
        except Exception as e:
            print(f"Database query error: {e}")
            return jsonify({"error": "Database query failed", "details": str(e)}), 500

        finally:
            connection.close()

        # 결과 반환
        return jsonify({
            "tiniping": {
                "name": tiniping_info['name'],
                "image_url": tiniping_info['image_url']
            }
        })


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
