# P-Quiz
main keywords : image captioning , dart , flutter , flask , Vit+GPT2 model , inception resnet v2 + Transformers

<img width="100" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/eab5b9f8-8dcc-4ff8-8a06-40d59b70292b">

## Team member
|<img src="https://avatars.githubusercontent.com/onsemiro11" width="100">|<img src="https://avatars.githubusercontent.com/goflvhxj" width="100">|<img src="https://avatars.githubusercontent.com/hanajibsa" width="100">|<img src="https://avatars.githubusercontent.com/vvinnit" width="100">|  
|-|-|-|-|
|[이현도](https://github.com/onsemiro11)|[박지영](https://github.com/goflvhxj)|[김지원](https://github.com/hanajibsa)|[이승열](https://github.com/vvinnit)|
  
image captioning을 활용해서 주변환경의 이미지에서 관련 문장을 생성하여 영어 학습 문제를 만드는 프로젝트

## 개발 배경
현대 사회에서 영어는 필수적인 언어로 자리매김하고 있지만, 많은 사람들이 영어 공부를 시작하기에 망설이거나 어려움을 겪는다. 특히 초보자들은 언어의 복잡한 구조와 문법, 발음 등으로 인해 영어 학습을 시작하기 어려워한다. 뿐만 아니라 시험 중심의 학습으로 인해 실제 일상 생활에서의 영어 활용에 난해함을 겪는 사람들이 많다.
영어 학습을 시작하기 어려움을 느끼는 분들을 위해 더 영어 학습의 문턱을 낮추고, 학습의 출발을 도와주기 위한 방법을 고민하였다.

영어영문학과 신동일 교수는 " 인물과 상황을 사실적으로 묘사하는 것은 일상생활에서도 꼭필요한 언어 능력이다. 하지만 사람들이 묘사 활동을 제대로 배울 기회는 많지 않으며 그 필요성 조차 잘 인식하지 못하고 있다. " 고 말했다.

다른 나라 언어를 배울 때 가장 직관적으로 배울 수 있는 방법에 대해 생각해보면, 주변의 상황과 인물을 묘사하는 것은 가장 기초적이면서도 꼭 필요한 부분이다. 또한 영어 공부를 쉽사리 시작하지 못하는 사람들도 자신의 실제 상황부터시작을 한다면 더 흥미를 가지고, 쉽게 영어 학습을 시작할 수 있을 것 입니다. 따라서 저희 조는 AI 기술을 활용한 상황 묘사 앱을 고안하였다. 


## 앱 개발
frontend : Flutter , Dart

backend : AWS , Flask

<img width="300" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/37e385c2-098c-402e-9731-b9de31717e4d">

## 차별점

개인 맞춤 환경 학습 : 일상에서 실제 상황에 적용할 수 있게 하여 접근성과 참요도를 높일 수 있다.

시각적 학습 : 추상적인 개념을 시각화하여 학습 이해도와 효율을 높일 수 있다.

게임적 요소 : 점수와 위젯을 표시하여 동기를 부여할 수 있다.

## UI image
![image](https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/d67eebeb-e050-47d3-b790-3f9f30c4d90e) ![image](https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/372aa1de-c04c-430b-82d0-b7e4a1ff644a) ![image](https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/b48a8e24-1d5a-4da3-8cdc-973fad2fcec8)


## Survice Flow
<img width="600" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/f2558068-79b4-4b6a-aece-29cd9fc5094d">

## Work Flow
<img width="600" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/e54a0f0a-167f-497d-8ba4-0cc0048b4552">

## Image Captioning Model Training

### Introduce Model
image captioning : 이미지를 입력값으로 받으면 해당 이미지에 적합한 text를 생성하는 기술을 말한다.

Encoder Decoder 구조
- Encoder : image 에서 feature 추출(CNN 계열 or ViT...)
- Decoder : 추출된 feature로부터 sequence 문장 생성 (LSTM, RNN, Transformer...)

<img width="600" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/7348567c-032b-4ef7-8d81-bef81226af4a">

### Train Data
상황 묘사에 가장 적합한 것은 인물이 포함된 상황을 묘사한 이미지로 정의하였다.

> 위 정의는 영어영문학과 신동일 교수의 "인물과 상황을 사실적으로 묘사하는 것은 일상생활에서도 꼭필요한 언어 능력이다." 말씀을 근거로 정의를 내렸다.

- HC-COCO dataset : 인간 중심의 코코 데이터셋이고 캡션의 70%이상이 인간 행동에 초점을 맞췄고 캡션의 49% 이상이 인간과 사물의 상호작용에 초점을 맞춘 데이터다. 본 데이터는 image 16,125개 , caption 78,462개로 구성돼있다.
- Flickr 8k dataset : 이미지 캡션의 대표 데이터셋이다. 다양한 장면과 상황 묘사에 적합한 데이터로 주요 객체 및 이벤트에 대한 명확한 설명을 제공한다. image 8,091개 , caption 40,455개로 구성돼있다.

flickr 8k dataset을 포함해서 모델을 돌린 것과 인간 중심의 데이터만 학습시킨 것 중에 어떤 모델이 더 좋은 성능을 보여주는지 실험해보려 한다.


### Data Preprocessing
<img width="802" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/155d7cf1-26a7-4be3-8b19-08d107ebbf7a">


### Augmentation

조금 더 일반화된 성능을 내기 위해 증강 기법을 활용하여 데이터를 늘렸다.

우선 사진을 촬영하게 된다면 다양한 환경의 이미지가 입력될 것이다.

특히 흐림과 밝기의 차이와 좌우반전의 차이가 클 것으로 예상되어 이 세가지의 요소를 변경하여 증강시켰다.

### Model Train

Inception ResNet v3 와 Transformer 로 구성된 모델과 ViT 와 GPT2 로 구성된 모델 두개를 학습시켜 평가지표를 비교해봤다.

[Inception ResNet v2 + Transformer]

<img width="720" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/4a36e5cc-7ed8-4c44-a337-6b57a7fb50ca">

[ViT + GPT2]

<img width="741" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/9c836233-358e-4653-9d76-4ec3a8e96d43">


### Rouge Score
- Rouge2_Precision : Label을 구성하는 단어 중 몇 개 가 inference와 겹치는지 평가 , 우선적으로 필요한 정보들이 담겨있는지 확인 가능
- Rouge2_Recall : inferenc를 구성하는 단어 중 몇 개가 Label과 겹치는지 평가 , 요약된 문장에 필요한 정보를 담고 있는지 확인 가능

<img width="741" alt="image" src="https://github.com/onsemiro11/P-Quiz_APP_image_captioning/assets/49609175/547d26fc-9787-412b-91d6-f0a2ee5296fd">

=> 최종적으로 inception resnet v2 + Transformer(HC-COCO_Augmentation) 모델이 다른 모델에 비해 비교적 높은 평가지표를 보여주었기에 본 model을 선택하였다.

## 결론

ViT 논문에서도 나왔듯이 아직 vision부분에서 Transformer는 많은 데이텅에서 성능이 높이 나오는 것 같다.

본 프로젝트와 같이 적은 데이터로 성능을 내려고 할 시에는 cnn계열의 모델들이 더 잘 feature를 추출하는 것을 확인하였다.

최종적으로 본 프로젝트에서는 inception resNet v2 + Transformer 구조의 모델에서 얻은 weight를 사용하여 앱을 구현했다.

## 발전 방향

1. 인물 외에 다양한 상황과 주변 환경 묘사 기술 개발에 기여한다.
2. 언어 번역 기능과 단어별 학습 기능을 추가하여 학습효과를 증대할 수 있다.
3. 퀴즈를 사용자의 역량에 맞게 차등을 두어 난이도 선택 기능을 영어 학습 전문과와 함께 개발할 수 있다.
